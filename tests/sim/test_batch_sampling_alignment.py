"""Downsampling must retain serial timestamps, controls and the actual terminal state."""

import pytest

from rosclaw.sim.backends.mujoco.backend import MujocoBackend

XML = """<mujoco><option gravity="0 0 0" timestep=".002"/>
<worldbody><body><joint name="j" damping="1"/><geom size=".1" mass="1"/>
</body></worldbody><actuator><motor joint="j"/></actuator>
<keyframe><key name="home" time="2" qpos=".1"/></keyframe></mujoco>"""


@pytest.mark.parametrize("steps", [5, 6, 500])
@pytest.mark.parametrize("initial_time", [0, 2])
def test_batch_recorded_samples_and_controls_match_serial_including_final(
    tmp_path, steps, initial_time
):
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model_xml(
        XML.replace('time="2"', f'time="{initial_time}"'), source={"kind": "fixture"}
    ).model_ref
    state = backend.initial_state_v2(ref, keyframe="home")
    controller = {"ctrl_series": [[step / steps] for step in range(steps)]}
    budgets = {"max_record_points": 2 if steps < 10 else 480}
    serial = backend.rollout(
        ref, state_ref=state, controller=controller, steps=steps, budgets=budgets
    )
    batch = backend.rollout_batch(
        [ref], state_refs=[state], controller=controller, steps=steps, budgets=budgets
    )[0]
    serial_trace = backend.store.get(serial.trace_ref)
    batch_trace = backend.store.get(batch["trace_ref"])
    assert batch_trace["duration_s"] == pytest.approx(steps * 0.002)
    assert len(batch_trace["states"]) == len(serial_trace["states"])
    for actual, expected in zip(batch_trace["states"], serial_trace["states"], strict=True):
        for channel in ["t", "qpos", "qvel", "ctrl"]:
            assert actual[channel] == pytest.approx(expected[channel], abs=1e-12)
    final = backend.store.get(batch["final_state_ref"])
    assert batch_trace["states"][-1]["t"] == final["time"]
    assert batch_trace["states"][-1]["qpos"] == final["qpos"]
