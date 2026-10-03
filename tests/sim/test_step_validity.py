"""Finite reset states are not evidence of a valid physical simulation."""

import mujoco
import numpy as np
import pytest

from rosclaw.sim.backends.mujoco.backend import MujocoBackend
from rosclaw.sim.backends.mujoco.rollout import run_rollout

UNSTABLE_XML = """<mujoco><option timestep=".002"/><worldbody>
<body name="link" pos="0 0 .4"><joint name="j1" axis="0 1 0" damping=".01"/>
<geom name="g" type="capsule" size=".04 .2" pos="0 0 -.2" mass="1.2"/>
</body></worldbody><actuator><position name="srv" joint="j1" kp="2000000"/>
</actuator></mujoco>"""


def test_public_python_simulation_api_rejects_solver_reset(tmp_path):
    from rosclaw.sim import api

    source = tmp_path / "unstable.xml"
    source.write_text(UNSTABLE_XML)
    ref, _ = api.load_model(source, task_root=tmp_path)
    with pytest.raises(ValueError, match="SIM_DIVERGED.*(warning|time)"):
        api.submit_simulation(ref, None, {"ctrl_series": [[1]] * 500}, 1, task_root=tmp_path)


@pytest.mark.parametrize("route", ["rollout", "receipt", "batch"])
def test_real_solver_reset_cannot_produce_successful_trace_or_receipt(tmp_path, route):
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model_xml(UNSTABLE_XML, source={"kind": "fixture"})
    with pytest.raises(ValueError, match="SIM_DIVERGED.*(warning|time)"):
        if route == "batch":
            backend.rollout_batch([ref.model_ref], controller={"position_targets": [1]}, steps=500)
        else:
            execute = backend.rollout if route == "rollout" else backend.run_experiment
            execute(ref.model_ref, controller={"position_targets": [1]}, steps=500)


@pytest.mark.parametrize("route", ["interaction", "audit_sweep"])
def test_warning_reset_is_invalid_in_other_execution_paths(route):
    from rosclaw.sim.audit.context import AuditContext
    from rosclaw.sim.backends.mujoco.interact import _step

    spec = mujoco.MjSpec.from_string(UNSTABLE_XML)
    model = spec.compile()
    if route == "interaction":
        data = mujoco.MjData(model)
        data.ctrl[0] = 1
        with pytest.raises(ValueError, match="SIM_DIVERGED.*(warning|time)"):
            _step(model, data, 1)
    else:
        result = AuditContext(model=model, spec=spec, xml_text=UNSTABLE_XML).sweep(
            1, lambda *_: None, ctrl={"kind": "ctrl", "values": [1]}
        )
        assert result["diverged"] is True
        assert result["diverged_step"] == 3


def test_trace_audit_checks_executed_motion_and_time_not_only_fresh_zero_hold():
    from rosclaw.sim.audit.context import AuditContext
    from rosclaw.sim.audit.dynamics import a15_nan_inf, a16_physics_divergence

    spec = mujoco.MjSpec.from_string(UNSTABLE_XML)
    trace = {
        "states": [
            {"t": 0.0, "qpos": [0.0], "qvel": [0.0], "ctrl": [1.0]},
            {"t": 0.004, "qpos": [-13038.0], "qvel": [-6577310.0], "ctrl": [1.0]},
            {"t": 0.004, "qpos": [0.0], "qvel": [0.0], "ctrl": [0.0]},
        ]
    }
    ctx = AuditContext(model=spec.compile(), spec=spec, xml_text=UNSTABLE_XML, trace_record=trace)
    assert a15_nan_inf(ctx)["status"] == "FAIL"
    assert a16_physics_divergence(ctx)["status"] == "FAIL"


@pytest.mark.parametrize("fault", ["time_reset", "warning", "qacc", "ctrl", "actuator_force"])
def test_each_actual_step_checks_time_warnings_and_full_finite_dynamics(monkeypatch, fault):
    model = mujoco.MjModel.from_xml_string(UNSTABLE_XML.replace('kp="2000000"', 'kp="10"'))
    data = mujoco.MjData(model)
    actual_step = mujoco.mj_step

    def bad_step(model, data):
        actual_step(model, data)
        if fault == "time_reset":
            data.time = 0
        elif fault == "warning":
            data.warning[0].number += 1
        else:
            getattr(data, fault)[0] = np.inf

    monkeypatch.setattr(mujoco, "mj_step", bad_step)
    with pytest.raises(ValueError, match="SIM_DIVERGED"):
        run_rollout(model, data, plan={"kind": "hold"}, steps=2)


def test_stable_tracking_time_is_continuous_from_nonzero_initial_time():
    model = mujoco.MjModel.from_xml_string(UNSTABLE_XML.replace('kp="2000000"', 'kp="10"'))
    data = mujoco.MjData(model)
    data.time = 50
    states, steps = run_rollout(
        model, data, plan={"kind": "position_targets", "values": [1]}, steps=500
    )
    assert steps == 500
    assert states[-1]["t"] == pytest.approx(51)
