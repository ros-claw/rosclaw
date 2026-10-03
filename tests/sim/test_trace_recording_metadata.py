"""Public recording claims describe actual saved samples, without changing refs."""

import pytest

from rosclaw.sim.backends.mujoco.backend import MujocoBackend
from rosclaw.sim.contracts import SimulationReceipt, SimulationTrace
from tests.sim.test_batch_sampling_alignment import XML


@pytest.mark.parametrize(
    "steps,budget,stride,samples,full",
    [(5, 480, 1, 6, True), (10, 4, 3, 5, False), (15, 3, 5, 4, False)],
)
def test_public_serial_experiment_and_batch_report_actual_recording(
    tmp_path, steps, budget, stride, samples, full
):
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model_xml(XML, source={"kind": "fixture"}).model_ref
    state = backend.initial_state_v2(ref, keyframe="home")
    kwargs = {
        "state_ref": state,
        "controller": {"hold": True},
        "steps": steps,
        "budgets": {"max_record_points": budget},
    }
    trace = backend.rollout(ref, **kwargs)
    receipt = backend.run_experiment(ref, audit=False, **kwargs)
    batch = backend.rollout_batch(
        [ref],
        state_refs=[state],
        controller={"hold": True},
        steps=steps,
        budgets={"max_record_points": budget},
    )[0]
    for result in [trace.to_canonical_dict(), receipt.to_canonical_dict(), batch]:
        recording = result["recording"]
        assert recording["sample_stride"] == stride
        assert recording["total_steps"] == steps
        assert recording["saved_samples"] == samples
        assert recording["full_step_recording"] is full
        assert recording["max_record_gap_steps"] == stride
        assert recording["max_record_gap_s"] == pytest.approx(stride * 0.002)
        assert recording["includes_initial_state"] is True
        raw = backend.store.get(result["trace_ref"])
        assert len(raw["states"]) == samples
        # Metadata is observational output, never inserted into immutable records.
        assert raw.get("recording", {}) in ({}, {"max_record_points": budget})
        assert backend.store.put("traces", raw) == result["trace_ref"]
    payload = backend.store.get(receipt.receipt_ref)
    assert payload.get("recording", {}) in ({}, {"max_record_points": budget})
    assert backend.store.put("experiments", payload) == receipt.receipt_ref


def test_old_contracts_do_not_claim_full_recording_by_default():
    assert SimulationTrace(steps=10).recording == {}
    assert SimulationReceipt(steps=10).recording == {}


@pytest.mark.parametrize("indices", [[0, 2, 3, 5], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4]])
def test_missing_or_irregular_samples_never_claim_full_step_recording(indices):
    from rosclaw.sim.backends.mujoco.rollout import recording_metadata

    states = [{"t": 2 + index * 0.002} for index in indices]
    actual = recording_metadata(states, steps=5, timestep=0.002, initial_time=2)
    assert actual["full_step_recording"] is False
    assert actual["saved_samples"] == len(indices)
    if indices == [0, 2, 3, 5]:
        assert actual["sample_stride"] is None
        assert actual["max_record_gap_steps"] == 2


@pytest.mark.parametrize("times", [[0, 0.002, 0.002, 0.006], [0, 0.003, 0.006], [0, float("nan")]])
def test_untrustworthy_sample_timestamps_remain_unknown(times):
    from rosclaw.sim.backends.mujoco.rollout import recording_metadata

    actual = recording_metadata([{"t": t} for t in times], steps=3, timestep=0.002, initial_time=0)
    assert actual["full_step_recording"] is None
    assert actual["sample_stride"] is None
    assert actual["max_record_gap_steps"] is None
