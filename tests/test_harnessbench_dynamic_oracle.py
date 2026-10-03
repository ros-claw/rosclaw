"""Independent trace math and body-contract counterexamples, no physical steps."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from benchmarks.harnessbench.dynamic_oracle import (
    convergence,
    runtime_validation_ok,
    same_compiled_body,
    trace_stats,
)


def _trace(dt=0.004, offset=0):
    states = [
        {
            "t": i * dt + offset,
            "qpos": [0, 0, 1 - 0.5 * (i * dt) ** 2],
            "qvel": [0, 0, -i * dt],
            "ctrl": [],
        }
        for i in range(round(1 / dt) + 1)
    ]
    return {
        "states": states,
        "model_ref": "simmdl_test",
        "model_digest": "sha256:model",
        "timestep_s": dt,
    }


def _receipt(trace):
    return {
        "model_ref": trace["model_ref"],
        "model_digest": trace["model_digest"],
        "steps": len(trace["states"]) - 1,
        "simulation_time_s": trace["states"][-1]["t"],
        "states_digest": "sha256:"
        + hashlib.sha256(
            json.dumps(
                trace["states"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest(),
    }


def test_original_reset_timestamps_not_hidden_by_finite_values():
    trace = _trace()
    trace["states"][2]["t"] = trace["states"][1]["t"]
    stats = trace_stats(trace, _receipt(trace))
    assert stats["finite_recorded_arrays"]
    assert not stats["strictly_increasing_time"]


def test_wrong_receipt_hash_and_step_coverage_rejected():
    trace = _trace()
    receipt = _receipt(trace)
    receipt["states_digest"] = "sha256:fake"
    receipt["steps"] += 1
    stats = trace_stats(trace, receipt)
    assert not stats["receipt_states_binding"]
    assert not stats["step_time_coverage"]


def test_actual_qvel_peak_not_producer_metric():
    trace = _trace()
    trace["states"][2]["qvel"][0] = 1e7
    receipt = _receipt(trace)
    receipt["metrics"] = {"peak_qvel": 0}
    assert trace_stats(trace, receipt)["peak_qvel"] == 1e7


def test_nonfinite_trace_never_finite():
    trace = _trace()
    trace["states"][2]["qvel"][0] = float("inf")
    with pytest.raises(ValueError):
        trace_stats(trace, _receipt(trace))


def test_missing_or_nonzero_warning_evidence_not_promoted():
    trace, receipt = _trace(), _receipt(_trace())
    assert not runtime_validation_ok(trace, receipt)
    validation = {
        "status": "PASS",
        "method": "serial_each_step",
        "steps_checked": 250,
        "time_continuity_checked": True,
        "warning_counts": [0] * 8,
    }
    trace["runtime_validation"] = validation
    receipt["runtime_validation"] = copy.deepcopy(validation)
    assert runtime_validation_ok(trace, receipt)
    trace["runtime_validation"]["warning_counts"][3] = 1
    assert not runtime_validation_ok(trace, receipt)


def test_convergence_exact_common_times_and_sparse_refusal():
    assert convergence(_trace(0.004), _trace(0.002)) == 0
    with pytest.raises(ValueError, match="undersampled"):
        convergence(_trace(0.02), _trace(0.01))
    with pytest.raises(ValueError, match="common recorded"):
        convergence(_trace(0.004), _trace(0.002, offset=0.0001))


def test_compiled_physics_not_text_format_preserved():
    original = """<mujoco><option timestep="0.005"/><worldbody>
    <body name="ball" pos="0 0 1"><freejoint/><geom name="g" type="sphere" size=".05" mass=".5"/></body>
    </worldbody></mujoco>"""
    candidate = original.replace('timestep="0.005"', 'timestep="0.001" integrator="RK4"')
    assert same_compiled_body(original, candidate)
    assert not same_compiled_body(original, candidate.replace('mass=".5"', 'mass=".6"'))
    assert not same_compiled_body(
        original, candidate.replace('timestep="0.001"', 'timestep="0.001" gravity="0 0 0"')
    )
    assert not same_compiled_body(original, candidate.replace('size=".05"', 'size=".1"'))
