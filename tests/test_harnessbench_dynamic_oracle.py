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
        "warning_counts": [0] * 7,
        "schema_version": "rosclaw.sim.runtime_validation.v1",
        "per_step_warning_check": True,
        "checked_finite_fields": [
            "qpos",
            "qvel",
            "qacc",
            "qacc_warmstart",
            "act",
            "ctrl",
            "actuator_force",
            "qfrc_actuator",
            "qfrc_constraint",
            "qfrc_bias",
            "qfrc_passive",
            "qfrc_smooth",
            "qfrc_applied",
            "xfrc_applied",
            "sensordata",
        ],
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


@pytest.mark.parametrize(
    "attribute",
    [
        'gravcomp=".5"',  # body gravity cancellation cannot masquerade as kp repair
    ],
)
def test_compiled_gravity_compensation_mutation_rejected(attribute):
    source = '<mujoco><worldbody><body name="b"><joint/><geom size=".1" mass="1"/></body></worldbody></mujoco>'
    assert not same_compiled_body(source, source.replace('name="b"', f'name="b" {attribute}'))


@pytest.mark.parametrize(
    "mutation",
    [
        ('<joint name="j"/>', '<joint name="j" stiffness="10"/>'),
        ('<joint name="j"/>', '<joint name="j" margin=".02"/>'),
        ('<joint name="j"/>', '<joint name="j" solreflimit=".04 1"/>'),
        ('<geom name="g" size=".1"/>', '<geom name="g" size=".1" margin=".03"/>'),
        ('<geom name="g" size=".1"/>', '<geom name="g" size=".1" gap=".02"/>'),
        ('<geom name="g" size=".1"/>', '<geom name="g" size=".1" condim="6"/>'),
        ('<geom name="g" size=".1"/>', '<geom name="g" size=".1" priority="1"/>'),
        ('<general joint="j"/>', '<general joint="j" dyntype="filter" dynprm=".1"/>'),
        ("<option/>", '<option o_solref=".04 1"/>'),
    ],
)
def test_compiled_dynamic_mutations_rejected(mutation):
    source = (
        '<mujoco><option/><worldbody><body><joint name="j"/>'
        '<geom name="g" size=".1"/></body></worldbody>'
        '<actuator><general joint="j"/></actuator></mujoco>'
    )
    assert not same_compiled_body(source, source.replace(*mutation))


def test_nonzero_initial_time_binds_elapsed_and_sample_grid():
    trace = _trace(offset=5)
    receipt = _receipt(trace)
    receipt["simulation_time_s"] = 1
    assert trace_stats(trace, receipt)["duration_binding"]
    trace["states"][2]["t"] += 0.0001
    assert not trace_stats(trace, _receipt(trace))["recording_grid_coverage"]


def test_nonfinite_time_and_dimension_change_fail_closed():
    trace = _trace()
    trace["states"][2]["t"] = float("nan")
    with pytest.raises(ValueError, match="state/time"):
        trace_stats(trace, _receipt(_trace()))
    trace = _trace()
    trace["states"][2]["ctrl"] = [1]
    with pytest.raises(ValueError, match="state/time"):
        trace_stats(trace, _receipt(_trace()))


def test_contact_pair_equality_and_mesh_mutations_rejected():
    source = (
        '<mujoco><worldbody><body name="a"><joint name="j"/>'
        '<geom name="g" size=".1"/></body><geom name="floor" size="1 1 .1" type="box"/>'
        '</worldbody><contact><pair geom1="g" geom2="floor" friction=".5 .5 .01 .01 .01"/>'
        '</contact><equality><joint joint1="j" polycoef="0 1 0 0 0"/></equality></mujoco>'
    )
    assert not same_compiled_body(source, source.replace('friction=".5 .5', 'friction=".7 .5'))
    assert not same_compiled_body(source, source.replace('polycoef="0 1', 'polycoef=".1 1'))
    mesh = (
        '<mujoco><asset><mesh name="m" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>'
        '<worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>'
    )
    assert not same_compiled_body(mesh, mesh.replace('0 0 1"', '0 0 1.1"'))


def test_failed_partial_requires_actual_time_or_warning_and_source_binding():
    from benchmarks.harnessbench.dynamic_oracle import failed_baseline_stats

    trace = {
        "kind": "failed_simulation_trace",
        "outcome": "FAILED",
        "model_ref": "model",
        "model_digest": "hash",
        "controller": {"position_targets": [1]},
        "action_digest": "action",
        "initial_state_ref": "state",
        "requested_steps": 500,
        "timestep_s": 0.002,
        "seed": 0,
        "failed_step": 3,
        "expected_time": 0.006,
        "valid_sampled_prefix": [{"t": 0, "qpos": [0], "qvel": [0], "ctrl": [1]}],
        "last_valid_state": {"t": 0.004, "qpos": [1], "qvel": [1e7], "ctrl": [1]},
        "failure_point": {"t": 0.002, "warning_counts": [0, 0, 0, 0, 0, 1, 0], "finite_fields": {}},
    }
    receipt = {
        k: trace[k]
        for k in (
            "model_ref",
            "model_digest",
            "controller",
            "action_digest",
            "initial_state_ref",
            "requested_steps",
            "timestep_s",
            "seed",
            "outcome",
        )
    }
    receipt.update(kind="simulation_failure", failure_code="SIM_DIVERGED")
    assert failed_baseline_stats(trace, receipt)["peak_qvel"] == 1e7
    forged = copy.deepcopy(trace)
    forged["failure_point"]["t"] = 0.006
    forged["failure_point"]["warning_counts"] = [0] * 7
    assert failed_baseline_stats(forged, receipt) is None
    receipt["action_digest"] = "wrong"
    assert failed_baseline_stats(trace, receipt) is None
