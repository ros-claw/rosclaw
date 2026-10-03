"""External saved-trace grading for dynamic repair and timestep experiments.

No new physical rollout is produced by this oracle. It compiles source contracts
and independently reads native recorded arrays, never self-reported scalar scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def same_compiled_body(
    original_xml: str, candidate_xml: str, *, allow_control: bool = False
) -> bool:
    """Compare actual compiled inertias/topology/limits, not MjSpec XML formatting."""
    import mujoco
    import numpy as np

    original = mujoco.MjModel.from_xml_string(original_xml)
    candidate = mujoco.MjModel.from_xml_string(candidate_xml)

    def public_values(owner, excluded):
        # Compare every exposed numeric buffer/scalar, including compiled
        # contact pairs, equality, meshes, actuator dynamics and option overrides.
        # Methods/accessors are not data. New MuJoCo arrays are included by default.
        result = {}
        for key in dir(owner):
            if key.startswith("_") or key in excluded:
                continue
            value = getattr(owner, key)
            if isinstance(value, (np.ndarray, int, float, bool, str, bytes)):
                result[key] = value
        return result

    allowed = {"dof_damping", "actuator_gainprm", "actuator_biasprm"} if allow_control else set()
    if allow_control:
        gains, bias = candidate.actuator_gainprm, candidate.actuator_biasprm
        if (
            np.any(~np.isfinite(gains))
            or np.any(~np.isfinite(bias))
            or np.any(~np.isfinite(candidate.dof_damping))
            or np.any(gains[:, 0] <= 0)
            or np.any(candidate.dof_damping < 0)
            or not np.array_equal(gains[:, 1:], original.actuator_gainprm[:, 1:])
            or not np.array_equal(bias[:, 0], original.actuator_biasprm[:, 0])
            or not np.array_equal(bias[:, 2:], original.actuator_biasprm[:, 2:])
            or not np.array_equal(bias[:, 1], -gains[:, 0])
        ):
            return False

    def equal(a, b):
        return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)

    return equal(public_values(original, allowed), public_values(candidate, allowed)) and equal(
        public_values(original.opt, {"timestep", "integrator"}),
        public_values(candidate.opt, {"timestep", "integrator"}),
    )


def trace_stats(trace: dict, receipt: dict, *, max_time: float | None = None) -> dict:
    """Recorded semantics; absent warning telemetry stays unknown."""
    states = trace.get("states", [])
    if not isinstance(states, list) or len(states) < 2:
        raise ValueError("trace lacks recorded state sequence")
    if any(
        not isinstance(state, dict)
        or type(state.get("t")) not in (int, float)
        or not math.isfinite(state["t"])
        or any(
            not isinstance(state.get(key), list)
            or any(type(v) not in (int, float) for v in state[key])
            or len(state[key]) != len(states[0].get(key, []))
            for key in ("qpos", "qvel", "ctrl")
        )
        for state in states
    ):
        raise ValueError("invalid typed state/time arrays")
    times = [float(state["t"]) for state in states]
    dt: Any = trace.get("timestep_s")
    steps = receipt.get("steps")
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid timestep")
    if type(steps) is not int or steps <= 0:
        raise ValueError("invalid receipt steps")
    intervals = [(b - a) / dt for a, b in zip(times, times[1:], strict=False)]
    integral_intervals = all(math.isclose(x, round(x), abs_tol=1e-7) for x in intervals)
    # Legacy serial recording defaults to 250 points. A new public density
    # budget must be preserved in trace metadata; infer no arbitrary jump allowance.
    recording = trace.get("recording", {})
    points = recording.get("max_record_points", 480) if isinstance(recording, dict) else 480
    if type(points) is not int or not 1 <= points <= 100000:
        raise ValueError("invalid recording density")
    stride = max(1, math.ceil(steps / points))
    if not recording and all(math.isclose(x, 1, abs_tol=1e-7) for x in intervals):
        stride = 1  # Full dense trace proves no samples omitted without legacy budget metadata.
    stride_bound = (
        all(math.isclose(x, stride, abs_tol=1e-7) for x in intervals[:-1])
        and 0 < intervals[-1] <= stride + 1e-7
    )

    finite = all(
        math.isfinite(value)
        for state in states
        for key in ("qpos", "qvel", "ctrl")
        for value in map(float, state[key])
    )
    monotonic = all(b > a for a, b in zip(times, times[1:], strict=False))
    bound = times[0] + max_time if max_time is not None else times[-1]
    chosen = [state for state in states if float(state["t"]) <= bound + 1e-10]
    states_hash = "sha256:" + hashlib.sha256(_canonical(states).encode()).hexdigest()
    return {
        "finite_recorded_arrays": finite,
        "strictly_increasing_time": monotonic,
        "receipt_states_binding": receipt.get("states_digest") == states_hash,
        "model_binding": trace.get("model_ref") == receipt.get("model_ref")
        and trace.get("model_digest") == receipt.get("model_digest"),
        "duration_binding": math.isclose(
            times[-1] - times[0], float(receipt.get("simulation_time_s", -1)), abs_tol=1e-8
        ),
        "step_time_coverage": math.isclose(
            times[-1] - times[0],
            int(receipt.get("steps", -1)) * float(trace.get("timestep_s", -1)),
            abs_tol=1e-8,
        ),
        "recording_grid_coverage": integral_intervals and stride_bound,
        "peak_qvel": max(abs(float(v)) for s in chosen for v in s["qvel"]),
        "start_time": times[0],
        "end_time": times[-1],
        "state_count": len(states),
        "selected_states": chosen,
    }


def _records(
    backend, refs: set[str], *, include_failures: bool = False
) -> list[tuple[str, dict, dict]]:
    result = []
    for ref in backend.store.list_children("experiments"):
        record = backend.store.get(ref)
        if isinstance(record, dict) and record.get("model_ref") in refs and record.get("trace_ref"):
            trace = backend.store.get(record["trace_ref"])
            if isinstance(trace, dict) and (
                trace.get("kind") != "failed_simulation_trace" or include_failures
            ):
                result.append((ref, record, trace))
    return result


def failed_baseline_stats(trace: dict, receipt: dict) -> dict | None:
    """Verify preserved actual guard failure, not a producer's failure adjective."""
    if (
        receipt.get("kind") != "simulation_failure"
        or trace.get("kind") != "failed_simulation_trace"
        or receipt.get("failure_code") != "SIM_DIVERGED"
        or receipt.get("outcome") != "FAILED"
        or trace.get("outcome") != "FAILED"
        or any(
            receipt.get(k) != trace.get(k)
            for k in (
                "model_ref",
                "model_digest",
                "controller",
                "action_digest",
                "initial_state_ref",
                "requested_steps",
                "timestep_s",
                "seed",
            )
        )
    ):
        return None
    prefix = trace.get("valid_sampled_prefix")
    point, last = trace.get("failure_point"), trace.get("last_valid_state")
    step: Any = trace.get("failed_step")
    dt: Any = trace.get("timestep_s")
    if (
        not isinstance(prefix, list)
        or not prefix
        or not isinstance(point, dict)
        or not isinstance(last, dict)
        or type(step) is not int
        or not 1 <= step <= trace.get("requested_steps", 0)
        or type(dt) not in (int, float)
        or not math.isfinite(dt)
        or dt <= 0
    ):
        return None
    counts = point.get("warning_counts")
    if (
        not isinstance(counts, list)
        or len(counts) != 7
        or any(type(c) is not int or c < 0 for c in counts)
    ):
        return None
    initial = prefix[0]
    try:
        expected = float(initial["t"]) + step * dt
        actual = float(point["t"])
        prior = float(last["t"])
        time_bad = not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-8)
        fields = point.get("finite_fields", {})
        nonfinite_bad = any(
            status is False
            and isinstance(point.get(key), list)
            and any(value in ("NaN", "+Inf", "-Inf") for value in point[key])
            for key, status in fields.items()
        )
        if not (
            math.isclose(float(trace["expected_time"]), expected, abs_tol=1e-8)
            and math.isclose(prior, float(initial["t"]) + (step - 1) * dt, abs_tol=1e-8)
            and (any(counts) or time_bad or nonfinite_bad)
        ):
            return None
        peak = max(abs(float(v)) for state in prefix + [last] for v in state["qvel"])
        return {
            "receipt_states_binding": True,
            "model_binding": True,
            "strictly_increasing_time": not time_bad,
            "finite_recorded_arrays": not nonfinite_bad,
            "peak_qvel": peak,
            "failed_step": step,
            "warning_counts": counts,
            "initial_qpos": initial["qpos"],
            "failure_evidence_verified": True,
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _valid_stats(stats: dict) -> bool:
    return all(
        stats[key]
        for key in (
            "finite_recorded_arrays",
            "strictly_increasing_time",
            "receipt_states_binding",
            "model_binding",
            "duration_binding",
            "step_time_coverage",
            "recording_grid_coverage",
        )
    )


def runtime_validation_ok(trace: dict, receipt: dict) -> bool:
    validation = trace.get("runtime_validation")
    if not isinstance(validation, dict) or validation.get("status") != "PASS":
        return False
    counts = validation.get("warning_counts")
    states = trace.get("states", [])
    if len(states) < 2:
        return False
    initial, final = states[0]["t"], states[-1]["t"]
    elapsed = final-initial
    if type(validation.get("steps_checked")) is not int:
        return False
    for key, expected in {
        "initial_time": initial, "final_time": final,
        "actual_elapsed_s": elapsed, "expected_elapsed_s": receipt["steps"]*trace["timestep_s"],
    }.items():
        value = validation.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not math.isclose(
            value,expected,abs_tol=1e-8
        ):
            return False
    return (
        validation.get("method") == "serial_each_step"
        and validation.get("steps_checked") == receipt.get("steps")
        and validation.get("time_continuity_checked") is True
        and validation.get("schema_version") == "rosclaw.sim.runtime_validation.v1"
        and validation.get("per_step_warning_check") is True
        and set(validation.get("checked_finite_fields", []))
        >= {
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
        }
        and isinstance(counts, list)
        and len(counts) == 7
        and all(type(count) is int and count == 0 for count in counts)
        and receipt.get("runtime_validation") == validation
    )



def initial_state_equal(backend, first_receipt: dict, second_receipt: dict) -> bool:
    """Bind complete native integration vectors, including time/velocity/control."""
    try:
        metas = [backend.store.get(r["initial_state_ref"]) for r in (first_receipt,second_receipt)]
        if any(
            not isinstance(m,dict) or m.get("kind") != "state_snapshot_v2"
            or m.get("fidelity") != "FULL_INTEGRATION"
            for m in metas
        ):
            return False
        if any(metas[0].get(k) != metas[1].get(k) for k in (
            "state_spec_value","state_size","structural_signature"
        )):
            return False
        blobs = [backend.store.get(m["state_vector_ref"]) for m in metas]
        return (
            all(isinstance(blob,bytes) for blob in blobs)
            and all(
                m.get("state_digest") == "sha256:"+hashlib.sha256(blob).hexdigest()
                for m,blob in zip(metas,blobs,strict=True)
            )
            and blobs[0] == blobs[1]
        )
    except (KeyError,TypeError,ValueError):
        return False

def _runtime_parts(root: Path, task, asset: str):
    from benchmarks.harnessbench import oracle

    runtime = oracle._runtime(root)
    original, modified = oracle._load_staged_original(root, task, asset)
    backend = runtime.backend
    candidates = {
        ref
        for ref in backend.store.list_children("models")
        if ref != original and oracle._lineage_root(backend, ref) == original
    }
    return oracle, backend, original, modified, candidates


def judge_dynamic_repair(root: Path, task) -> dict:
    oracle, backend, original, modified, candidates = _runtime_parts(
        root, task, task.oracle["original_asset"]
    )
    answer = oracle._read_answer(root) or {}
    claimed = answer.get("fixed_model_ref")
    verdict = {
        "task_success": False,
        "verified_success": False,
        "false_success": False,
        "reason": "missing_comparable_dynamic_evidence",
        "original_ref": original,
        "staged_file_modified": modified,
        "physics_steps_by_oracle": 0,
    }
    original_xml = task.staged_files[task.oracle["original_asset"]]
    baselines = _records(backend, {original}, include_failures=True)
    for baseline_ref, baseline_receipt, baseline_trace in baselines:
        target = baseline_trace.get("controller", {}).get("position_targets")
        if not isinstance(target, list) or len(target) != 1 or abs(float(target[0])) < 0.1:
            continue
        if baseline_trace.get("kind") == "failed_simulation_trace":
            before = failed_baseline_stats(baseline_trace, baseline_receipt)
            if before is None:
                continue
        else:
            before = trace_stats(baseline_trace, baseline_receipt, max_time=1.0)
            before["initial_qpos"] = baseline_trace["states"][0]["qpos"]
        if not before["receipt_states_binding"] or not before["model_binding"]:
            continue
        broken = (
            not before["strictly_increasing_time"]
            or not before["finite_recorded_arrays"]
            or before["peak_qvel"] > 10000
            or before.get("failure_evidence_verified") is True
        )
        if not broken:
            continue
        for candidate_ref, candidate_receipt, candidate_trace in _records(backend, candidates):
            if candidate_trace.get("controller") != baseline_trace.get("controller"):
                continue
            model_ref = candidate_receipt["model_ref"]
            manifest = backend.store.get(model_ref)
            if not same_compiled_body(original_xml, manifest["mjcf_xml"], allow_control=True):
                continue
            original_option, candidate_option = _option(original_xml), _option(manifest["mjcf_xml"])
            if original_option != candidate_option:
                continue
            after = trace_stats(candidate_trace, candidate_receipt, max_time=1.0)
            if (
                not _valid_stats(after)
                or after["end_time"] - after["start_time"] < 1.0 - 1e-8
                or after["peak_qvel"] > 100
            ):
                continue
            initial_matches = candidate_trace["states"][0]["qpos"] == before["initial_qpos"]
                and initial_state_equal(backend,baseline_receipt,candidate_receipt)
            errors = [
                (float(s["qpos"][0]) - float(target[0])) ** 2 for s in after["selected_states"]
            ]
            rmse = math.sqrt(sum(errors) / len(errors))
            controller_retained = all(
                len(s["ctrl"]) == 1
                and math.isclose(float(s["ctrl"][0]), float(target[0]), abs_tol=1e-10)
                for s in candidate_trace["states"]
            )
            if not initial_matches or not controller_retained or rmse > 0.2:
                continue
            verdict.update(
                reason="dynamic_repair_record_verified",
                baseline_ref=baseline_ref,
                candidate_receipt_ref=candidate_ref,
                candidate_ref=model_ref,
                baseline_peak_qvel=before["peak_qvel"],
                baseline_time_valid=before["strictly_increasing_time"],
                baseline_failed_step=before.get("failed_step"),
                baseline_warning_counts=before.get("warning_counts"),
                candidate_peak_qvel=after["peak_qvel"],
                candidate_first_second_rmse=rmse,
                model_body_preserved=True,
                same_controller=True,
            )
            # Strong warning/step validity is added by the runtime fix. Historical
            # traces without telemetry remain component-only, never retroactive PASS.
            if not runtime_validation_ok(candidate_trace, candidate_receipt):
                verdict.update(
                    reason="candidate_runtime_validity_evidence_missing",
                    warning_evidence="NOT_RECORDED",
                )
                return verdict
            verdict.update(task_success=True, verified_success=True)
            oracle._apply_claim_check(backend, verdict, claimed, model_ref)
            return verdict
    return verdict


def _option(xml: str) -> tuple[float, str]:
    option = ET.fromstring(xml).find("option")
    return (
        float(option.get("timestep", "0.002")) if option is not None else 0.002,
        option.get("integrator", "Euler") if option is not None else "Euler",
    )


def convergence(coarse: dict, fine: dict) -> float:
    """Compare positions at common recorded timestamps; never interpolate impacts."""
    import numpy as np

    tc = np.asarray([s["t"] for s in coarse["states"]], dtype=float)
    tf = np.asarray([s["t"] for s in fine["states"]], dtype=float)
    qc = np.asarray([s["qpos"][:3] for s in coarse["states"]], dtype=float)
    qf = np.asarray([s["qpos"][:3] for s in fine["states"]], dtype=float)
    overlap = (tc >= tf[0]) & (tc <= min(tf[-1], tc[-1]))
    if (
        overlap.sum() < 10 or min(tc[-1],tf[-1])-max(tc[0],tf[0]) < 1.0-1e-8
        or tc[-1]-tc[0] < 1.0-1e-8 or tf[-1]-tf[0] < 1.0-1e-8
    ):
        raise ValueError("paired trace lacks common physical-time coverage")
    if max(np.diff(tc)) > 0.005 + 1e-10 or max(np.diff(tf)) > 0.005 + 1e-10:
        raise ValueError("paired trace is undersampled for convergence evidence")
    indices = np.searchsorted(tf, tc[overlap])
    indices = np.clip(indices, 0, len(tf) - 1)
    previous = np.maximum(indices - 1, 0)
    indices = np.where(
        np.abs(tf[previous] - tc[overlap]) < np.abs(tf[indices] - tc[overlap]), previous, indices
    )
    matched = np.isclose(tf[indices], tc[overlap], rtol=0, atol=1e-8)
    if matched.sum() < 10:
        raise ValueError("paired traces lack common recorded physical timestamps")
    return float(np.max(np.linalg.norm(qc[overlap][matched] - qf[indices[matched]], axis=1)))


def judge_timestep(root: Path, task) -> dict:
    oracle, backend, original, modified, candidates = _runtime_parts(
        root, task, task.oracle["original_asset"]
    )
    answer = oracle._read_answer(root) or {}
    claimed = answer.get("best_model_ref")
    verdict = {
        "task_success": False,
        "verified_success": False,
        "false_success": False,
        "reason": "paired_timestep_trace_evidence_missing",
        "original_ref": original,
        "staged_file_modified": modified,
        "physics_steps_by_oracle": 0,
    }
    source = task.staged_files[task.oracle["original_asset"]]
    models = {}
    for ref in candidates | {original}:
        manifest = backend.store.get(ref)
        if same_compiled_body(source, manifest["mjcf_xml"]):
            models[ref] = _option(manifest["mjcf_xml"])
    records = _records(backend, set(models))
    pairs = []
    for coarse_ref, coarse_receipt, coarse in records:
        coarse_model = coarse_receipt["model_ref"]
        dt, integrator = models[coarse_model]
        for fine_ref, fine_receipt, fine in records:
            fine_model = fine_receipt["model_ref"]
            fine_dt, fine_integrator = models[fine_model]
            if fine_integrator != integrator or not math.isclose(fine_dt * 2, dt, rel_tol=1e-9):
                continue
            if (
                coarse.get("controller") != fine.get("controller")
                or coarse["states"][0] != fine["states"][0]
                or not initial_state_equal(backend,coarse_receipt,fine_receipt)
            ):
                continue
            a, b = trace_stats(coarse, coarse_receipt), trace_stats(fine, fine_receipt)
            if not _valid_stats(a) or not _valid_stats(b):
                continue
            try:
                error = convergence(coarse, fine)
            except ValueError:
                continue
            pairs.append((coarse_model, error, coarse_ref, fine_ref, coarse, fine))
    baseline = [pair for pair in pairs if pair[0] == original]
    candidate = [pair for pair in pairs if pair[0] != original and pair[0] == claimed]
    verdict["independent_pairs"] = [
        {
            "model_ref": p[0],
            "max_position_error_m": p[1],
            "coarse_receipt": p[2],
            "fine_receipt": p[3],
        }
        for p in pairs
    ]
    if not baseline or not candidate:
        return verdict
    baseline = [
        pair for pair in baseline if initial_state_equal(
            backend,
            next(r for ref,r,_ in records if ref == pair[2]),
            next(r for ref,r,_ in records if ref == candidate[0][2]),
        )
    ]
    if not baseline:
        verdict["reason"] = "baseline_candidate_initial_state_mismatch"
        return verdict
    baseline_error = max(p[1] for p in baseline)
    best = min(candidate, key=lambda p: p[1])
    improvement = 1 - best[1] / baseline_error if baseline_error else 0
    verdict.update(
        baseline_max_position_error_m=baseline_error,
        candidate_max_position_error_m=best[1],
        improvement_ratio=improvement,
        model_body_preserved=True,
    )
    if best[1] > 0.0005 or improvement < 0.3:
        verdict["reason"] = "candidate_not_converged_or_improved"
        return verdict
    receipt_by_ref = {ref: receipt for ref, receipt, _ in records}
    if any(
        not runtime_validation_ok(trace, receipt_by_ref[ref])
        for ref, trace in ((best[2], best[4]), (best[3], best[5]))
    ):
        verdict.update(
            reason="candidate_runtime_validity_evidence_missing", warning_evidence="NOT_RECORDED"
        )
        return verdict
    verdict.update(
        task_success=True, verified_success=True, reason="independent_timestep_convergence_verified"
    )
    return verdict
