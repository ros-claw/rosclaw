"""Aggregate verified GoalForge reports into Promotion Gate V4."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rosclaw.simforge.attestation import verify_scale_curve_signature
from rosclaw.simforge.g1_proof_replay import replay_goalforge_proof_bundle
from rosclaw.simforge.g1_proofs import GOALFORGE_E5_MODULES
from rosclaw.simforge.promotion_v4 import (
    GoalForgeMetrics,
    PromotionEvidenceV4,
    PromotionGateV4,
    PromotionResultV4,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


def evaluate_goalforge_promotion(
    *,
    doctor_path: Path,
    recovery_path: Path,
    flywheel_path: Path,
    four_gpu_root: Path,
    continual_path: Path,
    chaos_path: Path,
    proof_root: Path,
    output_path: Path,
) -> PromotionResultV4:
    doctor = _read(doctor_path)
    recovery = _read(recovery_path)
    flywheel = _read(flywheel_path)
    four_gpu = _read(four_gpu_root / "four-gpu-summary.json")
    continual = _read(continual_path)
    chaos = _read(chaos_path)
    if doctor.get("passed") is not True or doctor.get("real_hardware_opened") is not False:
        raise ValueError("GoalForge Promotion requires a passing simulation-only Doctor report")
    if not all(
        value.get("passed") is True for value in (recovery, flywheel, four_gpu, continual, chaos)
    ):
        raise ValueError("GoalForge Promotion requires passing source reports")
    proof_replay = replay_goalforge_proof_bundle(
        proof_root,
        requested_modules=GOALFORGE_E5_MODULES,
    )
    if not proof_replay.passed:
        raise ValueError("GoalForge Promotion proof replay failed")
    proof_bundle = _read(proof_replay.bundle_path)
    levels = tuple(
        sorted((str(proof["module"]), str(proof["level"])) for proof in proof_bundle["proofs"])
    )
    holdout = flywheel["private_holdout"]
    verify_scale_curve_signature(
        holdout,
        expected_public_key_path=Path(holdout["public_key_path"]),
    )
    baseline = _recovery_metrics(recovery, side="baseline")
    candidate = _recovery_metrics(recovery, side="retry")
    hidden = _holdout_metrics(holdout)
    body_hash = str(flywheel["champion"]["body_hash"])
    expected_body_hash = str(doctor.get("body_hash", ""))
    expected_kick_prior_hash = str(doctor.get("kick_prior_hash", ""))
    if (
        proof_bundle.get("body_snapshot_hash") != body_hash
        or expected_body_hash != body_hash
        or flywheel["champion"].get("kick_prior_hash") != expected_kick_prior_hash
    ):
        raise ValueError("Doctor, Champion, and proof qualification hashes do not match")
    strict_replay_rate = min(
        float(flywheel["dataset"]["quality"]["strict_replay_rate"]),
        float(flywheel["dataset"]["quality"]["independent_verification_rate"]),
    )
    evidence = PromotionEvidenceV4(
        baseline=baseline,
        candidate_validation=candidate,
        hidden_holdout=hidden,
        candidate_hash=str(flywheel["adapter"]["model_hash"]),
        body_hash=body_hash,
        expected_body_hash=expected_body_hash,
        counterexample_regression_passed=(
            recovery["unrecoverable_stop_rate"] == 1.0
            and chaos["executor"]["stale_task_verified_count"] == 0
        ),
        dds_sim_to_sim_passed=bool(chaos["dds"]["passed"]),
        strict_replay_rate=strict_replay_rate,
        module_levels=levels,
        canary_passed=(
            holdout["goal_success_rate"] > 0.0
            and holdout["fall_rate"] == 0.0
            and holdout["torque_violation_rate"] == 0.0
        ),
        shards_complete=(
            not four_gpu["missing_shards"]
            and four_gpu["total_scenarios"] >= 1000
            and four_gpu["unique_gpu_uuids"] == 4
        ),
        holdout_signature_verified=True,
        evidence_commitment_verified=(
            proof_replay.bundle_hash == proof_bundle["bundle_hash"]
            and all(shard["signature_verified"] for shard in four_gpu["shards"])
        ),
        critical_safety_forgetting=int(continual["critical_safety_forgetting"]),
        historical_success_delta=float(continual["mean_historical_success_delta"]),
    )
    result = PromotionGateV4().evaluate(evidence)
    output = {
        "schema_version": "rosclaw.g1_goalforge.promotion_run.v1",
        "evidence": {
            "baseline": _metrics_dict(baseline),
            "candidate_validation": _metrics_dict(candidate),
            "hidden_holdout": _metrics_dict(hidden),
            "candidate_hash": evidence.candidate_hash,
            "body_hash": body_hash,
            "doctor_report_hash": hash_json(doctor),
            "strict_replay_rate": strict_replay_rate,
            "module_levels": dict(levels),
            "proof_bundle_hash": proof_bundle["bundle_hash"],
            "evidence_domains": {
                "validation": "MUJOCO_PHYSICS",
                "holdout": "SIGNED_MUJOCO_AGGREGATE",
                "four_gpu": "CUDA_SCREENING",
                "hardware": "NOT_OPENED",
            },
        },
        "result": result.to_dict(),
    }
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _recovery_metrics(value: dict[str, Any], *, side: str) -> GoalForgeMetrics:
    pairs = value["pairs"]
    metrics = [pair[f"{side}_metrics"] for pair in pairs]
    is_retry = side == "retry"
    if is_retry:
        first_attempt_success = sum(
            pair["sandbox_attempts"][0]["status"] == "SUCCESS" for pair in pairs
        ) / len(pairs)
        retry_success = sum(metric["success"] for metric in metrics) / len(metrics)
        mean_retries = sum(len(pair["sandbox_attempts"]) for pair in pairs) / len(pairs)
    else:
        first_attempt_success = sum(metric["success"] for metric in metrics) / len(metrics)
        retry_success = 0.0
        mean_retries = sum(pair["retry_budget"] for pair in pairs) / len(pairs)
    return GoalForgeMetrics(
        episodes=len(metrics),
        goal_success_rate=_mean(metrics, "success"),
        target_zone_success_rate=_mean(metrics, "success"),
        mean_target_error_m=_mean(metrics, "target_error_m"),
        mean_ball_speed_mps=_mean(metrics, "ball_speed_mps"),
        mean_time_to_kick_sec=_mean(metrics, "contact_time_sec"),
        first_attempt_success_rate=first_attempt_success,
        retry_success_rate=retry_success,
        mean_retries=mean_retries,
        fall_rate=_mean(metrics, "fall"),
        torque_violation_rate=_mean(metrics, "torque_violation"),
        unsafe_allow_rate=0.0,
        mean_support_slip_m=_mean(metrics, "support_slip_m"),
        mean_com_margin_m=_mean(metrics, "com_margin_m"),
        torso_roll_p95_rad=_percentile(metrics, "torso_roll_rad", 0.95),
        torso_pitch_p95_rad=_percentile(metrics, "torso_pitch_rad", 0.95),
        joint_limit_violation_rate=_mean(metrics, "joint_limit_violation"),
        mean_post_kick_stability_sec=_mean(
            metrics,
            "post_kick_stability_sec",
        ),
    )


def _holdout_metrics(value: dict[str, Any]) -> GoalForgeMetrics:
    success = float(value["goal_success_rate"])
    return GoalForgeMetrics(
        episodes=int(value["episode_count"]),
        goal_success_rate=success,
        target_zone_success_rate=float(value["target_zone_success_rate"]),
        mean_target_error_m=float(value["mean_target_error_m"]),
        mean_ball_speed_mps=float(value["mean_ball_speed_mps"]),
        mean_time_to_kick_sec=float(value["mean_time_to_kick_sec"]),
        first_attempt_success_rate=success,
        retry_success_rate=success,
        mean_retries=0.0,
        fall_rate=float(value["fall_rate"]),
        torque_violation_rate=float(value["torque_violation_rate"]),
        unsafe_allow_rate=0.0,
        mean_support_slip_m=float(value["mean_support_slip_m"]),
        mean_com_margin_m=float(value["mean_com_margin_m"]),
        torso_roll_p95_rad=float(value["torso_roll_p95_rad"]),
        torso_pitch_p95_rad=float(value["torso_pitch_p95_rad"]),
        joint_limit_violation_rate=float(value["joint_limit_violation_rate"]),
        mean_post_kick_stability_sec=float(value["mean_post_kick_stability_sec"]),
    )


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [
        float(value)
        for row in rows
        if (value := row[field]) is not None and math.isfinite(float(value))
    ]
    return sum(values) / len(values)


def _percentile(
    rows: list[dict[str, Any]],
    field: str,
    quantile: float,
) -> float:
    values = sorted(float(row[field]) for row in rows)
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"GoalForge Promotion source is not an object: {path}")
    return value


def _metrics_dict(value: GoalForgeMetrics) -> dict[str, Any]:
    return dict(vars(value))


__all__ = ["evaluate_goalforge_promotion"]
