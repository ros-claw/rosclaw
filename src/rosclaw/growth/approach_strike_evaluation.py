"""Fail-closed baseline comparison for an approach-to-strike IQL residual."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash


@dataclass(frozen=True)
class G1ApproachStrikeResidualEvaluation:
    baseline_evidence_path: str
    baseline_evidence_hash: str
    candidate_evidence_path: str
    candidate_evidence_hash: str
    candidate_hash: str
    checks: dict[str, bool]
    deltas: dict[str, float]
    passed: bool
    evaluation_hash: str
    schema_version: str = "rosclaw.growth.g1_approach_strike_residual_evaluation.v1"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "baseline_evidence_path": self.baseline_evidence_path,
            "baseline_evidence_hash": self.baseline_evidence_hash,
            "candidate_evidence_path": self.candidate_evidence_path,
            "candidate_evidence_hash": self.candidate_evidence_hash,
            "candidate_hash": self.candidate_hash,
            "checks": self.checks,
            "deltas": self.deltas,
            "passed": self.passed,
            "disposition": "CANDIDATE_EVALUATED" if self.passed else "REJECTED",
            "promotion_authorized": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if include_hash:
            value["evaluation_hash"] = self.evaluation_hash
        return value


def evaluate_g1_approach_strike_residual(
    *,
    baseline_evidence_path: Path,
    candidate_evidence_path: Path,
    output_path: Path,
    source_checkout: Path,
    minimum_effect_fraction: float = 0.05,
) -> G1ApproachStrikeResidualEvaluation:
    """Compare one residual rollout against its controller-equivalent baseline."""

    if not 0.01 <= minimum_effect_fraction <= 0.25:
        raise ValueError("approach-strike learned-effect threshold must be in [0.01, 0.25]")
    baseline_path = baseline_evidence_path.expanduser().resolve()
    candidate_path = candidate_evidence_path.expanduser().resolve()
    baseline = _load_strict_evidence(baseline_path)
    candidate = _load_strict_evidence(candidate_path)
    _require_comparable(baseline, candidate)
    candidate_hash = str(candidate.get("approach_strike_candidate_hash", ""))
    if not candidate_hash.startswith("sha256:"):
        raise ValueError("approach-strike evaluation candidate is not bound to an IQL hash")
    base = baseline["result"]
    result = candidate["result"]
    base_error = _finite_metric(base, "goal_plane_target_error_m")
    candidate_error = _finite_metric(result, "goal_plane_target_error_m")
    effect = _finite_metric(result, "residual_effect_fraction")
    deltas = {
        "goal_plane_target_error_m": candidate_error - base_error,
        "actuator_saturation_steps": float(result["actuator_saturation_steps"])
        - float(base["actuator_saturation_steps"]),
        "kick_peak_tilt_rad": float(result["kick_peak_tilt_rad"])
        - float(base["kick_peak_tilt_rad"]),
        "final_speed_mps": float(result["final_speed_mps"]) - float(base["final_speed_mps"]),
        "residual_effect_fraction": effect,
    }
    checks = {
        "strict_replay": baseline.get("strict_replay") is True
        and candidate.get("strict_replay") is True,
        "candidate_executed": result.get("learned_approach_strike_residual_executed") is True,
        "minimum_learned_effect": effect >= minimum_effect_fraction,
        "precision_non_regression": candidate_error <= base_error + 0.005,
        "authority_non_regression": int(result["actuator_saturation_steps"])
        <= int(base["actuator_saturation_steps"]),
        "tilt_non_regression": float(result["kick_peak_tilt_rad"])
        <= float(base["kick_peak_tilt_rad"]) + 0.01,
        "settling_non_regression": float(result["final_speed_mps"])
        <= float(base["final_speed_mps"]) + 0.02,
        "no_new_safety_failure": not any(
            result.get(name) is True
            for name in ("post_kick_fall", "joint_limit_violation", "torque_limit_violation")
        ),
        "absolute_task_gate": candidate.get("passed") is True,
    }
    unsigned = {
        "schema_version": "rosclaw.growth.g1_approach_strike_residual_evaluation.v1",
        "baseline_evidence_path": str(baseline_path),
        "baseline_evidence_hash": _file_hash(baseline_path),
        "candidate_evidence_path": str(candidate_path),
        "candidate_evidence_hash": _file_hash(candidate_path),
        "candidate_hash": candidate_hash,
        "checks": checks,
        "deltas": deltas,
        "passed": all(checks.values()),
        "disposition": "CANDIDATE_EVALUATED" if all(checks.values()) else "REJECTED",
        "promotion_authorized": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    evaluation = G1ApproachStrikeResidualEvaluation(
        baseline_evidence_path=str(baseline_path),
        baseline_evidence_hash=str(unsigned["baseline_evidence_hash"]),
        candidate_evidence_path=str(candidate_path),
        candidate_evidence_hash=str(unsigned["candidate_evidence_hash"]),
        candidate_hash=candidate_hash,
        checks=checks,
        deltas=deltas,
        passed=bool(unsigned["passed"]),
        evaluation_hash=canonical_hash(unsigned),
    )
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("approach-strike evaluation must be outside the source checkout")
    if output.exists():
        raise FileExistsError("approach-strike evaluation output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation


def _load_strict_evidence(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("strict_replay") is not True or not isinstance(value.get("result"), dict):
        raise ValueError("approach-strike comparison requires strict result evidence")
    trajectory = Path(str(value.get("trajectory_path", ""))).resolve()
    if not trajectory.is_file() or value.get("trajectory_hash") != _file_hash(trajectory):
        raise ValueError("approach-strike evidence trajectory commitment is invalid")
    return value


def _require_comparable(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    if baseline.get("body_hash") != candidate.get("body_hash"):
        raise ValueError("approach-strike baseline and candidate body hashes differ")
    for section, keys in (
        ("runup_config", ("start_x_m", "start_y_m", "control_dt_sec", "physics_dt_sec")),
        (
            "flow_config",
            (
                "approach_provider",
                "bridge_duration_sec",
                "kick_phase_start_frame",
                "aim_bias_y_m",
                "shot_pelvis_yaw_offset_rad",
                "shot_foot_yaw_offset_rad",
                "shot_com_shift_y_m",
            ),
        ),
        (
            "sonic_runup_config",
            (
                "run_velocity_mps",
                "brake_velocity_mps",
                "execution_duration_sec",
                "planner_seed",
            ),
        ),
        ("goal_spec", ("plane_x_m", "target_y_m", "target_z_m", "precision_radius_m")),
    ):
        left = baseline.get(section, {})
        right = candidate.get(section, {})
        if any(left.get(key) != right.get(key) for key in keys):
            raise ValueError(f"approach-strike comparison differs in {section}")


def _finite_metric(result: dict[str, Any], name: str) -> float:
    value = result.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"approach-strike metric {name} is not finite")
    return float(value)


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["G1ApproachStrikeResidualEvaluation", "evaluate_g1_approach_strike_residual"]
