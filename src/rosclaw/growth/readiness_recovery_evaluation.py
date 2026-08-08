"""Frozen conditional validation for physical recovery after G1 abstention."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.proprioceptive_expert_router import (
    load_g1_proprioceptive_expert_router,
)
from rosclaw.growth.proprioceptive_readiness_gate import (
    load_g1_proprioceptive_readiness_gate,
)


@dataclass(frozen=True)
class G1ReadinessRecoveryEvaluation:
    router_hash: str
    readiness_gate_hash: str
    body_hash: str
    implementation_hash: str
    seeds: tuple[int, ...]
    episode_count: int
    passed_episodes: int
    strict_replay_all: bool
    zero_saturation_episodes: int
    no_fall_episodes: int
    minimum_recovery_pelvis_height_m: float
    maximum_recovery_tilt_rad: float
    maximum_final_speed_mps: float
    maximum_final_joint_velocity_rms_rad_s: float
    mean_initial_speed_mps: float
    mean_final_speed_mps: float
    mean_initial_joint_velocity_rms_rad_s: float
    mean_final_joint_velocity_rms_rad_s: float
    source_evidence_hashes: tuple[str, ...]
    source_trajectory_hashes: tuple[str, ...]
    accepted: bool
    failure_codes: tuple[str, ...]
    report_hash: str
    schema_version: str = "rosclaw.growth.g1_readiness_recovery_evaluation.v1"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "seeds": list(self.seeds),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "source_trajectory_hashes": list(self.source_trajectory_hashes),
            "failure_codes": list(self.failure_codes),
            "conditional_contract": {
                "readiness_abstained": True,
                "actuator_saturation_steps": 0,
                "post_abstention_fall": False,
                "minimum_pelvis_height_m": 0.65,
                "maximum_peak_tilt_rad": 0.65,
                "minimum_final_pelvis_height_m": 0.70,
                "maximum_final_speed_mps": 0.20,
                "maximum_final_joint_velocity_rms_rad_s": 0.50,
            },
            "evidence_domain": "SIM_ONLY_FROZEN_CONDITIONAL_VALIDATION",
            "generalizes_to_non_abstained_states": False,
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if not include_hash:
            value.pop("report_hash")
        return value


def evaluate_g1_readiness_recovery(
    *,
    evidence_paths: tuple[Path, ...],
    router_path: Path,
    gate_path: Path,
    output_path: Path,
    source_checkout: Path,
) -> G1ReadinessRecoveryEvaluation:
    if len(evidence_paths) < 3:
        raise ValueError("readiness recovery evaluation requires at least three episodes")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("readiness recovery evaluation must be outside the checkout")
    if output.exists():
        raise FileExistsError("readiness recovery evaluation output already exists")
    router = load_g1_proprioceptive_expert_router(router_path)
    gate = load_g1_proprioceptive_readiness_gate(gate_path)
    if gate.router_hash != router.router_hash or gate.body_hash != router.body_hash:
        raise ValueError("readiness recovery router/gate binding is invalid")

    seeds: list[int] = []
    results: list[dict[str, Any]] = []
    source_evidence_hashes: list[str] = []
    source_trajectory_hashes: list[str] = []
    implementation_hashes: set[str] = set()
    recovery_configs: set[str] = set()
    sonic_configs: set[str] = set()
    strict_values: list[bool] = []
    episode_passes: list[bool] = []
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if (
            evidence.get("schema_version")
            != "rosclaw.simforge.g1_readiness_recovery_evidence.v3"
            or evidence.get("evidence_domain")
            != "FROZEN_READINESS_RECOVERY_VALIDATION"
            or evidence.get("activation_ceiling") != "SIM_ONLY"
            or evidence.get("hardware_command_sent") is not False
            or evidence.get("router_hash") != router.router_hash
            or evidence.get("readiness_gate_hash") != gate.gate_hash
            or evidence.get("body_hash") != router.body_hash
        ):
            raise ValueError("readiness recovery evidence safety boundary is invalid")
        claims = dict(evidence.get("claims", {}))
        if (
            claims.get("readiness_abstention_executed") is not True
            or claims.get("continuous_world_no_state_reset") is not True
            or claims.get("ball_contact_attempted") is not False
            or claims.get("neural_feedback_active_during_hold") is not True
            or claims.get("promotion_evidence") is not False
            or claims.get("real_hardware") is not False
        ):
            raise ValueError("readiness recovery evidence claims are invalid")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        trajectory_hash = _file_hash(trajectory)
        if evidence.get("trajectory_hash") != trajectory_hash:
            raise ValueError("readiness recovery trajectory binding is invalid")
        request_path = path.with_name("request.json")
        if evidence.get("request_hash") != _file_hash(request_path):
            raise ValueError("readiness recovery request binding is invalid")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if (
            request.get("evidence_domain") != evidence.get("evidence_domain")
            or request.get("recovery_config") != evidence.get("recovery_config")
            or request.get("sonic_runup_config") != evidence.get("sonic_runup_config")
        ):
            raise ValueError("readiness recovery request context differs from evidence")
        result = dict(evidence.get("result", {}))
        seed = int(result.get("planner_seed", -1))
        if seed < 0 or seed in seeds:
            raise ValueError("readiness recovery seed is invalid or duplicated")
        config = dict(evidence.get("recovery_config", {}))
        recomputed_pass = _episode_passed(result, config)
        if bool(evidence.get("passed")) != (
            recomputed_pass and evidence.get("strict_replay") is True
        ):
            raise ValueError("readiness recovery pass claim does not match metrics")
        seeds.append(seed)
        results.append(result)
        strict_values.append(evidence.get("strict_replay") is True)
        episode_passes.append(recomputed_pass)
        source_evidence_hashes.append(_file_hash(path))
        source_trajectory_hashes.append(trajectory_hash)
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        recovery_configs.add(canonical_hash(config))
        sonic = dict(evidence.get("sonic_runup_config", {}))
        sonic.pop("planner_seed", None)
        sonic_configs.add(canonical_hash(sonic))

    if len(implementation_hashes) != 1 or len(recovery_configs) != 1 or len(sonic_configs) != 1:
        raise ValueError("readiness recovery frozen implementation or config differs")
    ordered = sorted(zip(seeds, results, episode_passes, strict=True), key=lambda item: item[0])
    seeds_tuple = tuple(item[0] for item in ordered)
    outcomes = [item[1] for item in ordered]
    passes = [item[2] for item in ordered]
    strict = all(strict_values)
    passed_count = sum(passes)
    zero_saturation = sum(int(item["actuator_saturation_steps"]) == 0 for item in outcomes)
    no_fall = sum(item["post_abstention_fall"] is False for item in outcomes)
    failures: list[str] = []
    if not strict:
        failures.append("STRICT_REPLAY_FAILED")
    if passed_count != len(outcomes):
        failures.append("CONDITIONAL_RECOVERY_FAILURE")
    if zero_saturation != len(outcomes):
        failures.append("RECOVERY_SATURATION_OBSERVED")
    if no_fall != len(outcomes):
        failures.append("POST_ABSTENTION_FALL_OBSERVED")
    accepted = not failures
    unsigned = {
        "schema_version": "rosclaw.growth.g1_readiness_recovery_evaluation.v1",
        "router_hash": router.router_hash,
        "readiness_gate_hash": gate.gate_hash,
        "body_hash": router.body_hash,
        "implementation_hash": next(iter(implementation_hashes)),
        "seeds": list(seeds_tuple),
        "episode_count": len(outcomes),
        "passed_episodes": passed_count,
        "strict_replay_all": strict,
        "zero_saturation_episodes": zero_saturation,
        "no_fall_episodes": no_fall,
        "minimum_recovery_pelvis_height_m": min(
            float(item["recovery_min_pelvis_height_m"]) for item in outcomes
        ),
        "maximum_recovery_tilt_rad": max(
            float(item["recovery_peak_tilt_rad"]) for item in outcomes
        ),
        "maximum_final_speed_mps": max(float(item["final_speed_mps"]) for item in outcomes),
        "maximum_final_joint_velocity_rms_rad_s": max(
            float(item["final_joint_velocity_rms_rad_s"]) for item in outcomes
        ),
        "mean_initial_speed_mps": _mean(float(item["initial_speed_mps"]) for item in outcomes),
        "mean_final_speed_mps": _mean(float(item["final_speed_mps"]) for item in outcomes),
        "mean_initial_joint_velocity_rms_rad_s": _mean(
            float(item["initial_joint_velocity_rms_rad_s"]) for item in outcomes
        ),
        "mean_final_joint_velocity_rms_rad_s": _mean(
            float(item["final_joint_velocity_rms_rad_s"]) for item in outcomes
        ),
        "source_evidence_hashes": source_evidence_hashes,
        "source_trajectory_hashes": source_trajectory_hashes,
        "accepted": accepted,
        "failure_codes": failures,
        "conditional_contract": {
            "readiness_abstained": True,
            "actuator_saturation_steps": 0,
            "post_abstention_fall": False,
            "minimum_pelvis_height_m": 0.65,
            "maximum_peak_tilt_rad": 0.65,
            "minimum_final_pelvis_height_m": 0.70,
            "maximum_final_speed_mps": 0.20,
            "maximum_final_joint_velocity_rms_rad_s": 0.50,
        },
        "evidence_domain": "SIM_ONLY_FROZEN_CONDITIONAL_VALIDATION",
        "generalizes_to_non_abstained_states": False,
        "promotion_truth_allowed": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    report = G1ReadinessRecoveryEvaluation(
        router_hash=router.router_hash,
        readiness_gate_hash=gate.gate_hash,
        body_hash=router.body_hash,
        implementation_hash=next(iter(implementation_hashes)),
        seeds=seeds_tuple,
        episode_count=len(outcomes),
        passed_episodes=passed_count,
        strict_replay_all=strict,
        zero_saturation_episodes=zero_saturation,
        no_fall_episodes=no_fall,
        minimum_recovery_pelvis_height_m=float(
            unsigned["minimum_recovery_pelvis_height_m"]
        ),
        maximum_recovery_tilt_rad=float(unsigned["maximum_recovery_tilt_rad"]),
        maximum_final_speed_mps=float(unsigned["maximum_final_speed_mps"]),
        maximum_final_joint_velocity_rms_rad_s=float(
            unsigned["maximum_final_joint_velocity_rms_rad_s"]
        ),
        mean_initial_speed_mps=float(unsigned["mean_initial_speed_mps"]),
        mean_final_speed_mps=float(unsigned["mean_final_speed_mps"]),
        mean_initial_joint_velocity_rms_rad_s=float(
            unsigned["mean_initial_joint_velocity_rms_rad_s"]
        ),
        mean_final_joint_velocity_rms_rad_s=float(
            unsigned["mean_final_joint_velocity_rms_rad_s"]
        ),
        source_evidence_hashes=tuple(source_evidence_hashes),
        source_trajectory_hashes=tuple(source_trajectory_hashes),
        accepted=accepted,
        failure_codes=tuple(failures),
        report_hash=canonical_hash(unsigned),
    )
    if report.report_hash != canonical_hash(report.to_dict(include_hash=False)):
        raise RuntimeError("readiness recovery report hash construction diverged")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _episode_passed(result: dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(
        result.get("readiness_abstained") is True
        and result.get("finite_state") is True
        and result.get("post_abstention_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
        and int(result.get("actuator_saturation_steps", -1)) == 0
        and float(result["recovery_min_pelvis_height_m"])
        >= float(config["minimum_pelvis_height_m"])
        and float(result["recovery_peak_tilt_rad"])
        <= float(config["maximum_peak_tilt_rad"])
        and float(result["final_pelvis_height_m"]) >= 0.70
        and float(result["final_speed_mps"])
        <= float(config["maximum_final_speed_mps"])
        and float(result["final_joint_velocity_rms_rad_s"])
        <= float(config["maximum_final_joint_velocity_rms_rad_s"])
    )


def _mean(values: Any) -> float:
    items = tuple(float(item) for item in values)
    return sum(items) / len(items)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = ["G1ReadinessRecoveryEvaluation", "evaluate_g1_readiness_recovery"]
