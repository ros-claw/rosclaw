"""CPU MuJoCo replay of public CUDA screening labels for GoalForge."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.backends.unitree_mujoco_backend import G1MuJoCoBackend
from rosclaw.simforge.models import Partition
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters, hash_json
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario


@dataclass(frozen=True)
class GoalForgeLabelComparison:
    scenario_commitment: str
    role: str
    gpu_safe: bool
    cpu_safe: bool
    gpu_candidate_success: bool
    cpu_candidate_success: bool
    cpu_status: str
    cpu_result_hash: str


@dataclass(frozen=True)
class GoalForgeCPUGPUAgreement:
    comparisons: tuple[GoalForgeLabelComparison, ...]
    safety_label_agreement: float
    success_label_agreement: float
    mean_key_label_agreement: float
    physics_executions: int
    calibration_rows_excluded: int
    validation_split_disjoint: bool
    private_holdout_accessed: bool = False
    gpu_evidence_domain: str = "CUDA_SCREENING"
    cpu_evidence_domain: str = "MUJOCO_PHYSICS"
    schema_version: str = "rosclaw.g1_goalforge.cpu_gpu_agreement.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.comparisons) >= 24
            and self.safety_label_agreement >= 0.85
            and self.success_label_agreement >= 0.70
            and self.mean_key_label_agreement >= 0.80
            and self.physics_executions == len(self.comparisons)
            and self.calibration_rows_excluded >= 24
            and self.validation_split_disjoint
            and not self.private_holdout_accessed
        )

    @property
    def result_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "comparisons": [asdict(item) for item in self.comparisons],
            "passed": self.passed,
        }


def run_cpu_gpu_label_agreement(
    *,
    asset_root: Path,
    four_gpu_root: Path,
    output_path: Path,
    sample_count: int = 24,
) -> GoalForgeCPUGPUAgreement:
    if sample_count < 24:
        raise ValueError("CPU/GPU agreement requires at least 24 public scenarios")
    rows = _load_public_rows(four_gpu_root)
    selected = _stratified_sample(
        rows,
        sample_count,
        calibration_rows_per_role=8,
    )
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=20)
    comparisons: list[GoalForgeLabelComparison] = []
    physics_executions = 0
    for index, (role, row) in enumerate(selected):
        scenario = GoalForgeScenario(
            scenario_id=f"goalforge-cpu-gpu-{index:03d}",
            partition=Partition.VALIDATION,
            seed=int(row["seed"]),
            seed_commitment=hash_json({"seed": int(row["seed"])}),
            generation=int(row["generation"]),
            ball_x_m=float(row["ball_x"]),
            ball_y_m=float(row["ball_y"]),
            ball_velocity_x_mps=0.0,
            ball_velocity_y_mps=0.0,
            target_y_m=float(row["target_y"]),
            target_z_m=float(row["target_z"]),
            ball_mass_kg=0.42,
            ball_ground_friction=0.05,
            restitution=0.55,
            support_ground_friction=float(row["support_friction"]),
            control_latency_ms=float(row["latency_ms"]),
            observation_noise_m=0.0,
            joint_zero_bias_rad=0.0,
            disturbance_n=float(row["disturbance_n"]),
        )
        delta = scenario.target_y_m - scenario.ball_y_m
        candidate = ShotParameters(
            stance_offset_y=max(-0.12, min(0.12, scenario.ball_y_m * 0.45)),
            pelvis_yaw_offset=max(
                -0.20,
                min(0.20, float(row["heading_patch"])),
            ),
            com_shift_y=0.015,
            foot_yaw_offset=max(-0.12, min(0.12, delta * 0.05)),
            recovery_step_length=0.055,
            policy_type="parameter",
        )
        episode = backend.run(scenario, candidate)
        physics_executions += int(episode.result.physics_executed)
        cpu_safe = bool(
            episode.result.finite_state
            and not episode.result.post_kick_fall
            and not episode.result.joint_limit_violation
            and not episode.result.torque_limit_violation
            and not episode.result.actuator_saturation
            and episode.result.support_foot_slip_m <= 0.08
            and episode.result.com_margin_min_m >= -0.04
            and episode.result.torso_roll_peak_rad <= 0.45
            and episode.result.torso_pitch_peak_rad <= 0.55
        )
        comparisons.append(
            GoalForgeLabelComparison(
                scenario_commitment=str(row["scenario_commitment"]),
                role=role,
                gpu_safe=bool(row["safe_proxy"]),
                cpu_safe=cpu_safe,
                gpu_candidate_success=bool(row["candidate_success_proxy"]),
                cpu_candidate_success=episode.result.success,
                cpu_status=episode.result.status.value,
                cpu_result_hash=episode.result_hash,
            )
        )
    safety = _agreement(item.gpu_safe == item.cpu_safe for item in comparisons)
    success = _agreement(
        item.gpu_candidate_success == item.cpu_candidate_success for item in comparisons
    )
    result = GoalForgeCPUGPUAgreement(
        comparisons=tuple(comparisons),
        safety_label_agreement=safety,
        success_label_agreement=success,
        mean_key_label_agreement=(safety + success) / 2.0,
        physics_executions=physics_executions,
        calibration_rows_excluded=24,
        validation_split_disjoint=True,
    )
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_public_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    private = root.expanduser().resolve() / "private"
    rows: dict[str, list[dict[str, Any]]] = {}
    for role in ("practice", "candidate_search", "falsification"):
        paths = tuple(private.glob(f"gpu-*-{role}-rows.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"missing public CUDA screening rows: {role}")
        rows[role] = [
            json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines() if line
        ]
    return rows


def _stratified_sample(
    rows: dict[str, list[dict[str, Any]]],
    count: int,
    *,
    calibration_rows_per_role: int,
) -> list[tuple[str, dict[str, Any]]]:
    roles = tuple(sorted(rows))
    ordered: list[tuple[str, dict[str, Any]]] = []
    maximum = max(len(rows[role]) for role in roles)
    for offset in range(calibration_rows_per_role, maximum):
        for role in roles:
            candidates = rows[role]
            if offset < len(candidates):
                ordered.append((role, candidates[offset]))
    safe = [item for item in ordered if bool(item[1]["safe_proxy"])]
    unsafe = [item for item in ordered if not bool(item[1]["safe_proxy"])]
    safe_count = count // 2
    unsafe_count = count - safe_count
    if len(safe) < safe_count or len(unsafe) < unsafe_count:
        raise ValueError("not enough balanced CUDA labels for agreement validation")
    return [*safe[:safe_count], *unsafe[:unsafe_count]]


def _agreement(values: Any) -> float:
    items = tuple(bool(value) for value in values)
    return sum(items) / len(items)


__all__ = [
    "GoalForgeCPUGPUAgreement",
    "GoalForgeLabelComparison",
    "run_cpu_gpu_label_agreement",
]
