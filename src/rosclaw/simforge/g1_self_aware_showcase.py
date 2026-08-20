"""Strict-replay moving-ball showcase for the Phase 8 self-aware G1 skill."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    trajectory_digest,
)
from rosclaw.simforge.g1_failure_curriculum_validation import (
    audit_g1_trajectory,
    build_g1_failure_curriculum,
    calibrate_g1_regime_belief,
    run_g1_contextual_candidate,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes, hash_json
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario


@dataclass(frozen=True)
class G1SelfAwareChallengeSpec:
    case_id: str
    title: str
    subtitle: str
    scenario: GoalForgeScenario
    camera_azimuth_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "scenario": self.scenario.to_private_dict(),
            "camera_azimuth_deg": self.camera_azimuth_deg,
        }


@dataclass(frozen=True)
class G1SelfAwareChallengeCase:
    spec: G1SelfAwareChallengeSpec
    result: dict[str, Any]
    belief_receipt: dict[str, Any]
    guard_receipt: dict[str, Any]
    goal_crossing_xyz_m: tuple[float, float, float]
    trajectory_path: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    quality_accepted: bool
    schema_version: str = "rosclaw.g1_goalforge.self_aware_showcase_case.v1"

    @property
    def passed(self) -> bool:
        return bool(
            self.strict_replay
            and self.quality_accepted
            and self.result["success"]
            and not self.result["post_kick_fall"]
            and not self.result["joint_limit_violation"]
            and not self.result["torque_limit_violation"]
            and not self.result["actuator_saturation"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "spec": self.spec.to_dict(),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class G1SelfAwareShowcaseEvidence:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    implementation_hash: str
    request_hash: str
    cases: tuple[G1SelfAwareChallengeCase, ...]
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "DEVELOPMENT_SHOWCASE"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.self_aware_showcase_evidence.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.cases) == 3
            and all(case.passed for case in self.cases)
            and self.activation_ceiling == "SIM_ONLY"
            and not self.hardware_command_sent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "cases": [case.to_dict() for case in self.cases],
            "passed": self.passed,
            "claims": {
                "three_independent_moving_ball_challenges": True,
                "strict_replay_every_challenge": all(
                    case.strict_replay for case in self.cases
                ),
                "development_selected_showcase": True,
                "sealed_generalization_evidence": False,
                "promotion_evidence": False,
                "pixels_used_for_promotion": False,
                "real_hardware": False,
            },
        }


def self_aware_showcase_specs() -> tuple[G1SelfAwareChallengeSpec, ...]:
    """Return the frozen development-only showcase challenges."""

    base = next(
        case.scenario
        for case in build_g1_failure_curriculum()
        if case.case_id == "development-difficult-g7"
    )

    def scenario(label: str, seed: int, **changes: Any) -> GoalForgeScenario:
        return replace(
            base,
            scenario_id=f"self-aware-showcase-{label}",
            partition=Partition.DEVELOPMENT,
            seed=seed,
            seed_commitment=hash_json(
                {"showcase": "g1-self-aware-v1", "label": label, "seed": seed}
            ),
            generation=9,
            **changes,
        )

    return (
        G1SelfAwareChallengeSpec(
            case_id="01-moving-ball-intercept",
            title="MOVING BALL INTERCEPT",
            subtitle="-0.10 m/s incoming · 30 N body push · 7 ms latency",
            scenario=scenario(
                "moving-ball-intercept",
                9701,
                target_y_m=0.75,
                target_z_m=0.55,
                ball_velocity_x_mps=-0.10,
                ball_velocity_y_mps=0.02,
                ball_launch_delay_sec=3.8,
                support_ground_friction=0.82,
                control_latency_ms=7.0,
                joint_zero_bias_rad=0.012,
                disturbance_n=30.0,
            ),
            camera_azimuth_deg=88.0,
        ),
        G1SelfAwareChallengeSpec(
            case_id="02-fast-moving-ball",
            title="FAST MOVING BALL",
            subtitle="-0.18 m/s incoming · lateral drift · 32 N body push",
            scenario=scenario(
                "fast-moving-ball",
                9711,
                target_y_m=0.70,
                target_z_m=0.55,
                ball_velocity_x_mps=-0.18,
                ball_velocity_y_mps=-0.04,
                ball_launch_delay_sec=3.7,
                support_ground_friction=0.82,
                control_latency_ms=7.0,
                joint_zero_bias_rad=0.012,
                disturbance_n=32.0,
            ),
            camera_azimuth_deg=96.0,
        ),
        G1SelfAwareChallengeSpec(
            case_id="03-friction-edge-combo",
            title="FRICTION-EDGE COMBO",
            subtitle="moving ball · friction 0.78 · 8 ms latency · 32 N push",
            scenario=scenario(
                "friction-edge-combo",
                9714,
                target_y_m=0.75,
                target_z_m=0.55,
                ball_velocity_x_mps=-0.10,
                ball_velocity_y_mps=0.02,
                ball_launch_delay_sec=3.8,
                support_ground_friction=0.78,
                control_latency_ms=8.0,
                joint_zero_bias_rad=0.018,
                disturbance_n=32.0,
            ),
            camera_azimuth_deg=104.0,
        ),
    )


def run_g1_self_aware_showcase(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> G1SelfAwareShowcaseEvidence:
    """Execute three selected development challenges with strict replay."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("self-aware showcase evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    trajectories = root / "trajectories"
    trajectories.mkdir()
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    implementation_hash = hash_json(
        {
            "showcase": hash_bytes(Path(__file__).read_bytes()),
            "failure_curriculum": hash_bytes(
                Path(__file__).with_name("g1_failure_curriculum_validation.py").read_bytes()
            ),
            "joint_guard": hash_bytes(
                Path(__file__).with_name("g1_joint_boundary_guard.py").read_bytes()
            ),
        }
    )
    specs = self_aware_showcase_specs()
    request = {
        "schema_version": "rosclaw.g1_goalforge.self_aware_showcase_request.v1",
        "body_hash": backend.qualification.body_hash,
        "kick_prior_hash": backend.qualification.kick_prior_hash,
        "implementation_hash": implementation_hash,
        "activation_ceiling": "SIM_ONLY",
        "evidence_domain": "DEVELOPMENT_SHOWCASE",
        "cases": [spec.to_dict() for spec in specs],
    }
    request_path = root / "request.json"
    _write_json(request_path, request)
    cases: list[G1SelfAwareChallengeCase] = []
    for spec in specs:
        episode = run_g1_contextual_candidate(backend, spec.scenario)
        replay = run_g1_contextual_candidate(backend, spec.scenario)
        digest = trajectory_digest(episode.trajectory)
        strict_replay = bool(
            episode.result.summary_dict() == replay.result.summary_dict()
            and digest == trajectory_digest(replay.trajectory)
        )
        audit = audit_g1_trajectory(episode)
        trajectory_path = trajectories / f"{spec.case_id}.npz"
        np.savez_compressed(trajectory_path, **episode.trajectory)
        crossing = _goal_crossing(episode.trajectory)
        if episode.torque_policy_receipt is None:
            raise RuntimeError("self-aware challenge did not emit a joint-guard receipt")
        cases.append(
            G1SelfAwareChallengeCase(
                spec=spec,
                result=episode.result.summary_dict(),
                belief_receipt=calibrate_g1_regime_belief(spec.scenario).to_dict(),
                guard_receipt=episode.torque_policy_receipt.to_dict(),
                goal_crossing_xyz_m=crossing,
                trajectory_path=str(trajectory_path),
                trajectory_hash=_file_hash(trajectory_path),
                trajectory_digest=digest,
                strict_replay=strict_replay,
                quality_accepted=audit.accepted_for_learning,
            )
        )
    evidence = G1SelfAwareShowcaseEvidence(
        body_hash=backend.qualification.body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        backend_commit=backend.qualification.backend_commit,
        implementation_hash=implementation_hash,
        request_hash=hash_bytes(request_path.read_bytes()),
        cases=tuple(cases),
    )
    _write_json(root / "g1-self-aware-showcase.json", evidence.to_dict())
    return evidence


def _goal_crossing(trajectory: dict[str, np.ndarray]) -> tuple[float, float, float]:
    ball = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    crossing = np.flatnonzero(ball[:, 0] >= 5.0)
    if not crossing.size:
        return (float("nan"), float("nan"), float("nan"))
    position = ball[int(crossing[0]), :3]
    return (float(position[0]), float(position[1]), float(position[2]))


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "G1SelfAwareChallengeCase",
    "G1SelfAwareChallengeSpec",
    "G1SelfAwareShowcaseEvidence",
    "run_g1_self_aware_showcase",
    "self_aware_showcase_specs",
]
