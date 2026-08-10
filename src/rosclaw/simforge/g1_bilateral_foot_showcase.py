"""Strict physical left/right-foot corner-kick showcase evidence."""

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
from rosclaw.simforge.g1_two_player_relay import _base_scenario
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GoalForgeResult,
    ShotParameters,
    hash_bytes,
    hash_json,
)


@dataclass(frozen=True)
class G1BilateralFootCase:
    kick_foot: str
    declared_corner: str
    target_m: tuple[float, float, float]
    ball_start_y_m: float
    result: GoalForgeResult
    trajectory_path: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    schema_version: str = "rosclaw.g1_goalforge.bilateral_foot_case.v1"

    @property
    def passed(self) -> bool:
        return bool(
            self.strict_replay
            and self.result.physics_executed
            and self.result.contact_observed
            and self.result.kick_foot_contacted
            and self.result.goal_crossed
            and self.result.target_error_m <= 0.10
            and self.result.ball_speed_mps >= 6.0
            and not self.result.post_kick_fall
            and not self.result.joint_limit_violation
            and not self.result.torque_limit_violation
            and not self.result.actuator_saturation
            and self.result.finite_state
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "result": self.result.summary_dict(),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class G1BilateralFootEvidence:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    implementation_hash: str
    request_hash: str
    cases: tuple[G1BilateralFootCase, ...]
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.bilateral_foot_evidence.v1"

    @property
    def passed(self) -> bool:
        return bool(
            {case.kick_foot for case in self.cases} == {"left", "right"}
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
                "actual_left_foot_physics": True,
                "actual_right_foot_physics": True,
                "left_foot_uses_live_mirrored_proprioception": True,
                "left_foot_is_not_pixel_mirroring": True,
                "strict_replay_every_case": all(case.strict_replay for case in self.cases),
                "pixels_used_for_task_scoring": False,
                "real_hardware": False,
            },
        }


def bilateral_candidates() -> tuple[tuple[str, float, float, float, float], ...]:
    """Frozen safe candidates from the bounded physical landing-point search."""

    # foot, ball_y, target_y, target_z, foot_yaw. The declared target remains
    # frozen before rollout so inverse calibration cannot relabel an outcome.
    return (
        ("right", 0.12, 1.00, 0.14, -0.12),
        ("left", -0.24, -1.00, 0.20, 0.12),
    )


def run_g1_bilateral_foot_showcase(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> G1BilateralFootEvidence:
    """Run two real-contact SIM episodes and strict deterministic replays."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("bilateral-foot evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    implementation_hash = hash_json(
        {
            "showcase": hash_bytes(Path(__file__).read_bytes()),
            "backend": hash_bytes(
                Path(__file__)
                .with_name("backends")
                .joinpath("unitree_mujoco_backend.py")
                .read_bytes()
            ),
        }
    )
    request = {
        "schema_version": "rosclaw.g1_goalforge.bilateral_foot_request.v1",
        "body_hash": backend.qualification.body_hash,
        "kick_prior_hash": backend.qualification.kick_prior_hash,
        "implementation_hash": implementation_hash,
        "candidates": [
            {
                "kick_foot": foot,
                "ball_start_y_m": ball_y,
                "target_m": [5.0, target_y, target_z],
                "foot_yaw_offset_rad": foot_yaw,
            }
            for foot, ball_y, target_y, target_z, foot_yaw in bilateral_candidates()
        ],
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
    }
    request_path = root / "request.json"
    _write_json(request_path, request)
    base = _base_scenario()
    cases: list[G1BilateralFootCase] = []
    for foot, ball_y, target_y, target_z, foot_yaw in bilateral_candidates():
        scenario = replace(
            base,
            scenario_id=f"bilateral-{foot}-lower-corner",
            ball_y_m=ball_y,
            target_y_m=target_y,
            target_z_m=target_z,
        )
        parameters = ShotParameters(
            kick_foot=foot,
            foot_yaw_offset=foot_yaw,
            foot_pitch_offset=0.01,
            recovery_step_length=0.055,
            policy_type="parameter",
        )
        episode = backend.run(scenario, parameters)
        replay = backend.run(scenario, parameters)
        strict = bool(
            episode.result.summary_dict() == replay.result.summary_dict()
            and trajectory_digest(episode.trajectory) == trajectory_digest(replay.trajectory)
        )
        trajectory_path = root / f"{foot}-foot-trajectory.npz"
        np.savez_compressed(trajectory_path, **episode.trajectory)  # type: ignore[arg-type]
        cases.append(
            G1BilateralFootCase(
                kick_foot=foot,
                declared_corner="left_lower" if target_y > 0.0 else "right_lower",
                target_m=(5.0, target_y, target_z),
                ball_start_y_m=ball_y,
                result=episode.result,
                trajectory_path=str(trajectory_path),
                trajectory_hash=_file_hash(trajectory_path),
                trajectory_digest=trajectory_digest(episode.trajectory),
                strict_replay=strict,
            )
        )
    evidence = G1BilateralFootEvidence(
        body_hash=backend.qualification.body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        backend_commit=backend.qualification.backend_commit,
        implementation_hash=implementation_hash,
        request_hash=_file_hash(request_path),
        cases=tuple(cases),
    )
    _write_json(root / "g1-bilateral-foot-showcase.json", evidence.to_dict())
    return evidence


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "G1BilateralFootCase",
    "G1BilateralFootEvidence",
    "bilateral_candidates",
    "run_g1_bilateral_foot_showcase",
]
