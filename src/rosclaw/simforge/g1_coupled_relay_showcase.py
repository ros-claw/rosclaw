"""Five-challenge, strict-replay showcase for the coupled two-G1 relay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    trajectory_digest,
)
from rosclaw.simforge.g1_coupled_relay import (
    G1CoupledRelayResult,
    _simulate,
    _standby_policy_hash,
    coupled_runtime_manifest,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes, hash_json


@dataclass(frozen=True)
class G1CoupledShowcaseSpec:
    case_id: str
    title: str
    subtitle: str
    shooter_start_sec: float
    ball_ground_friction: float
    camera_azimuth_deg: float


@dataclass(frozen=True)
class G1CoupledShowcaseCase:
    spec: G1CoupledShowcaseSpec
    result: G1CoupledRelayResult
    trajectory_path: str
    trajectory_hash: str
    trajectory_digest: str
    strict_replay: bool
    schema_version: str = "rosclaw.g1_goalforge.coupled_showcase_case.v1"

    @property
    def passed(self) -> bool:
        return bool(self.strict_replay and self.result.passed)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["result"] = self.result.to_dict()
        value["passed"] = self.passed
        return value


@dataclass(frozen=True)
class G1CoupledShowcaseEvidence:
    body_hash: str
    kick_prior_hash: str
    standby_policy_hash: str
    backend_commit: str
    implementation_hash: str
    request_hash: str
    cases: tuple[G1CoupledShowcaseCase, ...]
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    environment_hash: str = ""
    schema_version: str = "rosclaw.g1_goalforge.coupled_showcase_evidence.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.cases) == 5
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
                "five_independent_physics_challenges": True,
                "strict_replay_every_challenge": all(case.strict_replay for case in self.cases),
                "simultaneous_two_body_physics": True,
                "single_shared_ball_per_challenge": True,
                "pixels_used_for_promotion": False,
                "real_hardware": False,
            },
        }


def showcase_specs() -> tuple[G1CoupledShowcaseSpec, ...]:
    return (
        G1CoupledShowcaseSpec(
            case_id="01-early-arrival-reflex",
            title="EARLY ARRIVAL REFLEX",
            subtitle="-40 ms · ONLINE HOLD 2 FRAMES · SKY-HIGH FINISH",
            shooter_start_sec=1.98,
            ball_ground_friction=0.10,
            camera_azimuth_deg=84.0,
        ),
        G1CoupledShowcaseSpec(
            case_id="02-slick-pitch-speed",
            title="SLICK PITCH SPEED",
            subtitle="LOW FRICTION · FAST ROLL · ONE-TOUCH FINISH",
            shooter_start_sec=2.02,
            ball_ground_friction=0.05,
            camera_azimuth_deg=89.0,
        ),
        G1CoupledShowcaseSpec(
            case_id="03-high-target-precision",
            title="HIGH TARGET PRECISION",
            subtitle="1.09 m TARGET · STRICT REPLAY CENTERPIECE",
            shooter_start_sec=2.02,
            ball_ground_friction=0.10,
            camera_azimuth_deg=94.0,
        ),
        G1CoupledShowcaseSpec(
            case_id="04-grippy-pitch-control",
            title="GRIPPY PITCH CONTROL",
            subtitle="HIGH FRICTION · CONTROLLED RECEIVE · CLEAN RECOVERY",
            shooter_start_sec=2.02,
            ball_ground_friction=0.15,
            camera_azimuth_deg=99.0,
        ),
        G1CoupledShowcaseSpec(
            case_id="05-late-arrival-acceleration",
            title="LATE ARRIVAL ACCELERATION",
            subtitle="+40 ms · ONLINE ADVANCE 2 FRAMES · CLOSEST HIT",
            shooter_start_sec=2.06,
            ball_ground_friction=0.10,
            camera_azimuth_deg=104.0,
        ),
    )


def run_g1_coupled_showcase(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> G1CoupledShowcaseEvidence:
    """Run five independent challenges and persist strict physics traces."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("coupled showcase evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    coupled_source = Path(__file__).with_name("g1_coupled_relay.py")
    implementation_hash = hash_json(
        {
            "showcase": hash_bytes(Path(__file__).read_bytes()),
            "coupled_runtime": hash_bytes(coupled_source.read_bytes()),
        }
    )
    specs = showcase_specs()
    request = {
        "schema_version": "rosclaw.g1_goalforge.coupled_showcase_request.v1",
        "body_hash": backend.qualification.body_hash,
        "kick_prior_hash": backend.qualification.kick_prior_hash,
        "standby_policy_hash": _standby_policy_hash(asset_root.expanduser().resolve()),
        "implementation_hash": implementation_hash,
        "cases": [asdict(spec) for spec in specs],
        "target_m": [5.0, 1.10, 1.09],
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "runtime": coupled_runtime_manifest(),
    }
    request["environment_hash"] = hash_json(request["runtime"])
    request_path = root / "request.json"
    _write_json(request_path, request)
    cases: list[G1CoupledShowcaseCase] = []
    for spec in specs:
        result, trajectory = _simulate(
            asset_root,
            backend,
            shooter_start_sec=spec.shooter_start_sec,
            ball_ground_friction=spec.ball_ground_friction,
        )
        replay_result, replay_trajectory = _simulate(
            asset_root,
            backend,
            shooter_start_sec=spec.shooter_start_sec,
            ball_ground_friction=spec.ball_ground_friction,
        )
        strict = bool(
            result.to_dict() == replay_result.to_dict()
            and trajectory_digest(trajectory) == trajectory_digest(replay_trajectory)
        )
        trajectory_path = root / f"{spec.case_id}.npz"
        np.savez_compressed(trajectory_path, **trajectory)
        cases.append(
            G1CoupledShowcaseCase(
                spec=spec,
                result=result,
                trajectory_path=str(trajectory_path),
                trajectory_hash=_file_hash(trajectory_path),
                trajectory_digest=trajectory_digest(trajectory),
                strict_replay=strict,
            )
        )
    evidence = G1CoupledShowcaseEvidence(
        body_hash=backend.qualification.body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        standby_policy_hash=request["standby_policy_hash"],
        backend_commit=backend.qualification.backend_commit,
        implementation_hash=implementation_hash,
        request_hash=hash_bytes(request_path.read_bytes()),
        cases=tuple(cases),
        environment_hash=str(request["environment_hash"]),
    )
    _write_json(root / "g1-coupled-showcase.json", evidence.to_dict())
    return evidence


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "G1CoupledShowcaseCase",
    "G1CoupledShowcaseEvidence",
    "G1CoupledShowcaseSpec",
    "run_g1_coupled_showcase",
    "showcase_specs",
]
