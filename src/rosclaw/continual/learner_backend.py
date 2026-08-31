"""Backend-neutral contracts for external continual-learning engines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")


class LearnerCapability(StrEnum):
    MOTION_TRACKING = "motion_tracking"
    ADVERSARIAL_MOTION_PRIOR = "adversarial_motion_prior"
    SPECIALIST_TRAINING = "specialist_training"
    GENERALIST_DISTILLATION = "generalist_distillation"
    PERSONAL_ADAPTER = "personal_adapter"
    MULTI_GPU = "multi_gpu"
    SIM2SIM = "sim2sim"


class LearnerRunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class LearnerBackendContract:
    """Pinned capabilities and trust boundary of one learner implementation."""

    backend_id: str
    backend_version: str
    source_url: str
    source_commit: str
    license_id: str
    capabilities: tuple[LearnerCapability, ...]
    supported_body_ids: tuple[str, ...]
    training_available: bool
    inference_available: bool
    external_environment: bool = True
    hardware_execution_allowed: bool = False
    sim_only: bool = True
    schema_version: str = "rosclaw.continual.learner_backend.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("backend_id", self.backend_id),
            ("backend_version", self.backend_version),
            ("license_id", self.license_id),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"{label} is not a normalized identifier")
        if not self.source_url.startswith("https://"):
            raise ValueError("source_url must use https")
        if not _COMMIT.fullmatch(self.source_commit):
            raise ValueError("source_commit must be a pinned hexadecimal commit")
        if not self.capabilities or len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("capabilities must be non-empty and unique")
        if (
            not self.supported_body_ids
            or len(set(self.supported_body_ids)) != len(self.supported_body_ids)
            or any(not _IDENTIFIER.fullmatch(item) for item in self.supported_body_ids)
        ):
            raise ValueError("supported_body_ids must be non-empty unique identifiers")
        if not self.training_available and not self.inference_available:
            raise ValueError("backend must expose training or inference")
        if not self.external_environment:
            raise ValueError("learner backends must remain outside the control-plane environment")
        if self.hardware_execution_allowed or not self.sim_only:
            raise ValueError("learner backend contracts are SIM_ONLY and cannot execute hardware")

    @property
    def contract_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "source_url": self.source_url,
            "source_commit": self.source_commit,
            "license_id": self.license_id,
            "capabilities": sorted(item.value for item in self.capabilities),
            "supported_body_ids": sorted(self.supported_body_ids),
            "training_available": self.training_available,
            "inference_available": self.inference_available,
            "external_environment": self.external_environment,
            "hardware_execution_allowed": self.hardware_execution_allowed,
            "sim_only": self.sim_only,
        }


@dataclass(frozen=True)
class LearnerRunEvidence:
    """Content-addressed receipt for a backend run; never executes the backend."""

    run_id: str
    backend_contract_hash: str
    body_hash: str
    dataset_manifest_hash: str
    config_hash: str
    seed_commitment_hash: str
    physics_backend: str
    device_ids: tuple[int, ...]
    world_steps: int
    sample_count: int
    status: LearnerRunStatus
    candidate_artifact_hash: str | None = None
    sim_only: bool = True
    schema_version: str = "rosclaw.continual.learner_run_evidence.v1"

    def __post_init__(self) -> None:
        for label, value in (("run_id", self.run_id), ("physics_backend", self.physics_backend)):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"{label} is not a normalized identifier")
        for label, value in (
            ("backend_contract_hash", self.backend_contract_hash),
            ("body_hash", self.body_hash),
            ("dataset_manifest_hash", self.dataset_manifest_hash),
            ("config_hash", self.config_hash),
            ("seed_commitment_hash", self.seed_commitment_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256: content hash")
        if self.candidate_artifact_hash is not None and not _SHA256.fullmatch(
            self.candidate_artifact_hash
        ):
            raise ValueError("candidate_artifact_hash must be a sha256: content hash")
        if len(set(self.device_ids)) != len(self.device_ids) or any(
            device < 0 for device in self.device_ids
        ):
            raise ValueError("device_ids must contain unique non-negative integers")
        if self.world_steps < 0 or self.sample_count < 0:
            raise ValueError("run counts must be non-negative")
        if self.status is LearnerRunStatus.COMPLETED:
            if (
                self.candidate_artifact_hash is None
                or self.world_steps == 0
                or self.sample_count == 0
            ):
                raise ValueError("completed run requires samples, steps, and a candidate artifact")
        elif self.candidate_artifact_hash is not None:
            raise ValueError("only a completed run can publish a candidate artifact")
        if not self.sim_only:
            raise ValueError("learner run evidence is SIM_ONLY until a separate real-body gate")

    @property
    def evidence_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "backend_contract_hash": self.backend_contract_hash,
            "body_hash": self.body_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "config_hash": self.config_hash,
            "seed_commitment_hash": self.seed_commitment_hash,
            "physics_backend": self.physics_backend,
            "device_ids": list(self.device_ids),
            "world_steps": self.world_steps,
            "sample_count": self.sample_count,
            "status": self.status.value,
            "candidate_artifact_hash": self.candidate_artifact_hash,
            "sim_only": self.sim_only,
        }


__all__ = [
    "LearnerBackendContract",
    "LearnerCapability",
    "LearnerRunEvidence",
    "LearnerRunStatus",
]
