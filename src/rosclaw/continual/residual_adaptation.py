"""Backend-neutral contracts for stability-preserving residual adaptation."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")


def _require_hash(label: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def _require_selectors(label: str, values: tuple[str, ...]) -> None:
    if (
        not values
        or len(set(values)) != len(values)
        or any(not value.strip() or "\n" in value for value in values)
    ):
        raise ValueError(f"{label} must contain unique non-empty selectors")


@dataclass(frozen=True)
class ResidualAdaptationContract:
    """Seal what can learn while a proven parent remains immutable."""

    run_id: str
    backend_contract_hash: str
    parent_artifact_hash: str
    body_hash: str
    rehearsal_dataset_hash: str
    acquisition_dataset_hash: str
    frozen_parameter_selectors: tuple[str, ...]
    trainable_parameter_selectors: tuple[str, ...]
    device_ids: tuple[int, ...]
    maximum_world_steps: int
    policy_learning_rate: float
    rehearsal_fraction: float
    acquisition_fraction: float
    maximum_residual_output_rms: float
    maximum_frozen_parameter_drift: float = 0.0
    sim_only: bool = True
    hardware_execution_allowed: bool = False
    schema_version: str = "rosclaw.continual.residual_adaptation_contract.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.run_id):
            raise ValueError("run_id is not a normalized identifier")
        for label, value in (
            ("backend_contract_hash", self.backend_contract_hash),
            ("parent_artifact_hash", self.parent_artifact_hash),
            ("body_hash", self.body_hash),
            ("rehearsal_dataset_hash", self.rehearsal_dataset_hash),
            ("acquisition_dataset_hash", self.acquisition_dataset_hash),
        ):
            _require_hash(label, value)
        _require_selectors("frozen_parameter_selectors", self.frozen_parameter_selectors)
        _require_selectors("trainable_parameter_selectors", self.trainable_parameter_selectors)
        overlap = set(self.frozen_parameter_selectors) & set(self.trainable_parameter_selectors)
        if overlap:
            raise ValueError("frozen and trainable parameter selectors must be disjoint")
        if (
            not self.device_ids
            or len(set(self.device_ids)) != len(self.device_ids)
            or any(device < 0 for device in self.device_ids)
        ):
            raise ValueError("device_ids must be unique non-negative identifiers")
        if self.maximum_world_steps <= 0:
            raise ValueError("maximum_world_steps must be positive")
        for label, numeric_value in (
            ("policy_learning_rate", self.policy_learning_rate),
            ("maximum_residual_output_rms", self.maximum_residual_output_rms),
        ):
            if not math.isfinite(numeric_value) or numeric_value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
        if (
            not 0.0 < self.rehearsal_fraction < 1.0
            or not 0.0 < self.acquisition_fraction < 1.0
            or not math.isclose(
                self.rehearsal_fraction + self.acquisition_fraction,
                1.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("rehearsal and acquisition fractions must be positive and sum to one")
        if (
            not math.isfinite(self.maximum_frozen_parameter_drift)
            or self.maximum_frozen_parameter_drift < 0.0
        ):
            raise ValueError("maximum_frozen_parameter_drift must be finite and non-negative")
        if not self.sim_only or self.hardware_execution_allowed:
            raise ValueError("residual adaptation contracts are SIM_ONLY")

    @property
    def contract_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["frozen_parameter_selectors"] = sorted(self.frozen_parameter_selectors)
        value["trainable_parameter_selectors"] = sorted(self.trainable_parameter_selectors)
        value["device_ids"] = list(self.device_ids)
        return value


@dataclass(frozen=True)
class ParameterIsolationEvidence:
    """Post-training proof that only the contracted residual scope changed."""

    adaptation_contract_hash: str
    parent_artifact_hash: str
    candidate_artifact_hash: str
    frozen_base_hash_before: str
    frozen_base_hash_after: str
    matched_exam_hash: str
    examined_frozen_parameter_count: int
    examined_trainable_parameter_count: int
    candidate_world_steps: int
    maximum_frozen_parameter_drift: float
    residual_output_rms: float
    retention_passed: bool
    acquisition_passed: bool
    critical_safety_regressions: int
    sim_only: bool = True
    schema_version: str = "rosclaw.continual.parameter_isolation_evidence.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("adaptation_contract_hash", self.adaptation_contract_hash),
            ("parent_artifact_hash", self.parent_artifact_hash),
            ("candidate_artifact_hash", self.candidate_artifact_hash),
            ("frozen_base_hash_before", self.frozen_base_hash_before),
            ("frozen_base_hash_after", self.frozen_base_hash_after),
            ("matched_exam_hash", self.matched_exam_hash),
        ):
            _require_hash(label, value)
        if self.parent_artifact_hash == self.candidate_artifact_hash:
            raise ValueError("residual candidate must be a new artifact")
        if min(self.examined_frozen_parameter_count, self.examined_trainable_parameter_count) <= 0:
            raise ValueError("parameter-isolation evidence must inspect both scopes")
        if self.candidate_world_steps <= 0:
            raise ValueError("candidate_world_steps must be positive")
        if self.critical_safety_regressions < 0:
            raise ValueError("critical_safety_regressions must be non-negative")
        for label, numeric_value in (
            ("maximum_frozen_parameter_drift", self.maximum_frozen_parameter_drift),
            ("residual_output_rms", self.residual_output_rms),
        ):
            if not math.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        if not self.sim_only:
            raise ValueError("parameter-isolation evidence is SIM_ONLY")

    def passes(self, contract: ResidualAdaptationContract) -> bool:
        """Fail closed unless identity, isolation, safety, and both suites pass."""

        return bool(
            self.adaptation_contract_hash == contract.contract_hash
            and self.parent_artifact_hash == contract.parent_artifact_hash
            and self.frozen_base_hash_before == self.frozen_base_hash_after
            and self.maximum_frozen_parameter_drift <= contract.maximum_frozen_parameter_drift
            and self.residual_output_rms <= contract.maximum_residual_output_rms
            and self.candidate_world_steps <= contract.maximum_world_steps
            and self.retention_passed
            and self.acquisition_passed
            and self.critical_safety_regressions == 0
        )

    @property
    def evidence_hash(self) -> str:
        return canonical_hash(asdict(self))


def write_residual_adaptation_contract(
    contract: ResidualAdaptationContract, output_path: Path
) -> dict[str, Any]:
    """Atomically seal a new residual-learning contract before evaluation."""

    output = output_path.expanduser().resolve()
    if output.suffix != ".json" or output.exists():
        raise ValueError("residual adaptation contract requires a new JSON output")
    payload = contract.to_dict()
    payload["contract_hash"] = contract.contract_hash
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return payload


def load_residual_adaptation_contract(path: Path) -> ResidualAdaptationContract:
    """Load and verify a previously sealed residual-learning contract."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    expected = set(ResidualAdaptationContract.__dataclass_fields__) | {"contract_hash"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("residual adaptation contract fields are incomplete")
    expected_hash = payload.pop("contract_hash")
    for key in (
        "frozen_parameter_selectors",
        "trainable_parameter_selectors",
        "device_ids",
    ):
        if not isinstance(payload[key], list):
            raise ValueError("residual adaptation contract tuple field must be a list")
        payload[key] = tuple(payload[key])
    contract = ResidualAdaptationContract(**payload)
    if expected_hash != contract.contract_hash:
        raise ValueError("residual adaptation contract hash does not match its contents")
    return contract


__all__ = [
    "ParameterIsolationEvidence",
    "ResidualAdaptationContract",
    "load_residual_adaptation_contract",
    "write_residual_adaptation_contract",
]
