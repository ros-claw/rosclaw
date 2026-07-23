"""Causal module-proof contracts for the Failure-to-Success Arena.

The proof level is derived from evidence fields.  Callers cannot assign E3/E5
directly, which prevents an import, invocation, or self-reported success flag
from being relabelled as causal or replayable evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

_MODULE_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_REF_RE = re.compile(r"^[a-z][a-z0-9+.-]*://[^ \t\r\n]+$")


class ModuleEvidenceLevel(StrEnum):
    EXISTS = "E0"
    INVOKED = "E1"
    VALID_OUTPUT = "E2"
    DECISION_IMPACT = "E3"
    FAULT_TOLERANT = "E4"
    REPLAY_VERIFIED = "E5"

    @property
    def rank(self) -> int:
        return int(self.value[1:])


@dataclass(frozen=True)
class CounterfactualMetric:
    name: str
    control: float
    treatment: float
    lower_is_better: bool

    def __post_init__(self) -> None:
        if not _MODULE_RE.fullmatch(self.name):
            raise ValueError("counterfactual metric name must be a safe identifier")
        if not math.isfinite(self.control) or not math.isfinite(self.treatment):
            raise ValueError("counterfactual metrics must be finite")

    @property
    def improvement(self) -> float:
        delta = self.control - self.treatment
        return delta if self.lower_is_better else -delta


@dataclass(frozen=True)
class CounterfactualRun:
    """Matched control/treatment evidence for one module intervention."""

    control_run_id: str
    treatment_run_id: str
    same_seed: bool
    same_scenario: bool
    same_body_hash: bool
    decision_changed: bool
    outcome_changed: bool
    metrics: tuple[CounterfactualMetric, ...]
    control_ref: str
    treatment_ref: str

    def __post_init__(self) -> None:
        if not self.control_run_id or not self.treatment_run_id:
            raise ValueError("counterfactual run ids are required")
        if self.control_run_id == self.treatment_run_id:
            raise ValueError("control and treatment run ids must differ")
        if not self.metrics:
            raise ValueError("counterfactual evidence requires at least one metric")
        _validate_refs((self.control_ref, self.treatment_ref))

    @property
    def matched(self) -> bool:
        return self.same_seed and self.same_scenario and self.same_body_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_run_id": self.control_run_id,
            "treatment_run_id": self.treatment_run_id,
            "same_seed": self.same_seed,
            "same_scenario": self.same_scenario,
            "same_body_hash": self.same_body_hash,
            "decision_changed": self.decision_changed,
            "outcome_changed": self.outcome_changed,
            "metrics": [
                {
                    **asdict(metric),
                    "improvement": metric.improvement,
                }
                for metric in self.metrics
            ],
            "control_ref": self.control_ref,
            "treatment_ref": self.treatment_ref,
        }


@dataclass(frozen=True)
class FaultInjectionResult:
    name: str
    passed: bool
    evidence_ref: str

    def __post_init__(self) -> None:
        if not _MODULE_RE.fullmatch(self.name):
            raise ValueError("fault injection name must be a safe identifier")
        _validate_refs((self.evidence_ref,))


@dataclass(frozen=True)
class ModuleProof:
    module: str
    invoked: bool
    input_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()
    output_valid: bool = False
    decision_impacts: tuple[str, ...] = ()
    counterfactual: CounterfactualRun | None = None
    fault_injections: tuple[FaultInjectionResult, ...] = ()
    replay_verified: bool = False
    replay_ref: str | None = None
    schema_version: str = "rosclaw.module_proof.v1"

    def __post_init__(self) -> None:
        if not _MODULE_RE.fullmatch(self.module):
            raise ValueError("module name must be a safe identifier")
        _validate_refs(self.input_refs)
        _validate_refs(self.output_refs)
        if self.output_valid and not self.output_refs:
            raise ValueError("valid output requires an output reference")
        if self.decision_impacts and self.counterfactual is None:
            raise ValueError("decision impact requires a matched counterfactual run")
        if self.counterfactual is not None and not self.counterfactual.matched:
            raise ValueError("counterfactual control and treatment must match seed/scenario/body")
        if self.fault_injections and not self.decision_impacts:
            raise ValueError("fault evidence cannot substitute for causal decision evidence")
        if self.replay_verified and not self.replay_ref:
            raise ValueError("replay verification requires a replay reference")
        if self.replay_ref is not None:
            _validate_refs((self.replay_ref,))

    @property
    def level(self) -> ModuleEvidenceLevel:
        if not self.invoked:
            return ModuleEvidenceLevel.EXISTS
        if not (self.output_valid and self.output_refs):
            return ModuleEvidenceLevel.INVOKED
        if not (
            self.decision_impacts
            and self.counterfactual is not None
            and self.counterfactual.matched
            and self.counterfactual.decision_changed
        ):
            return ModuleEvidenceLevel.VALID_OUTPUT
        if not self.fault_injections or not all(item.passed for item in self.fault_injections):
            return ModuleEvidenceLevel.DECISION_IMPACT
        if not (self.replay_verified and self.replay_ref):
            return ModuleEvidenceLevel.FAULT_TOLERANT
        return ModuleEvidenceLevel.REPLAY_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module": self.module,
            "level": self.level.value,
            "invoked": self.invoked,
            "input_refs": list(self.input_refs),
            "output_refs": list(self.output_refs),
            "output_valid": self.output_valid,
            "decision_impact": {
                "changed": bool(self.decision_impacts),
                "effects": list(self.decision_impacts),
            },
            "counterfactual": (
                self.counterfactual.to_dict() if self.counterfactual is not None else None
            ),
            "fault_injection": [asdict(item) for item in self.fault_injections],
            "replay_verified": self.replay_verified,
            "replay_ref": self.replay_ref,
        }

    @property
    def proof_hash(self) -> str:
        return _hash_json(self.to_dict())


@dataclass(frozen=True)
class ProofBundle:
    run_id: str
    task_id: str
    body_snapshot_hash: str
    proofs: tuple[ModuleProof, ...]
    evidence_root_ref: str
    candidate_hash: str | None = None
    schema_version: str = "rosclaw.proof_bundle.v1"

    def __post_init__(self) -> None:
        if not self.run_id or not self.task_id:
            raise ValueError("proof bundle run_id and task_id are required")
        if not _sha256_ref(self.body_snapshot_hash):
            raise ValueError("proof bundle requires a sha256 body snapshot hash")
        if not self.proofs:
            raise ValueError("proof bundle cannot be empty")
        if self.candidate_hash is not None and not _sha256_ref(self.candidate_hash):
            raise ValueError("proof bundle candidate hash must be a sha256 identifier")
        modules = [proof.module for proof in self.proofs]
        if len(modules) != len(set(modules)):
            raise ValueError("proof bundle modules must be unique")
        _validate_refs((self.evidence_root_ref,))

    def require_levels(
        self,
        *,
        minimum: ModuleEvidenceLevel,
        modules: tuple[str, ...],
    ) -> None:
        available = {proof.module: proof.level for proof in self.proofs}
        missing = [module for module in modules if module not in available]
        below = [
            f"{module}:{available[module].value}"
            for module in modules
            if module in available and available[module].rank < minimum.rank
        ]
        if missing or below:
            details = []
            if missing:
                details.append("missing=" + ",".join(sorted(missing)))
            if below:
                details.append("below=" + ",".join(sorted(below)))
            raise ValueError(f"proof bundle does not meet {minimum.value}: " + "; ".join(details))

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "body_snapshot_hash": self.body_snapshot_hash,
            "evidence_root_ref": self.evidence_root_ref,
            "candidate_hash": self.candidate_hash,
            "proofs": [
                proof.to_dict() for proof in sorted(self.proofs, key=lambda item: item.module)
            ],
        }
        value["bundle_hash"] = _hash_json(value)
        return value

    @property
    def bundle_hash(self) -> str:
        return str(self.to_dict()["bundle_hash"])


def _validate_refs(refs: tuple[str, ...]) -> None:
    invalid = [ref for ref in refs if not isinstance(ref, str) or not _REF_RE.fullmatch(ref)]
    if invalid:
        raise ValueError("evidence references must be non-empty URI-like values")


def _sha256_ref(value: str) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", value))


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "CounterfactualMetric",
    "CounterfactualRun",
    "FaultInjectionResult",
    "ModuleEvidenceLevel",
    "ModuleProof",
    "ProofBundle",
]
