"""Independent, fail-closed replay of GoalForge module-proof bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoalForgeProofReplay:
    bundle_path: Path
    requested_modules: tuple[str, ...]
    verified_modules: tuple[str, ...]
    errors: tuple[str, ...]
    bundle_hash: str | None
    schema_version: str = "rosclaw.g1_goalforge.proof_replay.v2"

    @property
    def passed(self) -> bool:
        return not self.errors and set(self.requested_modules) <= set(self.verified_modules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_path": str(self.bundle_path),
            "requested_modules": list(self.requested_modules),
            "verified_modules": list(self.verified_modules),
            "bundle_hash": self.bundle_hash,
            "errors": list(self.errors),
            "passed": self.passed,
        }


def replay_goalforge_proof_bundle(
    source: Path,
    *,
    requested_modules: tuple[str, ...],
) -> GoalForgeProofReplay:
    """Recompute the bundle hash and re-derive E5 from primitive evidence fields."""

    path = _resolve_bundle(source.expanduser().resolve())
    errors: list[str] = []
    if path is None:
        missing = source.expanduser().resolve()
        return GoalForgeProofReplay(
            bundle_path=missing,
            requested_modules=requested_modules,
            verified_modules=(),
            errors=("proof_bundle_not_found",),
            bundle_hash=None,
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return GoalForgeProofReplay(
            bundle_path=path,
            requested_modules=requested_modules,
            verified_modules=(),
            errors=(f"invalid_bundle_json={type(exc).__name__}",),
            bundle_hash=None,
        )
    if not isinstance(value, dict):
        errors.append("bundle_must_be_object")
        value = {}
    claimed_hash = value.get("bundle_hash")
    unhashed = dict(value)
    unhashed.pop("bundle_hash", None)
    computed_hash = _hash_json(unhashed)
    if claimed_hash != computed_hash:
        errors.append("bundle_hash_mismatch")
    if value.get("schema_version") != "rosclaw.proof_bundle.v1":
        errors.append("unsupported_bundle_schema")
    body_hash = value.get("body_snapshot_hash")
    if not (
        isinstance(body_hash, str) and body_hash.startswith("sha256:") and len(body_hash) == 71
    ):
        errors.append("invalid_body_snapshot_hash")
    proofs = value.get("proofs")
    if not isinstance(proofs, list):
        errors.append("proofs_must_be_list")
        proofs = []
    verified: list[str] = []
    seen: set[str] = set()
    for raw in proofs:
        if not isinstance(raw, dict):
            errors.append("proof_must_be_object")
            continue
        module = raw.get("module")
        if not isinstance(module, str) or not module:
            errors.append("proof_module_invalid")
            continue
        if module in seen:
            errors.append(f"duplicate_module={module}")
            continue
        seen.add(module)
        if module not in requested_modules:
            continue
        module_errors = _rederive_e5(raw)
        if module_errors:
            errors.extend(f"{module}:{item}" for item in module_errors)
        else:
            verified.append(module)
    for module in requested_modules:
        if module not in seen:
            errors.append(f"missing_module={module}")
    return GoalForgeProofReplay(
        bundle_path=path,
        requested_modules=requested_modules,
        verified_modules=tuple(sorted(verified)),
        errors=tuple(errors),
        bundle_hash=computed_hash,
    )


def _resolve_bundle(source: Path) -> Path | None:
    if source.is_file():
        return source
    if not source.is_dir():
        return None
    for name in ("proof-bundle-final.json", "proof-bundle.json"):
        direct = source / name
        if direct.is_file():
            return direct
    candidates = sorted(source.rglob("proof-bundle-final.json"))
    if not candidates:
        candidates = sorted(source.rglob("proof-bundle.json"))
    return candidates[0] if len(candidates) == 1 else None


def _rederive_e5(proof: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if proof.get("schema_version") != "rosclaw.module_proof.v1":
        errors.append("unsupported_schema")
    if not proof.get("invoked"):
        errors.append("not_invoked")
    if not proof.get("output_valid") or not proof.get("output_refs"):
        errors.append("output_not_valid")
    impact = proof.get("decision_impact")
    if not (
        isinstance(impact, dict)
        and impact.get("changed")
        and isinstance(impact.get("effects"), list)
        and impact["effects"]
    ):
        errors.append("no_decision_impact")
    counterfactual = proof.get("counterfactual")
    if not (
        isinstance(counterfactual, dict)
        and counterfactual.get("same_seed") is True
        and counterfactual.get("same_scenario") is True
        and counterfactual.get("same_body_hash") is True
        and counterfactual.get("decision_changed") is True
        and isinstance(counterfactual.get("metrics"), list)
        and counterfactual["metrics"]
    ):
        errors.append("counterfactual_not_matched")
    faults = proof.get("fault_injection")
    if not (
        isinstance(faults, list)
        and faults
        and all(isinstance(item, dict) and item.get("passed") is True for item in faults)
    ):
        errors.append("fault_injection_not_passed")
    if not proof.get("replay_verified") or not proof.get("replay_ref"):
        errors.append("replay_not_verified")
    if proof.get("level") != "E5":
        errors.append("claimed_level_not_e5")
    return tuple(errors)


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = ["GoalForgeProofReplay", "replay_goalforge_proof_bundle"]
