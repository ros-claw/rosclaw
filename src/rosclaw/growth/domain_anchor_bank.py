"""Task-neutral, lossless multi-domain anchors for a frozen residual gate.

Domain labels prevent equal context names in different backends from colliding.
All rows (including duplicates) remain in sealed order. This is an offline
numeric contract, not verification of source evidence or trajectory retention.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.growth.anchor_kernel import AnchorKernelGuard

SCHEMA = "rosclaw.growth.domain_anchor_bank.v1"
_FLAGS = (
    "physical_batch_verified",
    "distributional_retention_guaranteed",
    "runtime_execution_authorized",
    "promotion_authorized",
    "hardware_authorized",
)


def _hash(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _identity(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _source_hash() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _checked_domains(domains: Any) -> tuple[list[dict[str, Any]], np.ndarray[Any, Any]]:
    if type(domains) is not list or not 1 <= len(domains) <= 16:
        raise ValueError("bounded nonempty declared domains required")
    result: list[dict[str, Any]] = []
    arrays: list[np.ndarray[Any, Any]] = []
    domain_ids: set[str] = set()
    dimension: int | None = None
    total = 0
    for domain in domains:
        if type(domain) is not dict or set(domain) != {
            "domain_id",
            "source_evidence_hash",
            "context_ids",
            "context_rows",
            "observations",
        }:
            raise ValueError("complete ordered domain declaration required")
        name, ids, counts = domain["domain_id"], domain["context_ids"], domain["context_rows"]
        if (
            type(name) is not str
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name) is None
            or name in domain_ids
            or not _identity(domain["source_evidence_hash"])
            or type(ids) is not list
            or not 1 <= len(ids) <= 256
            or any(type(v) is not str or not 1 <= len(v) <= 192 for v in ids)
            or len(set(ids)) != len(ids)
            or type(counts) is not list
            or len(counts) != len(ids)
            or any(type(v) is not int or not 1 <= v <= 32768 for v in counts)
        ):
            raise ValueError("unique domain and within-domain context identities required")
        values = np.asarray(domain["observations"])
        if (
            values.dtype.kind not in "fiu"
            or values.ndim != 2
            or values.shape[0] != sum(counts)
            or not 1 <= values.shape[1] <= 512
            or not np.isfinite(values).all()
            or np.max(np.abs(values)) > 1e6
            or (dimension is not None and values.shape[1] != dimension)
        ):
            raise ValueError("complete finite aligned observations required")
        dimension = values.shape[1]
        total += len(values)
        if total > 32768:
            raise ValueError("anchor capacity exceeded; no eviction or row dropping permitted")
        domain_ids.add(name)
        owned = np.array(values, dtype=np.float64, copy=True)
        arrays.append(owned)
        result.append(
            {
                "domain_id": name,
                "source_evidence_hash": domain["source_evidence_hash"],
                "context_ids": ids.copy(),
                "context_rows": counts.copy(),
                "observations": owned.tolist(),
            }
        )
    return result, np.concatenate(arrays)


def build_domain_anchor_bank(
    domains: list[dict[str, Any]], *, parent_hash: str, encoder_hash: str
) -> dict[str, Any]:
    if not _identity(parent_hash) or not _identity(encoder_hash):
        raise ValueError("immutable parent and encoder identities required")
    checked, observations = _checked_domains(domains)
    bank: dict[str, Any] = dict(
        schema=SCHEMA,
        parent_hash=parent_hash,
        encoder_hash=encoder_hash,
        source_hash=_source_hash(),
        domains=checked,
        row_count=len(observations),
        dimension=observations.shape[1],
        context_count=sum(len(v["context_ids"]) for v in checked),
        all_declared_rows_retained=True,
        ordering="DECLARED_DOMAIN_CONTEXT_FRAME_ORDER",
        duplicate_rows="RETAINED_WITH_DOMAIN_PROVENANCE",
        requires_frozen_parent=True,
        requires_frozen_encoder=True,
        local_guarantee_only=True,
        **dict.fromkeys(_FLAGS, False),
    )
    bank["bank_hash"] = _hash(bank)
    return bank


def validate_domain_anchor_bank(bank: dict[str, Any]) -> None:
    if type(bank) is not dict:
        raise ValueError("sealed multi-domain anchor bank required")
    expected = build_domain_anchor_bank(
        bank.get("domains"),
        parent_hash=bank.get("parent_hash"),
        encoder_hash=bank.get("encoder_hash"),
    )
    if (
        bank != expected
        or _hash({k: v for k, v in bank.items() if k != "bank_hash"}) != expected["bank_hash"]
    ):
        raise ValueError("multi-domain anchor identity, completeness or authority drift")


class DomainAnchorGuard(AnchorKernelGuard):
    """One frozen gate over every declared domain; no domain-conditioned bypass."""

    def __init__(self, bank: dict[str, Any], *, bandwidth: float = 1e-4) -> None:
        validate_domain_anchor_bank(bank)
        self._bank = copy.deepcopy(bank)
        _, values = _checked_domains(bank["domains"])
        super().__init__(values, bandwidth=bandwidth)
        self.bank_hash = bank["bank_hash"]

    def bank(self) -> dict[str, Any]:
        return copy.deepcopy(self._bank)
