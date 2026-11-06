"""Resolve only outcomes recorded for the same physical boundary and action.

This module is a matching guard, not an evidence authenticator or safety gate.
Callers must independently verify each referenced physical receipt and hash.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _require_hash(value: str) -> None:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("content-addressed evidence identity required")


@dataclass(frozen=True, slots=True)
class CounterfactualKey:
    """Exact intervention boundary, physical condition and requested action."""

    boundary_hash: str
    condition_hash: str
    action_hash: str

    def __post_init__(self) -> None:
        for value in (self.boundary_hash, self.condition_hash, self.action_hash):
            _require_hash(value)


@dataclass(frozen=True, slots=True)
class ObservedBranch:
    """A caller-authenticated branch; this class does not verify its evidence."""

    key: CounterfactualKey
    outcome_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        if type(self.key) is not CounterfactualKey:
            raise ValueError("exact counterfactual key required")
        _require_hash(self.outcome_hash)
        _require_hash(self.receipt_hash)


class CoverageStatus(StrEnum):
    OBSERVED = "observed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Resolution:
    status: CoverageStatus
    branch: ObservedBranch | None


def resolve_selected_branch(
    requested: CounterfactualKey, branches: Iterable[ObservedBranch]
) -> Resolution:
    """Never replace an unobserved selection with another action's outcome.

    Duplicate exact keys are rejected even if values agree, because callers
    must explicitly reconcile independent repeats or conflicting receipts.
    The caller is responsible for verifying the branch receipt and canonical
    hashes, physical replay and comparability before interpreting an outcome.
    """
    if type(requested) is not CounterfactualKey:
        raise ValueError("exact requested key required")
    found: ObservedBranch | None = None
    for branch in branches:
        if type(branch) is not ObservedBranch:
            raise ValueError("typed observed branch required")
        if branch.key == requested:
            if found is not None:
                raise ValueError("duplicate outcome for selected physical action")
            found = branch
    return (
        Resolution(CoverageStatus.OBSERVED, found)
        if found
        else Resolution(CoverageStatus.UNKNOWN, None)
    )
