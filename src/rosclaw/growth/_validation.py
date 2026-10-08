"""Shared validation helpers for task-neutral Growth contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from types import MappingProxyType

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def require_hash(label: str, value: str) -> None:
    if not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def require_identifier(label: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a normalized identifier")


def unique_identifiers(
    values: tuple[str, ...],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(values)
    if not allow_empty and not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    for value in normalized:
        require_identifier(label, value)
    return normalized


def unique_hashes(
    values: tuple[str, ...],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(values)
    if not allow_empty and not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    for value in normalized:
        require_hash(label, value)
    return normalized


def finite_mapping(
    values: Mapping[str, float],
    *,
    label: str,
    non_negative: bool,
) -> Mapping[str, float]:
    normalized = {str(key): float(value) for key, value in values.items()}
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    for key, value in normalized.items():
        require_identifier(f"{label} key", key)
        if not math.isfinite(value) or (non_negative and value < 0.0):
            qualifier = "finite and non-negative" if non_negative else "finite"
            raise ValueError(f"{label} values must be {qualifier}")
    return MappingProxyType(normalized)


__all__: list[str] = []
