"""Provenance and validity vocabularies for Gate B.1 synchronization.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It defines the integer codes written alongside resampled features
so downstream tooling can audit how each frame value was produced.
"""

from __future__ import annotations

from typing import Any


# Provenance integer codes.  Keep this stable; it is referenced from both
# sidecar metadata and optional per-frame features.
PROVENANCE_UNKNOWN = 0
PROVENANCE_EXACT = 1
PROVENANCE_NEAREST = 2
PROVENANCE_INTERPOLATED = 3
PROVENANCE_HELD = 4
PROVENANCE_AGGREGATED = 5
PROVENANCE_FILLED_NAN = 6

_PROVENANCE_LABELS: dict[int, str] = {
    PROVENANCE_UNKNOWN: "UNKNOWN",
    PROVENANCE_EXACT: "EXACT",
    PROVENANCE_NEAREST: "NEAREST",
    PROVENANCE_INTERPOLATED: "INTERPOLATED",
    PROVENANCE_HELD: "HELD",
    PROVENANCE_AGGREGATED: "AGGREGATED",
    PROVENANCE_FILLED_NAN: "FILLED_NAN",
}

# Validity integer codes for physical features.
VALIDITY_UNKNOWN = -1
VALIDITY_INVALID = 0
VALIDITY_VALID = 1

_VALIDITY_LABELS: dict[int, str] = {
    VALIDITY_UNKNOWN: "UNKNOWN",
    VALIDITY_INVALID: "INVALID",
    VALIDITY_VALID: "VALID",
}


def provenance_label(code: int) -> str:
    return _PROVENANCE_LABELS.get(code, "UNKNOWN")


def validity_label(code: int) -> str:
    return _VALIDITY_LABELS.get(code, "UNKNOWN")


def build_provenance_vocab() -> dict[str, dict[str, int]]:
    return {
        "rosclaw.provenance": {
            label: code for code, label in _PROVENANCE_LABELS.items()
        }
    }


def build_validity_vocab() -> dict[str, dict[str, int]]:
    return {
        "rosclaw.validity": {
            label: code for code, label in _VALIDITY_LABELS.items()
        }
    }


__all__ = [
    "PROVENANCE_AGGREGATED",
    "PROVENANCE_EXACT",
    "PROVENANCE_FILLED_NAN",
    "PROVENANCE_HELD",
    "PROVENANCE_INTERPOLATED",
    "PROVENANCE_NEAREST",
    "PROVENANCE_UNKNOWN",
    "VALIDITY_INVALID",
    "VALIDITY_UNKNOWN",
    "VALIDITY_VALID",
    "build_provenance_vocab",
    "build_validity_vocab",
    "provenance_label",
    "validity_label",
]
