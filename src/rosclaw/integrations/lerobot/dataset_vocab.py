"""Vocabulary encoding for ROSClaw categorical dataset fields.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It maps human-readable labels to compact integer codes for Parquet
storage and provides the ``meta/rosclaw/vocab.json`` sidecar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VOCAB_SCHEMA_VERSION = "rosclaw.lerobot.vocab.v1"

# 0 is always reserved for unknown/inactive so that missing data has a stable code.
UNKNOWN_LABEL = "UNKNOWN"
UNKNOWN_CODE = 0

SANDBOX_DECISION_VOCAB: dict[str, int] = {
    UNKNOWN_LABEL: 0,
    "ALLOW": 1,
    "CLAMP": 2,
    "BLOCK": 3,
    "ESTOP": 4,
}

ACTION_SOURCE_VOCAB: dict[str, int] = {
    UNKNOWN_LABEL: 0,
    "POLICY": 1,
    "HUMAN": 2,
    "RULE": 3,
    "HOW_CORRECTION": 4,
    "REPLAY": 5,
}

INTERVENTION_SOURCE_VOCAB: dict[str, int] = {
    UNKNOWN_LABEL: 0,
    "HUMAN_JOYSTICK": 1,
    "HUMAN_KEYFRAME": 2,
    "POLICY_OVERRIDE": 3,
    "RULE_OVERRIDE": 4,
}

FAILURE_CODE_VOCAB: dict[str, int] = {
    UNKNOWN_LABEL: 0,
    "NONE": 1,
    "COLLISION": 2,
    "SELF_COLLISION": 3,
    "JOINT_LIMIT": 4,
    "OVERCURRENT": 5,
    "OVERTEMPERATURE": 6,
    "DROP": 7,
    "SLIP": 8,
    "TIMEOUT": 9,
    "TASK_FAIL": 10,
}


@dataclass
class RosclawVocab:
    """Complete vocabulary sidecar for a ROSClaw-rich dataset."""

    schema_version: str = VOCAB_SCHEMA_VERSION
    vocabularies: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "vocabularies": {k: dict(v) for k, v in self.vocabularies.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RosclawVocab":
        return cls(
            schema_version=data.get("schema_version", VOCAB_SCHEMA_VERSION),
            vocabularies={
                k: {label: int(code) for label, code in v.items()}
                for k, v in data.get("vocabularies", {}).items()
            },
        )


def encode(label: str | None, vocab: dict[str, int]) -> int:
    """Encode a string label using a vocabulary.

    Unknown or missing labels map to ``UNKNOWN_CODE`` rather than raising.
    """
    if label is None:
        return UNKNOWN_CODE
    key = str(label).upper()
    if key in vocab:
        return vocab[key]
    # Accept numeric strings that are already codes.
    try:
        code = int(key)
        if code in vocab.values():
            return code
    except ValueError:
        pass
    return UNKNOWN_CODE


def decode(code: int, vocab: dict[str, int]) -> str:
    """Decode an integer code back to a label.

    Unknown codes map to ``UNKNOWN_LABEL``.
    """
    reverse = {v: k for k, v in vocab.items()}
    return reverse.get(int(code), UNKNOWN_LABEL)


def build_rosclaw_vocab(feature_groups: set[str] | list[str]) -> RosclawVocab:
    """Build the vocab sidecar for the enabled feature groups."""
    vocabs: dict[str, dict[str, int]] = {}
    if "safety" in feature_groups:
        vocabs["rosclaw.sandbox.decision"] = dict(SANDBOX_DECISION_VOCAB)
    if "failure" in feature_groups:
        vocabs["rosclaw.failure.code"] = dict(FAILURE_CODE_VOCAB)
    if "intervention" in feature_groups:
        vocabs["rosclaw.intervention.source"] = dict(INTERVENTION_SOURCE_VOCAB)
    if {"safety", "intervention", "failure", "success", "action", "outcome"} & set(feature_groups):
        vocabs["rosclaw.action.source"] = dict(ACTION_SOURCE_VOCAB)
    return RosclawVocab(vocabularies=vocabs)


def write_vocab(vocab: RosclawVocab, output_dir: Path | str) -> Path:
    """Write ``meta/rosclaw/vocab.json`` under the dataset directory."""
    output_dir = Path(output_dir)
    sidecar_dir = output_dir / "meta" / "rosclaw"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / "vocab.json"
    path.write_text(json.dumps(vocab.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_vocab(output_dir: Path | str) -> RosclawVocab | None:
    """Read ``meta/rosclaw/vocab.json`` if it exists."""
    path = Path(output_dir) / "meta" / "rosclaw" / "vocab.json"
    if not path.exists():
        return None
    try:
        return RosclawVocab.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "VOCAB_SCHEMA_VERSION",
    "UNKNOWN_LABEL",
    "UNKNOWN_CODE",
    "SANDBOX_DECISION_VOCAB",
    "ACTION_SOURCE_VOCAB",
    "INTERVENTION_SOURCE_VOCAB",
    "FAILURE_CODE_VOCAB",
    "RosclawVocab",
    "encode",
    "decode",
    "build_rosclaw_vocab",
    "write_vocab",
    "read_vocab",
]
