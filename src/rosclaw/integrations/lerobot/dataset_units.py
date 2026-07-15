"""Unit and feature-name sidecars for ROSClaw-rich LeRobotDatasets.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It writes ``meta/rosclaw/units.json`` and
``meta/rosclaw/feature_names.json`` so downstream tools can interpret telemetry
arrays without hard-coding ROSClaw conventions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default SI-ish units for the physical telemetry fields introduced in Gate B.
# Callers can override these by passing a custom unit map.
DEFAULT_UNIT_MAP: dict[str, dict[str, Any]] = {
    "observation.motor_current": {
        "unit": "A",
        "description": "Per-motor current draw",
    },
    "observation.joint_temperature": {
        "unit": "degC",
        "description": "Per-joint temperature",
    },
    "observation.force_torque": {
        "unit": "N_Nm",
        "description": "Force/torque wrench [Fx, Fy, Fz, Tx, Ty, Tz]",
    },
    "observation.contact": {
        "unit": "bool",
        "description": "Binary contact flag per contact sensor",
    },
    "observation.joint_velocity": {
        "unit": "rad_s",
        "description": "Per-joint velocity",
    },
    "observation.joint_effort": {
        "unit": "Nm",
        "description": "Per-joint effort/torque",
    },
    "rosclaw.sandbox.risk_score": {
        "unit": "scalar",
        "description": "Normalized risk score (0-1, NaN if unknown)",
    },
}


# Human-readable short names and axis semantics for ROSClaw features.
DEFAULT_FEATURE_NAMES: dict[str, dict[str, Any]] = {
    "observation.motor_current": {
        "name": "motor_current",
        "axis_names": ["motor"],
    },
    "observation.joint_temperature": {
        "name": "joint_temperature",
        "axis_names": ["joint"],
    },
    "observation.force_torque": {
        "name": "force_torque",
        "axis_names": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
    },
    "observation.contact": {
        "name": "contact",
        "axis_names": ["contact"],
    },
    "observation.joint_velocity": {
        "name": "joint_velocity",
        "axis_names": ["joint"],
    },
    "observation.joint_effort": {
        "name": "joint_effort",
        "axis_names": ["joint"],
    },
    "rosclaw.sandbox.decision": {"name": "sandbox_decision"},
    "rosclaw.sandbox.modified": {"name": "sandbox_modified"},
    "rosclaw.sandbox.risk_score": {"name": "risk_score"},
    "rosclaw.failure.active": {"name": "failure_active"},
    "rosclaw.failure.code": {"name": "failure_code"},
    "rosclaw.intervention.active": {"name": "intervention_active"},
    "rosclaw.intervention.source": {"name": "intervention_source"},
    "rosclaw.action.source": {"name": "action_source"},
    "rosclaw.action.was_clamped": {"name": "action_was_clamped"},
    "rosclaw.done": {"name": "done"},
    "rosclaw.success": {"name": "success"},
}


def build_unit_map(
    feature_keys: list[str],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a unit map containing only the exported feature keys."""
    overrides = overrides or {}
    result: dict[str, dict[str, Any]] = {}
    for key in feature_keys:
        if key not in DEFAULT_UNIT_MAP:
            continue
        entry = dict(DEFAULT_UNIT_MAP[key])
        if key in overrides:
            entry.update(overrides[key])
        result[key] = entry
    return result


def build_feature_names_map(
    feature_keys: list[str],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a feature-names map containing only the exported feature keys."""
    overrides = overrides or {}
    result: dict[str, dict[str, Any]] = {}
    for key in feature_keys:
        entry = dict(DEFAULT_FEATURE_NAMES.get(key, {"name": key.split(".")[-1]}))
        if key in overrides:
            entry.update(overrides[key])
        result[key] = entry
    return result


def write_units_json(
    feature_keys: list[str],
    output_root: Path,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Write ``meta/rosclaw/units.json``."""
    output_root = Path(output_root)
    sidecar_dir = output_root / "meta" / "rosclaw"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / "units.json"
    data = {
        "schema_version": "rosclaw.lerobot.units.v1",
        "units": build_unit_map(feature_keys, overrides),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_feature_names_json(
    feature_keys: list[str],
    output_root: Path,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Write ``meta/rosclaw/feature_names.json``."""
    output_root = Path(output_root)
    sidecar_dir = output_root / "meta" / "rosclaw"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / "feature_names.json"
    data = {
        "schema_version": "rosclaw.lerobot.feature_names.v1",
        "features": build_feature_names_map(feature_keys, overrides),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "DEFAULT_UNIT_MAP",
    "build_feature_names_map",
    "build_unit_map",
    "write_feature_names_json",
    "write_units_json",
]
