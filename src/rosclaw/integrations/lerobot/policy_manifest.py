"""Minimal LeRobot policy manifest parsing.

This module is safe to import from ROSClaw core because it only uses the
standard library.  It reads ``config.json`` or ``config.yaml`` from a policy
directory and extracts the fields the bridge cares about.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_policy_manifest(policy_dir: str | Path) -> dict[str, Any]:
    """Load a minimal manifest from a LeRobot policy directory.

    Returns a dict with at least ``policy_type``, ``config_found``,
    ``input_features``, ``output_features``, and ``raw_config_keys``.
    """
    path = Path(policy_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Policy directory not found: {path}")

    config = _read_config(path)
    if config is None:
        return {
            "policy_type": "unknown",
            "config_found": False,
            "input_features": {},
            "output_features": {},
            "raw_config_keys": [],
        }

    return _normalize_manifest(config)


def _read_config(policy_dir: Path) -> dict[str, Any] | None:
    json_path = policy_dir / "config.json"
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001
            return None

    yaml_path = policy_dir / "config.yaml"
    if yaml_path.exists():
        try:
            import yaml

            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001
            return None

    return None


def _normalize_manifest(config: dict[str, Any]) -> dict[str, Any]:
    policy_type = "unknown"
    for key in ("policy_type", "type", "name"):
        if key in config:
            policy_type = str(config[key])
            break
    if isinstance(config.get("policy"), dict) and "name" in config["policy"]:
        policy_type = str(config["policy"]["name"])

    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    if isinstance(config.get("policy"), dict):
        input_features = config["policy"].get("input_features", input_features)
        output_features = config["policy"].get("output_features", output_features)

    return {
        "policy_type": policy_type,
        "config_found": True,
        "input_features": input_features,
        "output_features": output_features,
        "raw_config_keys": sorted(config.keys()),
    }
