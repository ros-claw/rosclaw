"""Load LeRobot integration profiles from bundled YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from rosclaw.integrations.lerobot.schemas import ProfileSpec

_PROFILES_PATH = Path(__file__).with_name("profiles.yaml")


def load_profile(name: str) -> ProfileSpec:
    """Load a profile by name."""
    if not _PROFILES_PATH.exists():
        raise FileNotFoundError(f"LeRobot profiles file not found: {_PROFILES_PATH}")
    data = yaml.safe_load(_PROFILES_PATH.read_text(encoding="utf-8"))
    if name not in data:
        raise FileNotFoundError(f"Profile '{name}' not found in {_PROFILES_PATH}")
    spec = data[name]
    return ProfileSpec(
        name=name,
        pip=spec.get("pip", []),
        checks=spec.get("checks", []),
        enabled_capabilities=spec.get("enabled_capabilities", []),
        requires_python=spec.get("requires_python", ">=3.12"),
        capabilities=spec.get("capabilities", {}),
    )


def list_profile_names() -> list[str]:
    """Return all profile names."""
    if not _PROFILES_PATH.exists():
        return []
    data = yaml.safe_load(_PROFILES_PATH.read_text(encoding="utf-8"))
    return list(data.keys())
