"""Resolve RH56 reference artifacts in source and installed distributions."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_ROOTS = {
    "configs": _REPOSITORY_ROOT / "configs",
    "policies": (_REPOSITORY_ROOT / "worker_plugins" / "lerobot_policy_rosclaw_rh56" / "policies"),
}


def _resolve(kind: str, relative_path: str) -> Path:
    source_root = _SOURCE_ROOTS.get(kind, _REPOSITORY_ROOT / kind)
    candidates = (
        _PACKAGE_ROOT / "rh56_data" / kind / relative_path,
        source_root / relative_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"RH56 {kind} resource {relative_path!r} not found; searched {searched}"
    )


def rh56_config_path(name: str) -> Path:
    """Return a bundled RH56 configuration path."""
    return _resolve("configs", name)


def rh56_reference_policy_path() -> Path:
    """Return the bundled deterministic RH56 LeRobot policy directory."""
    return _resolve("policies", "rh56_reference_policy_v1")


__all__ = ["rh56_config_path", "rh56_reference_policy_path"]
