"""Validators for body update-state and schema."""

from __future__ import annotations

from typing import Any

FORBIDDEN_UPDATE_PATHS = {
    "model_ref.profile_id",
    "model_ref.profile_version",
    "model_ref.profile_checksum",
    "model_ref.eurdf_uri",
}


def validate_update_path(path: str) -> tuple[bool, str]:
    """Return (ok, reason)."""
    if not path:
        return False, "empty path"
    if any(path.startswith(forbidden) or path == forbidden for forbidden in FORBIDDEN_UPDATE_PATHS):
        return False, f"cannot modify protected field: {path}"
    if path.startswith("joints.") and "." in path[len("joints."):].split(".", 1)[0]:
        return False, "cannot modify e-URDF structural joint definitions"
    return True, ""


def parse_set_expression(expr: str) -> tuple[str, Any]:
    """Parse --set key=value into (key, value)."""
    if "=" not in expr:
        raise ValueError(f"Invalid --set expression (expected key=value): {expr}")
    key, value = expr.split("=", 1)
    key = key.strip()
    value = _coerce_value(value.strip())
    return key, value


def _coerce_value(value: str) -> Any:
    """Best-effort type coercion for CLI strings."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null" or lowered == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_nested_update(data: dict[str, Any], path: str, value: Any) -> None:
    """Set data[path] = value, creating intermediate dicts."""
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value
