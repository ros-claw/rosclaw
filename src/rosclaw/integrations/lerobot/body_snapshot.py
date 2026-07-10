"""Body snapshot sidecar for ROSClaw-rich datasets.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It copies selected body configuration files into the dataset sidecar
with optional sanitization.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


# Patterns used to strip sensitive values in sanitized mode.
_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("serial_number", re.compile(r"serial[_]?number\s*[:=]\s*[^\s]+", re.IGNORECASE), "serial_number"),
    ("serial", re.compile(r"serial\s*[:=]\s*[^\s]+", re.IGNORECASE), "serial"),
    ("ip_address", re.compile(r"ip[_]?addr(ess)?\s*[:=]\s*[^\s]+", re.IGNORECASE), "ip_address"),
    ("token", re.compile(r"token\s*[:=]\s*[^\s]+", re.IGNORECASE), "token"),
    ("api_key", re.compile(r"api[_]?key\s*[:=]\s*[^\s]+", re.IGNORECASE), "api_key"),
    ("password", re.compile(r"password\s*[:=]\s*[^\s]+", re.IGNORECASE), "password"),
    ("latitude", re.compile(r"latitude\s*[:=]\s*[^\s]+", re.IGNORECASE), "latitude"),
    ("longitude", re.compile(r"longitude\s*[:=]\s*[^\s]+", re.IGNORECASE), "longitude"),
    ("lat", re.compile(r"\blat(?:itude)?\b\s*[:=]\s*[^\s]+", re.IGNORECASE), "latitude"),
    ("lon", re.compile(r"\blon(?:gitude)?\b\s*[:=]\s*[^\s]+", re.IGNORECASE), "longitude"),
    ("location", re.compile(r"location\s*[:=]\s*[^\s]+", re.IGNORECASE), "location"),
]

# Heuristic path sanitization: replace home directory prefixes.
_HOME_RE = re.compile(r"/home/[^/]+")


class SensitiveBodyDataError(ValueError):
    """Raised when a full body snapshot is requested without explicit acknowledgement."""


def _sanitize_text(text: str) -> tuple[str, list[str]]:
    redacted_fields: list[str] = []
    for label, pattern, canonical in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            redacted_fields.append(canonical)
        text = pattern.sub(lambda m: m.group(0).split(":", 1)[0] + ": <redacted>", text)
        text = pattern.sub(lambda m: m.group(0).split("=", 1)[0] + "=<redacted>", text)
    text = _HOME_RE.sub("/home/<user>", text)
    return text, sorted(set(redacted_fields))


def _copy_or_sanitize(src: Path, dest: Path, mode: str) -> tuple[str, str, list[str]]:
    src_text = src.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(src_text.encode("utf-8")).hexdigest()
    if mode == "full":
        dest.write_text(src_text, encoding="utf-8")
        return source_hash, source_hash, []
    sanitized_text, redacted_fields = _sanitize_text(src_text)
    dest.write_text(sanitized_text, encoding="utf-8")
    sanitized_hash = hashlib.sha256(sanitized_text.encode("utf-8")).hexdigest()
    return source_hash, sanitized_hash, redacted_fields


def include_body_snapshot(
    output_dir: Path | str,
    body_yaml_path: Path | str | None,
    *,
    mode: str = "sanitized",
    acknowledge_sensitive: bool = False,
) -> dict[str, Any]:
    """Copy body files into ``meta/rosclaw/body_snapshots/``.

    Returns a manifest dict describing what was written.  If ``body_yaml_path``
    is None or does not exist, an empty manifest is returned and no files are
    written.

    Args:
        output_dir: Root of the exported LeRobotDataset.
        body_yaml_path: Path to the robot's ``body.yaml``.
        mode: ``none`` skips, ``sanitized`` redacts sensitive values,
            ``full`` copies as-is.
        acknowledge_sensitive: Required to be ``True`` when ``mode`` is
            ``full`` to confirm that the caller accepts the risk of exporting
            sensitive body data.
    """
    output_dir = Path(output_dir)
    manifest: dict[str, Any] = {
        "mode": mode,
        "files": {},
        "body_yaml_path": None,
        "acknowledged_sensitive": acknowledge_sensitive,
    }
    if mode == "none":
        return manifest
    if not body_yaml_path:
        return manifest

    body_yaml = Path(body_yaml_path)
    if not body_yaml.exists():
        return manifest

    if mode == "full" and not acknowledge_sensitive:
        raise SensitiveBodyDataError(
            "Full body snapshot exports potentially sensitive body data. "
            "Re-run with --acknowledge-sensitive-body-data to confirm."
        )

    body_dir = body_yaml.parent
    snapshot_dir = output_dir / "meta" / "rosclaw" / "body_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    manifest["body_yaml_path"] = str(body_yaml.resolve())

    candidate_files = ["body.yaml", "EMBODIMENT.md", "calibration.yaml"]
    for name in candidate_files:
        candidate = body_dir / name
        if not candidate.exists():
            continue
        dest = snapshot_dir / name
        try:
            source_hash, sanitized_hash, redacted_fields = _copy_or_sanitize(candidate, dest, mode)
            manifest["files"][name] = {
                "path": str(dest.relative_to(output_dir)),
                "source_sha256": source_hash,
                "sanitized_sha256": sanitized_hash,
                "redaction_count": len(redacted_fields),
                "redacted_fields": redacted_fields,
            }
        except Exception as exc:  # noqa: BLE001
            manifest["files"][name] = {
                "path": None,
                "error": str(exc),
                "source_sha256": None,
                "sanitized_sha256": None,
                "redaction_count": 0,
                "redacted_fields": [],
            }

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


__all__ = [
    "SensitiveBodyDataError",
    "include_body_snapshot",
]
