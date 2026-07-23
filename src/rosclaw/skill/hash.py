"""Hash helpers for ROSClaw skill packages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_CANDIDATE_FILE_BYTES = 4 * 1024 * 1024


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def validate_candidate_id(candidate_id: str) -> str:
    """Validate an identifier before using it in package-relative paths."""

    if not isinstance(candidate_id, str) or not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise ValueError("Candidate id must use 1-64 lowercase letters, digits, '_' or '-'")
    return candidate_id


def candidate_artifact_paths(root: Path, candidate_id: str) -> dict[str, Path]:
    candidate_id = validate_candidate_id(candidate_id)
    resolved_root = root.expanduser().resolve()
    return {
        "parameters": resolved_root / "policies" / "params" / f"{candidate_id}.yaml",
        "behavior_tree": resolved_root / f"behavior_tree.{candidate_id}.xml",
    }


def compute_candidate_evidence_hash(root: Path, candidate_id: str) -> str:
    """Bind simulation evidence to the exact candidate inputs it evaluates."""

    hashes: dict[str, str] = {}
    for name, path in candidate_artifact_paths(root, candidate_id).items():
        if not path.is_file() or path.stat().st_size > _MAX_CANDIDATE_FILE_BYTES:
            raise ValueError(f"Candidate artifact is missing or too large: {name}")
        hashes[name] = f"sha256:{sha256_file(path)}"
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{sha256_hex(payload)}"


def compute_skill_hashes(
    root: Path,
    include_evidence: bool = False,
    exclude_globs: list[str] | None = None,
) -> dict[str, Any]:
    """Compute sha256 hashes for tracked skill package files.

    Args:
        root: Skill package root directory.
        include_evidence: Whether to include evidence/ files (default False for summaries only).
        exclude_globs: Relative path globs to skip.
    """
    exclude_globs = exclude_globs or []
    tracked_prefixes = [
        "skill.yaml",
        "README.md",
        "SKILL.md",
        "behavior_tree.xml",
        "prompts/",
        "policies/",
        "providers.yaml",
        "e-urdf-compat.yaml",
        "safety.yaml",
        "dojo.yaml",
        "darwin_eval.yaml",
        "lineage.yaml",
        "CHANGELOG.md",
    ]
    if include_evidence:
        tracked_prefixes.append("evidence/reports/")

    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if any(rel.startswith(e) for e in exclude_globs):
            continue
        if not any(rel.startswith(p) for p in tracked_prefixes):
            continue
        files[rel] = f"sha256:{sha256_file(path)}"

    # Deterministic package hash over sorted file hashes.
    digest_input = json.dumps(files, sort_keys=True).encode("utf-8")
    package_hash = f"sha256:{sha256_hex(digest_input)}"
    return {
        "schema_version": "rosclaw.hashes.v1",
        "files": files,
        "package_hash": package_hash,
    }
