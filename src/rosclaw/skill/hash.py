"""Hash helpers for ROSClaw skill packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_hex(path.read_bytes())


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
