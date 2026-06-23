"""Packaging logic for ROSClaw skill assets."""

from __future__ import annotations

import re
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.hub.publisher import scan_secrets
from rosclaw.skill.hash import compute_skill_hashes
from rosclaw.skill.models import SkillPackage

# ---------------------------------------------------------------------------
# Secret and absolute-path scanning
# ---------------------------------------------------------------------------

_SECRET_KEYWORDS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "QWEN_API_KEY",
    "ROSCLAW_API_KEY",
    "ROSCLAW_ADMIN_API_KEY",
    "x-api-key",
    "sk-",
    "Bearer",
]

_ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|\s)(/home/[^\s]+|/mnt/[^\s]+|/Users/[^\s]+|C:\\\\Users\\\\[^\s]+|\\\\[^\s]+)(?:\s|$)"
)


def scan_forbidden_content(root: Path) -> tuple[list[str], list[str]]:
    """Return (secret_findings, absolute_path_findings)."""
    secret_findings: list[str] = []
    path_findings: list[str] = []

    # Reuse the publisher secret scanner as a first pass.
    secret_findings.extend(scan_secrets(root))

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        # Skip binary-ish files and large artifacts.
        if rel.startswith("evidence/videos/") or rel.endswith(".pt") or rel.endswith(".mcap"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for keyword in _SECRET_KEYWORDS:
            if keyword in text:
                # Avoid flagging template placeholders and documentation references.
                if keyword == "x-api-key" and rel.endswith(".md"):
                    continue
                if keyword in text and not _is_innocuous_context(text, keyword):
                    secret_findings.append(f"Possible secret {keyword!r} in {rel}")
        for match in _ABSOLUTE_PATH_RE.finditer(text):
            path_findings.append(f"Absolute path {match.group(1)!r} in {rel}")
    return secret_findings, path_findings


def _is_innocuous_context(text: str, keyword: str) -> bool:
    """Heuristic to reduce false positives in docs/templates."""
    for line in text.splitlines():
        if keyword in line:
            lowered = line.lower()
            if "env" in lowered or "export" in lowered or "optional" in lowered:
                # Still require manual review but treat as warning, not error.
                return False
            if "example" in lowered or "your_" in lowered or "placeholder" in lowered:
                return True
    return False


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def prepare_manifest(pkg: SkillPackage) -> dict[str, Any]:
    if pkg.skill is None:
        raise RuntimeError("skill.yaml not loaded")
    eurdf = pkg.eurdf_compat or None
    compatible_robots = []
    robot_types: list[str] = []
    if eurdf:
        compatible_robots = [r.robot for r in eurdf.compatible_robots]
        # Infer robot types from names (heuristic).
        for r in eurdf.compatible_robots:
            rt = _infer_robot_type(r.robot)
            if rt and rt not in robot_types:
                robot_types.append(rt)
    hashes = compute_skill_hashes(pkg.root, include_evidence=False)
    return {
        "schema_version": "rosclaw.skill_manifest.v1",
        "name": pkg.skill_id,
        "version": pkg.version,
        "stage": pkg.skill.metadata.stage,
        "package_hash": hashes["package_hash"],
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "entrypoint": pkg.skill.execution.entrypoint.file,
        "required_files": [
            "skill.yaml",
            "README.md",
            "behavior_tree.xml",
            "providers.yaml",
            "e-urdf-compat.yaml",
            "safety.yaml",
            "dojo.yaml",
            "darwin_eval.yaml",
            "lineage.yaml",
        ],
        "hub": {
            "category": pkg.skill.metadata.category,
            "robot_types": robot_types,
            "compatible_robots": compatible_robots,
            "tags": pkg.skill.metadata.tags,
        },
    }


def _infer_robot_type(robot_name: str) -> str | None:
    name = robot_name.lower()
    if "g1" in name or "humanoid" in name:
        return "humanoid"
    if "ur5" in name or "franka" in name or "panda" in name:
        return "manipulator"
    if "go2" in name or "a1" in name or "spot" in name or "quadruped" in name:
        return "legged"
    if "turtle" in name or "mobile" in name:
        return "mobile"
    if "drone" in name or "uav" in name:
        return "drone"
    return None


def package_skill(
    pkg: SkillPackage,
    output_dir: Path,
    include_evidence: str = "summary",
    format: str = "tar.gz",
) -> Path:
    if format != "tar.gz":
        raise ValueError(f"Unsupported package format: {format}")

    # Refresh hashes.
    hashes = compute_skill_hashes(
        pkg.root,
        include_evidence=(include_evidence in ("full", "summary")),
    )
    pkg.write_hashes_json(hashes)

    manifest = prepare_manifest(pkg)
    pkg.write_manifest_json(manifest)

    # Update lock hashes.
    lock_path = pkg.root / ".rosclaw" / "lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {"schema_version": "rosclaw.lock.v1"}
    lock["hashes"] = hashes["files"]
    pkg.write_lock_yaml(lock)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{pkg.name}-{pkg.version}.tar.gz"
    archive_path = output_dir / archive_name

    included_prefixes = _package_included_prefixes(include_evidence)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(pkg.root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(pkg.root).as_posix()
            if not _should_include(rel, included_prefixes):
                continue
            tar.add(path, arcname=rel)
    return archive_path


def _package_included_prefixes(include_evidence: str) -> list[str]:
    prefixes = [
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
        "tests/",
        "lineage.yaml",
        "CHANGELOG.md",
        ".rosclaw/",
    ]
    if include_evidence in ("full", "summary"):
        prefixes.append("evidence/reports/")
    if include_evidence == "full":
        prefixes.append("evidence/")
    return prefixes


def _should_include(rel: str, prefixes: list[str]) -> bool:
    return any(rel.startswith(p) for p in prefixes)


def verify_package(archive_path: Path) -> dict[str, Any]:
    result = {"ok": True, "errors": [], "warnings": [], "files": []}
    if not archive_path.exists():
        result["ok"] = False
        result["errors"].append(f"Archive not found: {archive_path}")
        return result

    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
        result["files"] = names
        required = ["skill.yaml", "behavior_tree.xml", "providers.yaml", "safety.yaml"]
        for r in required:
            if r not in names:
                result["ok"] = False
                result["errors"].append(f"Missing required file in archive: {r}")
        # Check manifest presence.
        if ".rosclaw/manifest.json" not in names:
            result["warnings"].append("Missing .rosclaw/manifest.json")
        if ".rosclaw/hashes.json" not in names:
            result["warnings"].append("Missing .rosclaw/hashes.json")

        # Extract to temp for secret scan.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tar.extractall(path=tmp)
            secrets, paths = scan_forbidden_content(Path(tmp))
            if secrets:
                result["ok"] = False
                result["errors"].extend(f"SECRET: {s}" for s in secrets)
            if paths:
                result["warnings"].extend(f"PATH: {p}" for p in paths)
    return result
