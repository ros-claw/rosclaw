"""Post-init validation and secrets scanning for agent onboarding files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.agent.merge import MergeConflictError, json_merge_with_conflict_detection
from rosclaw.agent.templates import MANAGED_BEGIN, MANAGED_END


@dataclass
class ValidationResult:
    """Result of validating an initialized project."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    secrets: list[tuple[Path, int, str]] = field(default_factory=list)


# Simple heuristic patterns for secrets. These are intentionally conservative:
# they flag common key formats but may need tuning for specific projects.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "aws-access-key",
        re.compile(r"AKIA[0-9A-Z]{16}"),
    ),
    (
        "generic-api-key",
        re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?[a-z0-9_\-]{16,}['\"]?"),
    ),
    (
        "private-key-block",
        re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "github-token",
        re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}"),
    ),
    (
        "password-assignment",
        re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    ),
]


def scan_for_secrets(root: Path, paths: list[Path]) -> list[tuple[Path, int, str]]:
    """Scan generated files for likely leaked secrets.

    Returns a list of (path, line_number, pattern_name) tuples.
    """
    findings: list[tuple[Path, int, str]] = []
    for file_path in paths:
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            continue
        lines = text.splitlines()
        rel = file_path.relative_to(root) if file_path.is_relative_to(root) else file_path
        for line_no, line in enumerate(lines, start=1):
            for name, pattern in _SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append((rel, line_no, name))
    return findings


def _has_managed_block(content: str) -> bool:
    return MANAGED_BEGIN in content and MANAGED_END in content


def validate_mcp_json(path: Path) -> list[str]:
    """Validate .mcp.json structure."""
    errors: list[str] = []
    if not path.exists():
        errors.append(f"Missing {path}")
        return errors
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in {path}: {e}")
        return errors

    if not isinstance(data, dict):
        errors.append(f"{path} must be a JSON object")
        return errors
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        errors.append(f"{path} has no mcpServers configured")
    elif "rosclaw" not in servers:
        errors.append(f"{path} is missing the rosclaw server")
    return errors


def validate_markdown(path: Path, label: str) -> list[str]:
    """Validate a managed markdown file."""
    errors: list[str] = []
    if not path.exists():
        errors.append(f"Missing {path}")
        return errors
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        errors.append(f"Could not read {path}: {e}")
        return errors
    if not _has_managed_block(content):
        errors.append(f"{label} ({path}) is missing the managed block markers")
    return errors


def validate_context_snapshot(path: Path) -> list[str]:
    """Validate the context.snapshot.json file."""
    errors: list[str] = []
    if not path.exists():
        errors.append(f"Missing {path}")
        return errors
    data = _load_json(path)
    if not isinstance(data, dict):
        errors.append(f"{path} must be a JSON object")
        return errors
    if data.get("schema_version") != "rosclaw.agent.context.v1":
        errors.append(f"{path} has unexpected schema_version")
    tools = data.get("tools", {}).get("available", [])
    expected_tools = [
        "get_robot_state",
        "list_skills",
        "query_memory",
        "validate_trajectory",
        "sandbox_run",
        "practice_query",
        "emergency_stop",
    ]
    missing = [t for t in expected_tools if t not in tools]
    if missing:
        errors.append(f"{path} is missing tools: {missing}")
    return errors


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def validate_project(
    project_root: Path,
    generated_paths: dict[str, Path],
    skip_secrets: bool = False,
) -> ValidationResult:
    """Run all post-init validation checks and return a structured result."""
    errors: list[str] = []
    warnings: list[str] = []

    required = [".mcp.json", "CLAUDE.md", "ROSCLAW.md"]
    for key in required:
        path = generated_paths.get(key)
        if path is None or not path.exists():
            errors.append(f"Required file {key} was not generated")

    if ".mcp.json" in generated_paths:
        errors.extend(validate_mcp_json(generated_paths[".mcp.json"]))
    if "CLAUDE.md" in generated_paths:
        errors.extend(validate_markdown(generated_paths["CLAUDE.md"], "CLAUDE.md"))
    if "ROSCLAW.md" in generated_paths:
        errors.extend(validate_markdown(generated_paths["ROSCLAW.md"], "ROSCLAW.md"))
    if "context.snapshot.json" in generated_paths:
        errors.extend(validate_context_snapshot(generated_paths["context.snapshot.json"]))

    secrets: list[tuple[Path, int, str]] = []
    if not skip_secrets:
        secrets = scan_for_secrets(project_root, list(generated_paths.values()))
        if secrets:
            warnings.append(
                f"Potential secrets detected in {len(secrets)} location(s); "
                "review before committing."
            )

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors, warnings=warnings, secrets=secrets)


def check_merge_conflicts(
    base: dict[str, Any],
    override: dict[str, Any],
) -> list[str]:
    """Check whether merging override into base would create conflicts."""
    _, conflicts = json_merge_with_conflict_detection(base, override)
    return conflicts


__all__ = [
    "ValidationResult",
    "scan_for_secrets",
    "validate_mcp_json",
    "validate_markdown",
    "validate_context_snapshot",
    "validate_project",
    "check_merge_conflicts",
    "MergeConflictError",
]
