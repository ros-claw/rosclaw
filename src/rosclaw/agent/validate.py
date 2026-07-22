"""Post-init validation and secrets scanning for agent onboarding files."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.agent.merge import MergeConflictError, json_merge_with_conflict_detection
from rosclaw.agent.templates import MANAGED_BEGIN, MANAGED_END
from rosclaw.agent.tool_catalog import P0_AGENT_MCP_TOOLS


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


def _has_rosclaw_stdio_command(server: dict[str, Any]) -> bool:
    command = server.get("command")
    args = server.get("args", [])
    if not isinstance(command, str) or not command:
        return False
    if Path(command).name == "rosclaw":
        return True
    return (
        isinstance(args, list)
        and len(args) >= 2
        and args[:2] == ["-m", "rosclaw.entrypoint"]
        and Path(command).name.startswith("python")
    )


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
    else:
        server = servers["rosclaw"]
        if not isinstance(server, dict):
            errors.append(f"{path} rosclaw server must be a JSON object")
        elif not _has_rosclaw_stdio_command(server) and not isinstance(server.get("url"), str):
            errors.append(f"{path} rosclaw server has no ROSClaw command or URL")
    return errors


def validate_codex_config(path: Path) -> list[str]:
    """Validate the project-scoped Codex ROSClaw MCP table."""
    errors: list[str] = []
    if not path.exists():
        return [f"Missing {path}"]
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"Invalid TOML in {path}: {exc}"]

    servers = data.get("mcp_servers")
    server = servers.get("rosclaw") if isinstance(servers, dict) else None
    if not isinstance(server, dict):
        return [f"{path} is missing mcp_servers.rosclaw"]
    if not _has_rosclaw_stdio_command(server) and not isinstance(server.get("url"), str):
        errors.append(f"{path} has no ROSClaw command or URL")
    enabled_tools = server.get("enabled_tools")
    if enabled_tools != list(P0_AGENT_MCP_TOOLS):
        errors.append(f"{path} does not enforce the canonical P0 tool allowlist")
    return errors


def validate_claude_settings(path: Path) -> list[str]:
    """Validate the Claude Code safety settings generated by ROSClaw."""
    if not path.exists():
        return [f"Missing {path}"]
    data = _load_json(path)
    if not isinstance(data, dict) or not data:
        return [f"{path} must be a valid JSON object"]
    permissions = data.get("permissions")
    deny = permissions.get("deny") if isinstance(permissions, dict) else None
    if not isinstance(deny, list) or not deny:
        return [f"{path} has no ROSClaw command deny rules"]
    return []


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


def validate_skill(path: Path) -> list[str]:
    """Validate the repo-local ROSClaw skill entry point."""
    if not path.exists():
        return [f"Missing {path}"]
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"Could not read {path}: {exc}"]
    if not content.startswith("---\n") or "\nname: rosclaw\n" not in content:
        return [f"{path} is missing the ROSClaw skill frontmatter"]
    return []


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
    if data.get("schema_version") != "rosclaw.agent.context.v2":
        errors.append(f"{path} has unexpected schema_version")
    tool_section = data.get("tools")
    tools = tool_section.get("available", []) if isinstance(tool_section, dict) else []
    missing = [t for t in P0_AGENT_MCP_TOOLS if t not in tools]
    if missing:
        errors.append(f"{path} is missing tools: {missing}")
    return errors


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def agent_target_paths(project_root: Path, target: str) -> dict[str, Path]:
    """Return files that a specific Agent target needs for a usable integration."""
    paths = {
        ".mcp.json": project_root / ".mcp.json",
        "ROSCLAW.md": project_root / "ROSCLAW.md",
        "context.snapshot.json": project_root / ".rosclaw" / "agent" / "context.snapshot.json",
    }
    if target in {"universal", "all", "claude-code"}:
        paths["CLAUDE.md"] = project_root / "CLAUDE.md"
        paths[".claude/settings.json"] = project_root / ".claude" / "settings.json"
    if target in {"universal", "all", "codex", "openclaw"}:
        paths["AGENTS.md"] = project_root / "AGENTS.md"
        paths[".agents/skills/rosclaw/SKILL.md"] = (
            project_root / ".agents" / "skills" / "rosclaw" / "SKILL.md"
        )
    if target in {"universal", "all", "codex"}:
        paths[".codex/config.toml"] = project_root / ".codex" / "config.toml"
    return paths


def validate_project(
    project_root: Path,
    generated_paths: dict[str, Path],
    skip_secrets: bool = False,
) -> ValidationResult:
    """Run all post-init validation checks and return a structured result."""
    errors: list[str] = []
    warnings: list[str] = []

    for key, path in generated_paths.items():
        if not path.exists():
            errors.append(f"Required file {key} is missing: {path}")

    if ".mcp.json" in generated_paths:
        errors.extend(validate_mcp_json(generated_paths[".mcp.json"]))
    if ".codex/config.toml" in generated_paths:
        errors.extend(validate_codex_config(generated_paths[".codex/config.toml"]))
    if ".claude/settings.json" in generated_paths:
        errors.extend(validate_claude_settings(generated_paths[".claude/settings.json"]))
    if "AGENTS.md" in generated_paths:
        errors.extend(validate_markdown(generated_paths["AGENTS.md"], "AGENTS.md"))
    if "CLAUDE.md" in generated_paths:
        errors.extend(validate_markdown(generated_paths["CLAUDE.md"], "CLAUDE.md"))
    if "ROSCLAW.md" in generated_paths:
        errors.extend(validate_markdown(generated_paths["ROSCLAW.md"], "ROSCLAW.md"))
    if ".agents/skills/rosclaw/SKILL.md" in generated_paths:
        errors.extend(validate_skill(generated_paths[".agents/skills/rosclaw/SKILL.md"]))
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
    "agent_target_paths",
    "scan_for_secrets",
    "validate_claude_settings",
    "validate_mcp_json",
    "validate_codex_config",
    "validate_markdown",
    "validate_skill",
    "validate_context_snapshot",
    "validate_project",
    "check_merge_conflicts",
    "MergeConflictError",
]
