"""Command handler for `rosclaw agent init claude-code`."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

from rosclaw.agent.detectors import ProjectProfile, build_project_profile
from rosclaw.agent.merge import (
    atomic_write_json,
    atomic_write_text,
    backup_file,
    json_merge_with_conflict_detection,
    managed_block_merge,
    read_json_if_exists,
)
from rosclaw.agent.templates import (
    CODEX_MANAGED_BEGIN,
    CODEX_MANAGED_END,
    render_agents_md,
    render_claude_md,
    render_claude_settings_json,
    render_codex_config_toml,
    render_context_snapshot,
    render_mcp_json,
    render_rosclaw_md,
    render_rosclaw_skill_md,
)
from rosclaw.agent.validate import validate_project

_AGENT_DIR = Path(".rosclaw") / "agent"


class AgentConfigConflictError(RuntimeError):
    """Raised when ROSClaw cannot safely merge an agent configuration."""


def _write_markdown(
    path: Path,
    content: str,
    *,
    backup: bool = True,
    preserve_unmanaged: bool = False,
) -> None:
    if path.exists() and backup:
        backup_file(path)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if (
            preserve_unmanaged
            and "<!-- ROSCLAW-MANAGED-BEGIN -->" not in existing
            and "<!-- ROSCLAW-MANAGED-END -->" not in existing
        ):
            merged = existing.rstrip() + "\n\n" + content
        else:
            merged = managed_block_merge(
                existing,
                content,
                "<!-- ROSCLAW-MANAGED-BEGIN -->",
                "<!-- ROSCLAW-MANAGED-END -->",
            )
    else:
        merged = content
    atomic_write_text(path, merged)


def _write_json_file(
    path: Path,
    new_data: dict[str, Any],
    *,
    backup: bool = True,
    check_mode: bool = False,
) -> list[str]:
    conflicts: list[str] = []
    if path.exists() and backup:
        backup_file(path)

    existing = read_json_if_exists(path)
    if existing and new_data:
        merged, conflicts = json_merge_with_conflict_detection(existing, new_data)
    else:
        merged = new_data

    if conflicts and not check_mode:
        # In normal mode we still write the merged result but warn. The doctor
        # command can be used to inspect conflicts afterward.
        pass

    atomic_write_json(path, merged)
    return conflicts


def _write_claude_settings(
    project_root: Path,
    profile: ProjectProfile,
    *,
    backup: bool = True,
) -> list[str]:
    settings_dir = project_root / ".claude"
    settings_path = settings_dir / "settings.json"
    new_settings = render_claude_settings_json(profile)
    return _write_json_file(settings_path, new_settings, backup=backup)


def _write_codex_config(path: Path, content: str) -> None:
    """Merge the managed ROSClaw table into a project Codex config."""
    if not path.exists():
        try:
            tomllib.loads(content)
        except tomllib.TOMLDecodeError as exc:
            raise AgentConfigConflictError(
                f"Generated ROSClaw configuration for {path} is invalid TOML: {exc}"
            ) from exc
        atomic_write_text(path, content)
        return

    existing = path.read_text(encoding="utf-8")
    begin_count = existing.count(CODEX_MANAGED_BEGIN)
    end_count = existing.count(CODEX_MANAGED_END)
    if begin_count != end_count or begin_count > 1:
        raise AgentConfigConflictError(
            f"{path} must contain at most one complete ROSClaw managed block; "
            "repair it before installing."
        )

    if begin_count == 1:
        merged = managed_block_merge(
            existing,
            content,
            CODEX_MANAGED_BEGIN,
            CODEX_MANAGED_END,
        )
    else:
        try:
            parsed = tomllib.loads(existing)
        except tomllib.TOMLDecodeError as exc:
            raise AgentConfigConflictError(f"{path} is invalid TOML: {exc}") from exc
        servers = parsed.get("mcp_servers", {})
        if isinstance(servers, dict) and "rosclaw" in servers:
            raise AgentConfigConflictError(
                f"{path} already defines mcp_servers.rosclaw outside the managed block."
            )
        merged = existing.rstrip() + "\n\n" + content

    try:
        tomllib.loads(merged)
    except tomllib.TOMLDecodeError as exc:
        raise AgentConfigConflictError(
            f"Merging ROSClaw into {path} would produce invalid TOML: {exc}"
        ) from exc
    if merged == existing:
        return
    backup_file(path)
    atomic_write_text(path, merged)


def _generate_files(
    project_root: Path,
    profile: ProjectProfile,
    *,
    check_mode: bool = False,
    dry_run: bool = False,
    include_universal: bool = False,
) -> dict[str, Path]:
    """Generate or update all onboarding files. Returns a mapping of logical
    names to written paths.
    """
    agent_client = (
        "${ROSCLAW_AGENT_CLIENT:-universal-agent}" if include_universal else "claude-code"
    )
    mcp_json = render_mcp_json(profile, check=check_mode, agent_client=agent_client)
    claude_md = render_claude_md(profile)
    rosclaw_md = render_rosclaw_md(profile)
    snapshot = render_context_snapshot(profile)
    agents_md = render_agents_md(profile) if include_universal else ""
    skill_md = render_rosclaw_skill_md(profile) if include_universal else ""
    codex_config = render_codex_config_toml(profile) if include_universal else ""

    generated: dict[str, Path] = {}

    paths = {
        ".mcp.json": project_root / ".mcp.json",
        "AGENTS.md": project_root / "AGENTS.md",
        "CLAUDE.md": project_root / "CLAUDE.md",
        "ROSCLAW.md": project_root / "ROSCLAW.md",
        ".agents/skills/rosclaw/SKILL.md": project_root
        / ".agents"
        / "skills"
        / "rosclaw"
        / "SKILL.md",
        ".claude/settings.json": project_root / ".claude" / "settings.json",
        ".codex/config.toml": project_root / ".codex" / "config.toml",
        "context.snapshot.json": project_root / ".rosclaw" / "agent" / "context.snapshot.json",
    }

    if dry_run:
        selected = {
            key: paths[key]
            for key in (
                ".mcp.json",
                "CLAUDE.md",
                "ROSCLAW.md",
                ".claude/settings.json",
                "context.snapshot.json",
            )
        }
        if include_universal:
            for key in (
                "AGENTS.md",
                ".agents/skills/rosclaw/SKILL.md",
                ".codex/config.toml",
            ):
                selected[key] = paths[key]
        return selected

    # .mcp.json
    _write_json_file(paths[".mcp.json"], mcp_json)
    generated[".mcp.json"] = paths[".mcp.json"]

    if include_universal:
        _write_markdown(paths["AGENTS.md"], agents_md, preserve_unmanaged=True)
        generated["AGENTS.md"] = paths["AGENTS.md"]

        skill_path = paths[".agents/skills/rosclaw/SKILL.md"]
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(skill_path, skill_md)
        generated[".agents/skills/rosclaw/SKILL.md"] = skill_path

        _write_codex_config(paths[".codex/config.toml"], codex_config)
        generated[".codex/config.toml"] = paths[".codex/config.toml"]

    # CLAUDE.md
    _write_markdown(paths["CLAUDE.md"], claude_md)
    generated["CLAUDE.md"] = paths["CLAUDE.md"]

    # ROSCLAW.md
    _write_markdown(paths["ROSCLAW.md"], rosclaw_md)
    generated["ROSCLAW.md"] = paths["ROSCLAW.md"]

    # .claude/settings.json
    _write_claude_settings(project_root, profile)
    generated[".claude/settings.json"] = paths[".claude/settings.json"]

    # context.snapshot.json
    snapshot_dir = paths["context.snapshot.json"].parent
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["context.snapshot.json"], snapshot)
    generated["context.snapshot.json"] = paths["context.snapshot.json"]

    return generated


def cmd_agent_init_claude_code(args: argparse.Namespace) -> int:
    """Implementation of `rosclaw agent init claude-code`."""
    project_root = Path(args.project_root) if args.project_root else None
    profile = build_project_profile(
        project_root=project_root,
        profile=args.profile,
        robot=args.robot,
        transport=args.transport,
        host=args.host,
        port=args.port,
    )

    generated = _generate_files(
        profile.project_root,
        profile,
        check_mode=args.check,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("Would generate the following files:")
        for key, path in generated.items():
            print(f"  {key}: {path}")
        return 0

    validation = validate_project(
        profile.project_root,
        generated,
        skip_secrets=args.skip_secrets,
    )

    if not validation.ok:
        print("ERROR: generated files failed validation:", file=sys.stderr)
        for err in validation.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("ROSClaw Claude Code onboarding initialized.")
    print(f"Project root: {profile.project_root}")
    if profile.robot_id:
        print(f"Robot ID: {profile.robot_id}")
    print(f"MCP transport: {profile.default_transport}")
    print("Generated files:")
    for key in generated:
        print(f"  - {key}")

    if validation.warnings:
        print("Warnings:")
        for warn in validation.warnings:
            print(f"  - {warn}")

    return 0


def add_init_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "init",
        help="Initialize agent onboarding files for Claude Code.",
    )
    parser.add_argument("agent", choices=["claude-code"], help="Agent target.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path.")
    parser.add_argument("--profile", type=str, default=None, help="Runtime profile YAML path.")
    parser.add_argument("--robot", type=str, default=None, help="Explicit robot ID.")
    parser.add_argument(
        "--transport", choices=["stdio", "http", "sse"], default=None, help="MCP transport."
    )
    parser.add_argument("--host", type=str, default=None, help="MCP HTTP/SSE host.")
    parser.add_argument("--port", type=int, default=None, help="MCP HTTP/SSE port.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Generate files in check/audit mode without making permanent changes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be generated without writing them.",
    )
    parser.add_argument(
        "--skip-secrets",
        action="store_true",
        help="Skip the heuristic secrets scan.",
    )
    parser.set_defaults(func=cmd_agent_init_claude_code)


__all__ = ["cmd_agent_init_claude_code", "add_init_parser"]
