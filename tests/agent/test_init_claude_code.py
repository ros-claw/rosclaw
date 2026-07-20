"""Tests for `rosclaw agent init claude-code`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw.agent.init_claude_code import cmd_agent_init_claude_code
from rosclaw.agent.merge import managed_block_merge
from rosclaw.agent.tool_catalog import P0_AGENT_MCP_TOOLS
from rosclaw.agent.validate import validate_project


def _make_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    defaults = {
        "project_root": str(tmp_path),
        "profile": None,
        "robot": None,
        "transport": None,
        "host": None,
        "port": None,
        "check": False,
        "dry_run": False,
        "skip_secrets": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _bootstrap_project(tmp_path: Path) -> None:
    """Create a minimal ROSClaw project root in tmp_path."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "rosclaw"\n', encoding="utf-8")


async def test_init_generates_all_files(tmp_path: Path) -> None:
    _bootstrap_project(tmp_path)
    args = _make_args(tmp_path)
    assert cmd_agent_init_claude_code(args) == 0

    expected = [
        ".mcp.json",
        "CLAUDE.md",
        "ROSCLAW.md",
        ".claude/settings.json",
        ".rosclaw/agent/context.snapshot.json",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), f"missing {name}"

    generated = {
        ".mcp.json": tmp_path / ".mcp.json",
        "CLAUDE.md": tmp_path / "CLAUDE.md",
        "ROSCLAW.md": tmp_path / "ROSCLAW.md",
        ".claude/settings.json": tmp_path / ".claude/settings.json",
        "context.snapshot.json": tmp_path / ".rosclaw/agent/context.snapshot.json",
    }
    result = validate_project(tmp_path, generated)
    assert result.ok, result.errors

    snapshot = json.loads((tmp_path / ".rosclaw/agent/context.snapshot.json").read_text())
    assert snapshot["schema_version"] == "rosclaw.agent.context.v2"
    available = snapshot["tools"]["available"]
    for tool in P0_AGENT_MCP_TOOLS:
        assert tool in available


async def test_init_dry_run_does_not_write(tmp_path: Path) -> None:
    _bootstrap_project(tmp_path)
    args = _make_args(tmp_path, dry_run=True)
    assert cmd_agent_init_claude_code(args) == 0
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / "CLAUDE.md").exists()


async def test_init_idempotent_preserves_human_notes(tmp_path: Path) -> None:
    _bootstrap_project(tmp_path)
    first = _make_args(tmp_path)
    assert cmd_agent_init_claude_code(first) == 0

    human_note = "\n\n## Maintainer notes\n\nKeep this note.\n"
    claude_path = tmp_path / "CLAUDE.md"
    claude_path.write_text(claude_path.read_text(encoding="utf-8") + human_note, encoding="utf-8")

    second = _make_args(tmp_path, robot="turtle1")
    assert cmd_agent_init_claude_code(second) == 0

    content = claude_path.read_text(encoding="utf-8")
    assert "Keep this note." in content
    assert "turtle1" in content


async def test_managed_block_merge_replaces_managed_section() -> None:
    existing = "head\n<!-- BEGIN -->\nold\n<!-- END -->\ntail"
    new = "<!-- BEGIN -->\nnew\n<!-- END -->"
    merged = managed_block_merge(existing, new, "<!-- BEGIN -->", "<!-- END -->")
    assert "new" in merged
    assert "old" not in merged
    assert "head" in merged
    assert "tail" in merged


async def test_init_with_profile_detects_transport(tmp_path: Path) -> None:
    _bootstrap_project(tmp_path)
    profile_dir = tmp_path / ".rosclaw/profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / "default.yaml").write_text(
        "robot:\n  id: profile_bot\nmcp:\n  transport: http\n  port: 1234\n",
        encoding="utf-8",
    )
    args = _make_args(tmp_path)
    assert cmd_agent_init_claude_code(args) == 0

    mcp = json.loads((tmp_path / ".mcp.json").read_text())
    assert mcp["rosclaw"]["robot_id"] == "profile_bot"
    assert mcp["rosclaw"]["transport"] == "http"
