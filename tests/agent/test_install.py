"""Tests for `rosclaw agent install`."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import pytest

from rosclaw.agent.install import cmd_agent_install
from rosclaw.agent.tool_catalog import P0_AGENT_MCP_TOOLS


def _make_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    defaults = {
        "target": "universal",
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
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "rosclaw"\n', encoding="utf-8")


async def test_install_generates_cross_agent_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)

    assert cmd_agent_install(_make_args(tmp_path)) == 0

    assert (tmp_path / ".mcp.json").exists()
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / "ROSCLAW.md").exists()
    assert (tmp_path / ".agents/skills/rosclaw/SKILL.md").exists()
    assert (tmp_path / ".codex/config.toml").exists()
    assert (tmp_path / ".rosclaw/agent/context.snapshot.json").exists()
    mcp_config = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert (
        mcp_config["mcpServers"]["rosclaw"]["env"]["ROSCLAW_AGENT_CLIENT"]
        == "${ROSCLAW_AGENT_CLIENT:-universal-agent}"
    )
    with (tmp_path / ".codex/config.toml").open("rb") as file:
        codex_config = tomllib.load(file)
    codex_server = codex_config["mcp_servers"]["rosclaw"]
    assert codex_server["command"] == "rosclaw"
    assert codex_server["env"]["ROSCLAW_AGENT_CLIENT"] == "codex"
    assert codex_server["default_tools_approval_mode"] == "approve"
    assert codex_server["enabled_tools"] == list(P0_AGENT_MCP_TOOLS)

    agents_guide = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    runtime_guide = (tmp_path / "ROSCLAW.md").read_text(encoding="utf-8")
    skill_guide = (tmp_path / ".agents/skills/rosclaw/SKILL.md").read_text(encoding="utf-8")
    assert "request_action" in agents_guide
    assert "request_action" in runtime_guide
    assert "request_action" in skill_guide
    assert "Never instantiate" in agents_guide
    assert "Never instantiate a local Runtime" in skill_guide
    assert "direct ROS, DDS, serial, CAN, SDK, or motor commands" in skill_guide
    assert "Request daemon E-Stop; verify physical-stop evidence" in agents_guide
    assert "Halt all motion immediately" not in agents_guide

    snapshot = json.loads((tmp_path / ".rosclaw/agent/context.snapshot.json").read_text())
    assert snapshot["schema_version"] == "rosclaw.agent.context.v2"
    assert snapshot["tools"]["available"] == list(P0_AGENT_MCP_TOOLS)
    assert snapshot["policies"]["direct_hardware_access"] is False
    assert snapshot["policies"]["real_execution_requires_rosclawd_permit"] is True
    assert snapshot["policies"]["agent_may_self_authorize"] is False

    captured = capsys.readouterr()
    assert "ROSClaw universal agent integration installed." in captured.out
    assert "rosclaw agent install --project-root . --skip-secrets" in captured.out
    assert "rosclaw agent test universal --project-root . --quick --mcp-probe" in captured.out
    assert "same environment and PATH" in captured.out


async def test_install_dry_run_does_not_write(tmp_path: Path) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_args(tmp_path, dry_run=True)) == 0
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / ".agents/skills/rosclaw/SKILL.md").exists()


async def test_install_preserves_existing_unmanaged_agents_md(tmp_path: Path) -> None:
    _bootstrap_project(tmp_path)
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Existing Agent Guide\n\nKeep this project rule.\n", encoding="utf-8")

    assert cmd_agent_install(_make_args(tmp_path)) == 0

    content = agents.read_text(encoding="utf-8")
    assert "Keep this project rule." in content
    assert "<!-- ROSCLAW-MANAGED-BEGIN -->" in content
    assert "rosclaw agent install --project-root . --skip-secrets" in content
    assert content.count("# Existing Agent Guide") == 1
    assert content.count("# ROSClaw Agent Instructions") == 0


async def test_install_preserves_unmanaged_codex_config(tmp_path: Path) -> None:
    _bootstrap_project(tmp_path)
    config = tmp_path / ".codex/config.toml"
    config.parent.mkdir()
    config.write_text('model_reasoning_effort = "high"\n', encoding="utf-8")

    assert cmd_agent_install(_make_args(tmp_path)) == 0
    assert cmd_agent_install(_make_args(tmp_path)) == 0

    content = config.read_text(encoding="utf-8")
    assert 'model_reasoning_effort = "high"' in content
    assert content.count("# ROSCLAW-MANAGED-BEGIN") == 1
    with config.open("rb") as file:
        parsed = tomllib.load(file)
    assert parsed["mcp_servers"]["rosclaw"]["command"] == "rosclaw"


async def test_install_rejects_unmanaged_codex_server_conflict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    config = tmp_path / ".codex/config.toml"
    config.parent.mkdir()
    config.write_text(
        '[mcp_servers.rosclaw]\ncommand = "custom-rosclaw"\n',
        encoding="utf-8",
    )

    assert cmd_agent_install(_make_args(tmp_path)) == 1
    assert "already defines mcp_servers.rosclaw" in capsys.readouterr().err
    assert "ROSCLAW-MANAGED-BEGIN" not in config.read_text(encoding="utf-8")


async def test_install_rejects_conflicting_mcp_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    mcp_config = tmp_path / ".mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "rosclaw": {
                        "type": "stdio",
                        "command": "custom-runtime",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert cmd_agent_install(_make_args(tmp_path)) == 1

    captured = capsys.readouterr()
    assert "has no ROSClaw command or URL" in captured.err
    merged = json.loads(mcp_config.read_text(encoding="utf-8"))
    assert merged["mcpServers"]["rosclaw"]["command"] == "custom-runtime"


async def test_install_rejects_duplicate_codex_table_without_rewriting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_args(tmp_path)) == 0
    capsys.readouterr()
    config = tmp_path / ".codex/config.toml"
    conflicting = (
        config.read_text(encoding="utf-8") + '\n[mcp_servers.rosclaw]\ncommand = "duplicate"\n'
    )
    config.write_text(conflicting, encoding="utf-8")

    assert cmd_agent_install(_make_args(tmp_path)) == 1

    assert "would produce invalid TOML" in capsys.readouterr().err
    assert config.read_text(encoding="utf-8") == conflicting
