"""Tests for `rosclaw agent test`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import rosclaw.agent.test_claude_code as test_module
from rosclaw.agent.init_claude_code import cmd_agent_init_claude_code
from rosclaw.agent.install import cmd_agent_install
from rosclaw.agent.test_claude_code import _assess_probe_payload, cmd_agent_test_claude_code


def _make_init_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        project_root=str(tmp_path),
        profile=None,
        robot=None,
        transport=None,
        host=None,
        port=None,
        check=False,
        dry_run=False,
        skip_secrets=True,
    )


def _make_test_args(
    tmp_path: Path,
    *,
    agent: str = "claude-code",
    quick: bool = True,
    verbose: bool = False,
    mcp_probe: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        agent=agent,
        project_root=str(tmp_path),
        quick=quick,
        verbose=verbose,
        mcp_probe=mcp_probe,
    )


def _make_install_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        target="universal",
        project_root=str(tmp_path),
        profile=None,
        robot=None,
        transport=None,
        host=None,
        port=None,
        check=False,
        dry_run=False,
        skip_secrets=True,
    )


def _bootstrap_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "rosclaw"\n', encoding="utf-8")


async def test_test_command_quick_passes_after_init(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_install_args(tmp_path)) == 0
    assert cmd_agent_test_claude_code(_make_test_args(tmp_path)) == 0
    captured = capsys.readouterr()
    assert ".mcp.json: OK" in captured.out
    assert "Tools advertised: 18" in captured.out


async def test_test_command_accepts_cross_agent_targets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_install_args(tmp_path)) == 0

    assert cmd_agent_test_claude_code(_make_test_args(tmp_path, agent="codex")) == 0
    assert cmd_agent_test_claude_code(_make_test_args(tmp_path, agent="openclaw")) == 0

    captured = capsys.readouterr()
    assert "Agent target: codex" in captured.out
    assert "Agent target: openclaw" in captured.out


async def test_codex_quick_test_requires_guidance_and_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_install_args(tmp_path)) == 0
    capsys.readouterr()
    (tmp_path / "AGENTS.md").unlink()

    assert cmd_agent_test_claude_code(
        _make_test_args(tmp_path, agent="codex")
    ) == 1

    captured = capsys.readouterr()
    assert "AGENTS.md: MISSING" in captured.out


async def test_test_command_quick_fails_without_init(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_test_claude_code(_make_test_args(tmp_path)) == 1
    captured = capsys.readouterr()
    assert ".mcp.json: MISSING" in captured.out


async def test_test_command_mcp_probe_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_init_claude_code(_make_init_args(tmp_path)) == 0

    monkeypatch.setattr(
        test_module,
        "_run_mcp_probe",
        lambda _server_config, *, project_root: test_module.McpProbeResult(
            True,
            tools=["get_robot_state", "sandbox_run"],
            readiness_limits=[
                "get_robot_state: unavailable as expected (SYNTHETIC_SOURCE_NOT_ALLOWED)"
            ],
            verified_run_id="run-test-123",
        ),
    )

    assert cmd_agent_test_claude_code(_make_test_args(tmp_path, mcp_probe=True)) == 0
    captured = capsys.readouterr()
    assert "MCP stdio probe: OK" in captured.out
    assert "MCP tools discovered: 2" in captured.out
    assert "MCP verified simulation run: run-test-123" in captured.out
    assert "Readiness: get_robot_state: unavailable as expected" in captured.out


def test_probe_accepts_truthful_readiness_failure() -> None:
    error, readiness = _assess_probe_payload(
        "get_robot_state",
        {
            "schema_version": "rosclaw.mcp.v1",
            "ok": False,
            "error": {
                "code": "SYNTHETIC_SOURCE_NOT_ALLOWED",
                "message": "fixture mode was not requested",
                "details": {
                    "trust_level": "UNAVAILABLE",
                    "usable_for_real_execution": False,
                },
            },
        },
    )

    assert error is None
    assert readiness == ("get_robot_state: unavailable as expected (SYNTHETIC_SOURCE_NOT_ALLOWED)")


def test_probe_rejects_internal_runtime_error() -> None:
    error, readiness = _assess_probe_payload(
        "sandbox_run",
        {
            "schema_version": "rosclaw.mcp.v1",
            "ok": False,
            "error": {
                "code": "RUNTIME_ERROR",
                "message": "unexpected exception",
                "details": {"usable_for_real_execution": False},
            },
        },
    )

    assert "returned ok=false" in str(error)
    assert readiness is None
