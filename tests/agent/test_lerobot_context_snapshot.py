"""Context snapshot LeRobot integration tests (终稿 §8.4).

Covers: snapshot block presence, no torch/lerobot import in core, redaction,
corrupt-config degradation, not-configured degradation, policy presence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rosclaw.agent.detectors import ProjectProfile
from rosclaw.agent.lerobot_status import detect_lerobot_integration
from rosclaw.agent.templates import render_context_snapshot


def _profile(tmp_path: Path) -> ProjectProfile:
    return ProjectProfile(
        project_root=tmp_path,
        robot_id="rh56_mock",
        default_transport="stdio",
        has_pyproject=False,
        has_rosclaw_src=False,
        profile_path=None,
        runtime_profile={"mcp": {}},
        cli_command="rosclaw",
        cli_args=(),
    )


def test_snapshot_contains_lerobot_block(tmp_path: Path) -> None:
    snapshot = render_context_snapshot(_profile(tmp_path))
    block = snapshot["integrations"]["lerobot"]
    assert "configured" in block
    assert "state" in block
    assert block["agent_action_entry"] == "mcp.request_action"
    assert block["direct_execution_allowed"] is False
    assert "inspire_rh56_left" in block["supported_bodies"]


def test_snapshot_no_torch_import_in_core(tmp_path: Path) -> None:
    before = set(sys.modules)
    render_context_snapshot(_profile(tmp_path))
    new_modules = set(sys.modules) - before
    assert not any(m.startswith("torch") for m in new_modules)
    assert not any(m.startswith("lerobot") for m in new_modules)


def test_snapshot_redaction(tmp_path: Path) -> None:
    snapshot = render_context_snapshot(_profile(tmp_path))
    block = json.dumps(snapshot["integrations"]["lerobot"])
    for forbidden in ("token", "api_key", "secret", "permit_", "/home/"):
        assert forbidden not in block


def test_detect_never_raises_on_corrupt_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    config_dir = home / "integrations"
    config_dir.mkdir(parents=True)
    (config_dir / "lerobot.yaml").write_text("{not: valid: yaml: [", encoding="utf-8")
    result = detect_lerobot_integration()
    assert result["configured"] is False or result["state"] in {
        "not_configured",
        "ready",
        "degraded",
        "runtime_missing",
        "policy_missing",
    }


def test_detect_not_configured_when_no_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / "empty_home"))
    result = detect_lerobot_integration()
    assert result["configured"] is False
    assert result["state"] == "not_configured"
    assert result["agent_action_entry"] == "mcp.request_action"


def test_detect_reports_reference_policy_presence(tmp_path: Path) -> None:
    result = detect_lerobot_integration()
    if result["configured"]:
        assert "reference_policy_present" in result
        assert result["reference_policy"] == "rosclaw_rh56_reference"
