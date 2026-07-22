"""Agent guidance safety contract tests (终稿 §8.3 + §8.5).

Verifies every generated onboarding file carries the required LeRobot
workflow guidance AND the required prohibitions, without leaking a direct
execution path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.agent.detectors import ProjectProfile
from rosclaw.agent.templates import (
    render_agents_md,
    render_claude_md,
    render_context_snapshot,
    render_rosclaw_md,
    render_rosclaw_skill_md,
)


@pytest.fixture
def profile(tmp_path: Path) -> ProjectProfile:
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


@pytest.fixture
def all_docs(profile) -> dict[str, str]:
    return {
        "CLAUDE.md": render_claude_md(profile),
        "AGENTS.md": render_agents_md(profile),
        "ROSCLAW.md": render_rosclaw_md(profile),
        "SKILL.md": render_rosclaw_skill_md(profile),
    }


def test_all_docs_contain_lerobot_workflow(all_docs) -> None:
    for name, content in all_docs.items():
        assert "LeRobot" in content, f"{name} missing LeRobot guidance"


def test_guidance_requires_request_action(all_docs) -> None:
    for name, content in all_docs.items():
        assert "request_action" in content, f"{name} missing request_action"


def test_guidance_requires_shadow_before_real(all_docs) -> None:
    for name, content in all_docs.items():
        assert "SHADOW" in content, f"{name} missing SHADOW guidance"


def test_guidance_requires_receipt(all_docs) -> None:
    for name, content in all_docs.items():
        assert "get_execution_receipt" in content, f"{name} missing receipt guidance"
        assert "explain_execution" in content, f"{name} missing explain_execution"


def test_guidance_forbids_direct_rollout_execute(all_docs) -> None:
    for name, content in all_docs.items():
        assert "rollout execute" in content, (
            f"{name} must explicitly forbid direct rollout execute"
        )


def test_guidance_forbids_serial_and_vendor(all_docs) -> None:
    for name, content in all_docs.items():
        assert "serial" in content.lower(), f"{name} missing serial prohibition"
        assert "vendor SDK" in content, f"{name} missing vendor SDK prohibition"


def test_guidance_forbids_self_issued_permit(all_docs) -> None:
    for name, content in all_docs.items():
        lowered = content.lower()
        assert "never create or approve a permit" in lowered or "does not issue or approve a permit" in lowered, (
            f"{name} missing self-permit prohibition"
        )


def test_skill_contains_decision_table(all_docs) -> None:
    skill = all_docs["SKILL.md"]
    assert "Decision table" in skill
    assert "AUTHORIZATION_REQUIRED" in skill


def test_skill_contains_authorization_required_explanation(all_docs) -> None:
    skill = all_docs["SKILL.md"]
    assert "AUTHORIZATION_REQUIRED" in skill
    assert "rosclawd" in skill


def test_snapshot_tools_include_action_tools(profile) -> None:
    snapshot = render_context_snapshot(profile)
    tools = snapshot["tools"]["available"]
    for name in (
        "get_product_status",
        "get_runtime_status",
        "request_action",
        "get_action_status",
        "get_execution_receipt",
        "explain_execution",
    ):
        assert name in tools
