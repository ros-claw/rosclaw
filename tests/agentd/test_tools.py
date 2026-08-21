"""Tests for the built-in agentd tool registry (P0 sim tools)."""

from __future__ import annotations

import asyncio

import pytest

from rosclaw.agentd.tools import (
    SIM_BODY_TOOL,
    SIM_REACH_TOOL,
    SIM_STATE_TOOL,
    BuiltinToolRegistry,
)
from rosclaw.contracts.common import ValidationError


def _registry() -> BuiltinToolRegistry:
    return BuiltinToolRegistry(body_id="sim/ur5e", body_summary="simulated UR5e test body")


def test_state_tool_marks_evidence_simulated() -> None:
    payload = asyncio.run(_registry().execute(SIM_STATE_TOOL, {"verbose": False}))
    assert payload["evidence_class"] == "simulated"
    assert payload["mode"] == "SIMULATION"
    assert payload["health"] == "OK"


def test_body_tool_returns_configured_summary() -> None:
    payload = asyncio.run(_registry().execute(SIM_BODY_TOOL, {"verbose": False}))
    assert payload["evidence_class"] == "configured"
    assert payload["summary"] == "simulated UR5e test body"


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not allowlisted"):
        asyncio.run(_registry().execute("shell_exec", {}))


def test_reach_tool_runs_physics_and_reports_success() -> None:
    payload = asyncio.run(
        _registry().execute(SIM_REACH_TOOL, {"x": -0.1, "y": 0.5, "z": 0.25})
    )
    assert payload["evidence_class"] == "simulated"
    assert payload["final_state"] == "COMPLETED"
    assert payload["task_success"] is True
    assert payload["evidence_verified"] is True
    assert payload["collision_check"] == "PASS"
    assert payload["final_distance_m"] < 0.008
    assert payload["physics_steps"] > 0


def test_reach_tool_reports_blocked_target_honestly() -> None:
    payload = asyncio.run(
        _registry().execute(SIM_REACH_TOOL, {"x": 0.4, "y": 0.1, "z": 0.12})
    )
    assert payload["final_state"] == "BLOCKED"
    assert payload["task_success"] is False
    assert payload["collision_check"] == "FAIL"


def test_reach_tool_rejects_non_finite_target() -> None:
    with pytest.raises(ValidationError, match="target invalid"):
        asyncio.run(_registry().execute(SIM_REACH_TOOL, {"x": float("nan"), "y": 0.0, "z": 0.1}))
