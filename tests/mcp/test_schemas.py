"""Tests that P0 MCP schema responses contain the required fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.schemas.common import make_error, make_response
from rosclaw.mcp.tools import (
    emergency_stop,
    get_robot_state,
    list_skills,
    practice_query,
    query_memory,
    sandbox_run,
    set_client,
    validate_trajectory,
)


@pytest.fixture(autouse=True)
def _fixture_client() -> None:
    client = RuntimeClient(
        project_root=Path("/tmp/rosclaw-test"),
        robot_id="schema_bot",
        runtime_profile={},
        fixture_mode=True,
    )
    set_client(client)


def _data(name: str) -> dict:
    payload = json.loads(locals().get(name, {}))
    return payload["data"]


async def test_get_robot_state_schema() -> None:
    result = json.loads(await get_robot_state())
    data = result["data"]
    assert "robot_id" in data
    assert "mode" in data
    assert "body_state" in data
    assert "readiness" in data
    assert "is_stale" in data
    assert "age_ms" in data


async def test_list_skills_schema() -> None:
    result = json.loads(await list_skills())
    data = result["data"]
    assert "skills" in data
    assert "count" in data
    assert "mode" in data


async def test_query_memory_schema() -> None:
    result = json.loads(await query_memory("pick cup"))
    data = result["data"]
    assert "experiences" in data
    assert "count" in data
    assert "mode" in data


async def test_validate_trajectory_schema() -> None:
    result = json.loads(await validate_trajectory([[0.0] * 6, [0.1] * 6]))
    data = result["data"]
    assert "is_safe" in data
    assert "risk_score" in data
    assert "reason" in data
    assert "violations" in data
    assert "replay_id" in data


async def test_sandbox_run_schema() -> None:
    result = json.loads(await sandbox_run([0.1] * 6))
    data = result["data"]
    assert "physics_state" in data
    assert "mode" in data


async def test_practice_query_schema() -> None:
    result = json.loads(await practice_query())
    data = result["data"]
    assert "episodes" in data
    assert "count" in data
    assert "mode" in data


async def test_emergency_stop_schema() -> None:
    result = json.loads(await emergency_stop("schema test"))
    data = result["data"]
    assert "stopped" in data
    assert "reason" in data
    assert "mode" in data


def test_common_envelope_fields() -> None:
    envelope = make_response({"ok": True})
    assert envelope["ok"] is True
    assert envelope["schema_version"].startswith("p0.")
    assert "trace_id" in envelope
    assert "timestamp" in envelope
    assert "data" in envelope


def test_error_envelope_fields() -> None:
    envelope = make_error("TEST_CODE", "test message")
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "TEST_CODE"
    assert envelope["error"]["message"] == "test message"
    assert "details" in envelope["error"]
