"""Tests for the P0 MCP body tools exposed through RuntimeClient."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.service import BodyInstanceService
from rosclaw.mcp.adapters.runtime_client import RuntimeClient


@pytest.fixture
def linked_workspace(tmp_path: Path, monkeypatch) -> Path:
    """Create a workspace with a linked unitree-g1 body under HOME/.rosclaw."""
    monkeypatch.setenv("HOME", str(tmp_path))
    service = BodyInstanceService()
    service.create_or_init(robot="unitree-g1", name="g1-test", mode="single")
    return tmp_path


@pytest.fixture
def client(linked_workspace: Path) -> RuntimeClient:
    return RuntimeClient(
        project_root=linked_workspace,
        robot_id="unitree-g1",
        runtime_profile={},
        fixture_mode=True,
    )


@pytest.mark.asyncio
async def test_get_body_profile(client: RuntimeClient):
    result = await client.get_body_profile()
    assert result["mode"] == "live"
    profile = result["profile"]
    assert profile["body_instance_id"] == "g1-test"
    assert profile["robot_model"] == "unitree-g1"
    assert profile["eurdf_uri"].startswith("rosclaw://eurdf/")
    assert profile["effective_body_hash"]


@pytest.mark.asyncio
async def test_get_body_state(client: RuntimeClient):
    result = await client.get_body_state()
    assert result["mode"] == "live"
    state = result["state"]
    assert state["body_instance_id"] == "g1-test"
    assert "enabled_capabilities" in state
    assert "forbidden_capabilities" in state
    assert "calibration_status" in state


@pytest.mark.asyncio
async def test_list_body_capabilities(client: RuntimeClient):
    result = await client.list_body_capabilities(status="all")
    assert result["mode"] == "live"
    caps = result["capabilities"]
    assert "enabled" in caps
    assert "degraded" in caps
    assert "disabled" in caps
    assert "forbidden" in caps


@pytest.mark.asyncio
async def test_list_body_capabilities_filtered(client: RuntimeClient):
    result = await client.list_body_capabilities(status="enabled")
    assert result["mode"] == "live"
    assert "enabled" in result["capabilities"]


@pytest.mark.asyncio
async def test_query_body_identity(client: RuntimeClient):
    result = await client.query_body("What robot body is this?")
    assert result["mode"] == "live"
    q = result["result"]
    assert "answer" in q
    assert "decision" in q
    assert "evidence" in q
    assert "unitree-g1" in q["answer"].lower() or "g1-test" in q["answer"].lower()


@pytest.mark.asyncio
async def test_query_body_bypass_sandbox_refused(client: RuntimeClient):
    result = await client.query_body("Can I bypass sandbox validation?")
    assert result["mode"] == "live"
    q = result["result"]
    assert q["decision"] == "blocked"
    assert "sandbox" in q["answer"].lower()


@pytest.mark.asyncio
async def test_validate_body_action_forbidden(client: RuntimeClient):
    result = await client.validate_body_action("bypass sandbox", "bypass_sandbox", risk="critical")
    assert result["mode"] == "live"
    v = result["validation"]
    assert v["allowed_to_propose"] is False
    assert v["allowed_to_execute_real_robot"] is False


@pytest.mark.asyncio
async def test_validate_body_action_unknown_capability(client: RuntimeClient):
    result = await client.validate_body_action("fly", "fly", risk="high")
    assert result["mode"] == "live"
    v = result["validation"]
    assert v["body_check"] == "unknown"
    assert v["allowed_to_execute_real_robot"] is False


@pytest.mark.asyncio
async def test_get_calibration_status_overall(client: RuntimeClient):
    result = await client.get_calibration_status()
    assert result["mode"] == "live"
    cal = result["calibration"]
    assert cal["component"] == "*"
    assert cal["status"]


@pytest.mark.asyncio
async def test_get_calibration_status_component(client: RuntimeClient):
    result = await client.get_calibration_status(component="head_rgb_camera")
    assert result["mode"] == "live"
    cal = result["calibration"]
    assert cal["component"] == "head_rgb_camera"
    assert "status" in cal
