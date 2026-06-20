"""Tests for MCP body registry and fleet compatibility tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from rosclaw.body.registry import BodyRegistryManager
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.tools import (
    check_skill_compatibility,
    fleet_skill_compatibility,
    get_body,
    list_bodies,
    list_body_history,
    set_client,
    switch_body,
)


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def client(tmp_path: Path) -> RuntimeClient:
    client = RuntimeClient(
        project_root=tmp_path,
        robot_id=None,
        runtime_profile={},
        fixture_mode=True,
    )
    set_client(client)
    return client


@pytest.fixture
def two_bodies() -> Path:
    ws = Path.home() / ".rosclaw"
    manager = BodyRegistryManager(ws)
    manager.create_body("g1-sim", "unitree-g1")
    manager.create_body("g1-real", "unitree-g1")
    return ws


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_list_bodies_tool(client: RuntimeClient, two_bodies: Path) -> None:
    result = _run(list_bodies())
    data = _json(result)
    assert data["mode"] == "live"
    assert data["total"] == 2
    assert {b["body_id"] for b in data["bodies"]} == {"g1-sim", "g1-real"}


def test_get_body_tool(client: RuntimeClient, two_bodies: Path) -> None:
    result = _run(get_body("g1-sim"))
    data = _json(result)
    assert data["mode"] == "live"
    assert data["body"]["body_id"] == "g1-sim"
    assert "effective_body" in data


def test_switch_body_tool(client: RuntimeClient, two_bodies: Path) -> None:
    result = _run(switch_body("g1-sim"))
    data = _json(result)
    assert data["mode"] == "live"
    assert data["current_body_id"] == "g1-sim"
    assert BodyRegistryManager(two_bodies).get_current_body_id() == "g1-sim"


def test_list_body_history_tool(client: RuntimeClient, two_bodies: Path) -> None:
    result = _run(list_body_history("g1-sim"))
    data = _json(result)
    assert data["mode"] == "live"
    assert isinstance(data["snapshots"], list)


def test_check_skill_compatibility_tool(client: RuntimeClient, two_bodies: Path) -> None:
    result = _run(check_skill_compatibility())
    data = _json(result)
    assert data["mode"] == "live"
    assert "report" in data


def test_fleet_skill_compatibility_tool(client: RuntimeClient, two_bodies: Path) -> None:
    result = _run(fleet_skill_compatibility())
    data = _json(result)
    assert data["mode"] == "live"
    report = data["report"]
    assert report["fleet_summary"]["total_bodies"] == 2


def _json(result: str) -> dict[str, Any]:
    import json

    envelope = json.loads(result)
    assert envelope.get("ok") is True, envelope
    return envelope["data"]
