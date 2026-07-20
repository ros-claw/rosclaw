"""MCP black-box coverage for the first verified product workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.mcp.tools import (
    explain_execution,
    get_execution_receipt,
    get_product_status,
    list_product_demos,
    run_product_demo,
)


def _payload(raw: str) -> dict:
    payload = json.loads(raw)
    assert payload["schema_version"] == "rosclaw.mcp.v1"
    return payload


async def test_product_status_and_demo_catalog_are_discoverable() -> None:
    status = _payload(await get_product_status())
    assert status["ok"] is True
    assert status["data"]["product_status"]["release"]["version"] == "1.0.1"
    assert status["usable_for_real_execution"] is False

    demos = _payload(await list_product_demos())
    assert demos["ok"] is True
    assert demos["data"]["demos"][0]["id"] == "ur5e-reach"
    assert demos["data"]["demos"][0]["mode"] == "SIMULATION"


async def test_agent_can_run_demo_and_inspect_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))

    result = _payload(await run_product_demo())
    assert result["ok"] is True
    receipt = result["data"]["receipt"]
    assert receipt["final_state"] == "COMPLETED"
    assert receipt["evidence_level"] == "TASK_VERIFIED"
    assert receipt["simulation_result"]["has_physics"] is True
    assert result["usable_for_real_execution"] is False

    stored = _payload(await get_execution_receipt())
    assert stored["ok"] is True
    assert stored["data"]["receipt"]["action_id"] == receipt["action_id"]

    explanation = _payload(await explain_execution())
    assert explanation["ok"] is True
    assert explanation["data"]["explanation"]["verification"]["task_verified"] is True
    assert explanation["data"]["explanation"]["run_id"] == receipt["action_id"]


async def test_product_demo_rejects_unbounded_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))

    result = _payload(await run_product_demo(max_steps=5001))

    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_ARGUMENT"
    assert not (tmp_path / "runs").exists()

    result = _payload(await run_product_demo(target=[float("nan"), 0.0, 0.5]))
    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_ARGUMENT"
    assert not (tmp_path / "runs").exists()

    result = _payload(await run_product_demo(target=[True, 0.0, 0.5]))
    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_ARGUMENT"
    assert not (tmp_path / "runs").exists()
