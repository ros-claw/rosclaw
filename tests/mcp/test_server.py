"""Tests for the P0 MCP server wiring and audit helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.mcp.server import _audit, _redact_for_audit
from rosclaw.mcp.tools import P0_TOOLS


async def test_p0_tools_contains_expected_set() -> None:
    """P0_TOOLS must contain the original seven tools plus the six body tools."""
    names = {t.__name__ for t in P0_TOOLS}
    expected = {
        "get_robot_state",
        "list_skills",
        "query_memory",
        "validate_trajectory",
        "sandbox_run",
        "practice_query",
        "emergency_stop",
        "list_bodies",
        "get_body",
        "switch_body",
        "list_body_history",
        "check_skill_compatibility",
        "fleet_skill_compatibility",
    }
    assert names == expected


async def test_redact_for_audit_masks_secrets() -> None:
    arguments = {
        "api_key": "super-secret",
        "instruction": "pick cup",
        "password": "hunter2",
    }
    safe = _redact_for_audit(arguments)
    assert safe["api_key"] == "<REDACTED>"
    assert safe["password"] == "<REDACTED>"
    assert safe["instruction"] == "pick cup"


async def test_audit_log_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_dir = tmp_path / ".rosclaw/logs/mcp"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    response = {"ok": True, "data": {"robot_id": "test"}}
    _audit("trace-123", "get_robot_state", {}, response)
    audit_file = log_dir / "audit.jsonl"
    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["trace_id"] == "trace-123"
    assert entry["tool"] == "get_robot_state"
    assert entry["ok"] is True
