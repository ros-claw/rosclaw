"""Tests for the P0 MCP server wiring and audit helpers."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from rosclaw.agent.tool_catalog import P0_AGENT_MCP_TOOLS
from rosclaw.mcp.tools import P0_TOOLS, _audit, _redact_for_audit


async def test_p0_tools_contains_expected_set() -> None:
    """P0_TOOLS contains core, body-context, and product workflow tools."""
    names = tuple(tool.__name__ for tool in P0_TOOLS)
    assert names == P0_AGENT_MCP_TOOLS


async def test_redact_for_audit_masks_secrets() -> None:
    arguments = {
        "api_key": "super-secret",
        "instruction": "pick cup",
        "password": "hunter2",
        "approval_id": "permit-live-1",
        "arguments": {
            "vendor_token": "nested-token",
            "targets": [{"private_key": "nested-key", "joint": 1}],
        },
    }
    safe = _redact_for_audit(arguments)
    assert safe["api_key"] == "<REDACTED>"
    assert safe["password"] == "<REDACTED>"
    assert safe["approval_id"] == "<REDACTED>"
    assert safe["arguments"]["vendor_token"] == "<REDACTED>"
    assert safe["arguments"]["targets"][0]["private_key"] == "<REDACTED>"
    assert safe["arguments"]["targets"][0]["joint"] == 1
    assert safe["instruction"] == "pick cup"


async def test_audit_log_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_dir = tmp_path / ".rosclaw/logs/mcp"
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / ".rosclaw"))
    response = {"ok": True, "data": {"robot_id": "test"}}
    _audit("trace-123", "get_robot_state", {}, response, 12.3)
    audit_file = log_dir / "audit.jsonl"
    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["trace_id"] == "trace-123"
    assert entry["tool"] == "get_robot_state"
    assert entry["ok"] is True
    assert stat.S_IMODE(log_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(audit_file.stat().st_mode) == 0o600
