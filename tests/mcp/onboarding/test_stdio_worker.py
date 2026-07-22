"""MCP Adapter subprocess isolation and timeout regression tests."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from rosclaw.mcp.onboarding.stdio_client import McpStdioClient, McpStdioError

FIXTURE = Path(__file__).parents[2] / "fixtures" / "mcp_stdio_server.py"


def _client(mode: str) -> McpStdioClient:
    return McpStdioClient(sys.executable, [str(FIXTURE), "--mode", mode])


def test_mcp_worker_handshake_list_and_call() -> None:
    client = _client("healthy")
    try:
        client.start(timeout=1.0)
        assert client.list_tools(timeout=1.0) == [{"name": "capture_aligned_rgbd"}]
        result = client.call_tool("capture_aligned_rgbd", {}, timeout=1.0)
    finally:
        client.stop(timeout=0.2)

    assert result["result"]["content"][0]["text"] == "ok"


@pytest.mark.parametrize("mode", ["stall", "crash", "oversize", "malformed", "wrong-id"])
def test_mcp_worker_fault_is_bounded_and_process_is_terminated(mode: str) -> None:
    client = _client(mode)
    client.start(timeout=1.0)
    process = client._proc
    assert process is not None
    started = time.monotonic()
    try:
        with pytest.raises(McpStdioError):
            client.call_tool("capture_aligned_rgbd", {}, timeout=0.15)
    finally:
        client.stop(timeout=0.1)

    assert time.monotonic() - started < 1.0
    assert process.poll() is not None
