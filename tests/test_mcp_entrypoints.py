"""Regression tests for MCP server entry points documented in README.

Ensures `python3 -m rosclaw.mcp.minimal_server` and `python3 -m
rosclaw.mcp.ur5_server` can be imported without error.
"""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "module_name",
    [
        "rosclaw.mcp.minimal_server",
        "rosclaw.mcp.ur5_server",
    ],
)
def test_mcp_module_imports(module_name: str) -> None:
    """The module can be imported without error."""
    env = {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}; print('ok')"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Failed to import {module_name}: {result.stderr}"
    )


def test_no_fictitious_mcp_server_module() -> None:
    """README no longer references the non-existent rosclaw.mcp.server."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    zh = (PROJECT_ROOT / "README.zh.md").read_text(encoding="utf-8")
    for text, name in [(readme, "README.md"), (zh, "README.zh.md")]:
        assert "rosclaw.mcp.server" not in text, (
            f"{name} still references non-existent rosclaw.mcp.server"
        )
