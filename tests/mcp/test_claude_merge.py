"""Tests for merging Hardware MCP fragments into .mcp.json."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.mcp.onboarding.claude_merge import ClaudeMergeError, ClaudeMcpMerge, MANAGED_KEY


@pytest.fixture
def merger(project_root: Path) -> ClaudeMcpMerge:
    return ClaudeMcpMerge(project_root=project_root)


@pytest.fixture
def unitree_fragment() -> dict:
    return {
        "mcpServers": {
            "rosclaw-unitree-g1": {
                "type": "stdio",
                "command": "rosclaw-mcp-run",
                "args": ["unitree-g1"],
            }
        }
    }


def test_merge_creates_new_file(merger: ClaudeMcpMerge, unitree_fragment: dict) -> None:
    result = merger.merge(
        server_name="rosclaw-unitree-g1",
        manifest_id="io.rosclaw.hardware.unitree-g1",
        version="1.0.0",
        mcp_json_fragment=unitree_fragment,
        dry_run=False,
    )
    assert result.action == "created"
    assert merger.mcp_json_path.exists()
    data = merger.load()
    assert "rosclaw-unitree-g1" in data["mcpServers"]
    assert data["mcpServers"]["rosclaw-unitree-g1"][MANAGED_KEY]["manifest_id"] == "io.rosclaw.hardware.unitree-g1"


def test_merge_upgrades_managed_server(merger: ClaudeMcpMerge, unitree_fragment: dict) -> None:
    merger.merge(
        server_name="rosclaw-unitree-g1",
        manifest_id="io.rosclaw.hardware.unitree-g1",
        version="1.0.0",
        mcp_json_fragment=unitree_fragment,
        dry_run=False,
    )
    result = merger.merge(
        server_name="rosclaw-unitree-g1",
        manifest_id="io.rosclaw.hardware.unitree-g1",
        version="1.0.1",
        mcp_json_fragment=unitree_fragment,
        dry_run=False,
    )
    assert result.action == "upgraded"


def test_merge_aborts_on_unmanaged_conflict(
    claude_mcp_with_existing: Path,
    merger: ClaudeMcpMerge,
    unitree_fragment: dict,
) -> None:
    # existing-server conflicts if we try to write a managed server with the same name.
    with pytest.raises(ClaudeMergeError):
        merger.merge(
            server_name="existing-server",
            manifest_id="io.rosclaw.hardware.unitree-g1",
            version="1.0.0",
            mcp_json_fragment={
                "mcpServers": {
                    "existing-server": {
                        "type": "stdio",
                        "command": "new-cmd",
                    }
                }
            },
            conflict="abort",
            dry_run=False,
        )


def test_merge_rename_unmanaged_conflict(
    claude_mcp_conflict: Path,
    merger: ClaudeMcpMerge,
    unitree_fragment: dict,
) -> None:
    result = merger.merge(
        server_name="rosclaw-unitree-g1",
        manifest_id="io.rosclaw.hardware.unitree-g1",
        version="1.0.0",
        mcp_json_fragment=unitree_fragment,
        conflict="rename",
        dry_run=False,
    )
    assert result.action == "renamed"
    data = merger.load()
    assert "rosclaw-unitree-g1-unmanaged" in data["mcpServers"]
    assert "rosclaw-unitree-g1" in data["mcpServers"]


def test_merge_replace_unmanaged_conflict(
    claude_mcp_conflict: Path,
    merger: ClaudeMcpMerge,
    unitree_fragment: dict,
) -> None:
    result = merger.merge(
        server_name="rosclaw-unitree-g1",
        manifest_id="io.rosclaw.hardware.unitree-g1",
        version="1.0.0",
        mcp_json_fragment=unitree_fragment,
        conflict="replace",
        dry_run=False,
    )
    assert result.action == "replaced"
    data = merger.load()
    managed = data["mcpServers"]["rosclaw-unitree-g1"]
    assert managed[MANAGED_KEY]
    assert managed["command"] == "rosclaw-mcp-run"


def test_dry_run_does_not_write_file(merger: ClaudeMcpMerge, unitree_fragment: dict) -> None:
    result = merger.merge(
        server_name="rosclaw-unitree-g1",
        manifest_id="io.rosclaw.hardware.unitree-g1",
        version="1.0.0",
        mcp_json_fragment=unitree_fragment,
        dry_run=True,
    )
    assert result.action == "dry-run:created"
    assert not merger.mcp_json_path.exists()


def test_remove_managed_server(merger: ClaudeMcpMerge, unitree_fragment: dict) -> None:
    merger.merge(
        server_name="rosclaw-unitree-g1",
        manifest_id="io.rosclaw.hardware.unitree-g1",
        version="1.0.0",
        mcp_json_fragment=unitree_fragment,
        dry_run=False,
    )
    result = merger.remove_managed_server("rosclaw-unitree-g1")
    assert result.action == "removed"
    assert "rosclaw-unitree-g1" not in merger.load().get("mcpServers", {})


def test_list_managed_servers(merger: ClaudeMcpMerge, unitree_fragment: dict) -> None:
    merger.merge(
        server_name="rosclaw-unitree-g1",
        manifest_id="io.rosclaw.hardware.unitree-g1",
        version="1.0.0",
        mcp_json_fragment=unitree_fragment,
        dry_run=False,
    )
    managed = merger.list_managed_servers()
    assert "rosclaw-unitree-g1" in managed


def test_remove_unmanaged_server_raises(merger: ClaudeMcpMerge, claude_mcp_with_existing: Path) -> None:
    with pytest.raises(ClaudeMergeError):
        merger.remove_managed_server("existing-server")
