"""Tests for ROSClaw Hub MCP configuration merging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rosclaw.hub.errors import HubError
from rosclaw.hub.mcp_merge import McpMerger
from rosclaw.hub.refs import AssetRef
from rosclaw.hub.schema import load_manifest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "hub_assets"
HARDWARE_MCP_FIXTURE = FIXTURES_DIR / "hardware_mcp_valid"


def _hardware_ref() -> AssetRef:
    return AssetRef(
        type="hardware_mcp",
        namespace="rosclaw",
        name="unitree-g1",
        version="1.0.0",
    )


@pytest.fixture
def merger(tmp_path: Path) -> McpMerger:
    """Fresh McpMerger backed by a temporary home and project root."""
    return McpMerger(project_root=tmp_path, home=tmp_path)


@pytest.fixture
def hardware_manifest(tmp_path: Path) -> Any:
    """Loaded hardware_mcp fixture manifest."""
    return load_manifest(HARDWARE_MCP_FIXTURE / "manifest.yaml")


class TestMcpMergerAddServer:
    """Tests for adding managed MCP server entries."""

    def test_add_server_writes_mcp_json_and_fragment(
        self,
        merger: McpMerger,
        hardware_manifest: Any,
    ) -> None:
        """add_server updates .mcp.json and writes a runtime fragment."""
        asset_dir = HARDWARE_MCP_FIXTURE
        server_name = merger.add_server(hardware_manifest, asset_dir)

        assert (merger.project_root / ".mcp.json").exists()
        assert (merger.fragments_dir / f"{server_name}.json").exists()

        data = json.loads((merger.project_root / ".mcp.json").read_text())
        assert server_name in data["servers"]
        entry = data["servers"][server_name]
        assert entry["command"] == "python"
        assert entry["args"] == ["-m", "rosclaw_unitree_g1_mcp.server"]
        assert entry["transport"] == "stdio"
        assert entry["rosclaw"]["ref"] == str(_hardware_ref())

    def test_add_server_is_idempotent(
        self,
        merger: McpMerger,
        hardware_manifest: Any,
    ) -> None:
        """Adding the same asset twice overwrites rather than duplicates."""
        asset_dir = HARDWARE_MCP_FIXTURE
        name1 = merger.add_server(hardware_manifest, asset_dir)
        name2 = merger.add_server(hardware_manifest, asset_dir)
        assert name1 == name2

        data = json.loads((merger.project_root / ".mcp.json").read_text())
        assert len(data["servers"]) == 1


class TestMcpMergerRemoveServer:
    """Tests for removing managed MCP server entries."""

    def test_remove_server_deletes_entry_and_fragment(
        self,
        merger: McpMerger,
        hardware_manifest: Any,
    ) -> None:
        """remove_server cleans both .mcp.json and the fragment file."""
        server_name = merger.add_server(hardware_manifest, HARDWARE_MCP_FIXTURE)
        ref = _hardware_ref()

        removed = merger.remove_server(ref)
        assert removed is True

        data = json.loads((merger.project_root / ".mcp.json").read_text())
        assert server_name not in data["servers"]
        assert not (merger.fragments_dir / f"{server_name}.json").exists()

    def test_remove_server_preserves_unmanaged_servers(
        self,
        merger: McpMerger,
        hardware_manifest: Any,
    ) -> None:
        """Only rosclaw-managed servers are touched."""
        mcp_path = merger.project_root / ".mcp.json"
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_path.write_text(
            json.dumps(
                {
                    "version": "1.0.0",
                    "servers": {
                        "manual-server": {
                            "command": "manual",
                            "args": [],
                            "env": {},
                            "transport": "stdio",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        merger.add_server(hardware_manifest, HARDWARE_MCP_FIXTURE)
        ref = _hardware_ref()
        merger.remove_server(ref)

        data = json.loads(mcp_path.read_text())
        assert "manual-server" in data["servers"]

    def test_remove_missing_server_returns_false(self, merger: McpMerger) -> None:
        """Removing a ref that was never added returns False."""
        removed = merger.remove_server(_hardware_ref())
        assert removed is False


class TestMcpMergerQueries:
    """Tests for list_servers and is_managed helpers."""

    def test_list_servers_only_returns_managed(
        self,
        merger: McpMerger,
        hardware_manifest: Any,
    ) -> None:
        """list_servers filters out entries without rosclaw metadata."""
        mcp_path = merger.project_root / ".mcp.json"
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_path.write_text(
            json.dumps(
                {
                    "version": "1.0.0",
                    "servers": {
                        "manual-server": {"command": "manual"},
                        "managed-server": {"command": "managed", "rosclaw": {"ref": "x"}},
                    },
                }
            ),
            encoding="utf-8",
        )

        managed = merger.list_servers()
        assert "manual-server" not in managed
        assert "managed-server" in managed

    def test_is_managed_true_after_add(
        self,
        merger: McpMerger,
        hardware_manifest: Any,
    ) -> None:
        """is_managed reflects an added server."""
        ref = _hardware_ref()
        assert merger.is_managed(ref) is False
        merger.add_server(hardware_manifest, HARDWARE_MCP_FIXTURE)
        assert merger.is_managed(ref) is True


class TestMcpMergerErrors:
    """Error handling for malformed MCP configuration."""

    def test_corrupt_mcp_json_raises_hub_error(self, merger: McpMerger) -> None:
        """A non-JSON .mcp.json file is reported as a HubError."""
        mcp_path = merger.project_root / ".mcp.json"
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_path.write_text("not json", encoding="utf-8")

        with pytest.raises(HubError) as exc_info:
            merger.list_servers()
        assert "Corrupt MCP config" in exc_info.value.message

    def test_missing_command_raises_hub_error(
        self,
        merger: McpMerger,
        tmp_path: Path,
    ) -> None:
        """An MCP entrypoint without a command is rejected."""
        manifest = load_manifest(HARDWARE_MCP_FIXTURE / "manifest.yaml")
        manifest.install["entrypoints"]["mcp"]["command"] = ""

        with pytest.raises(HubError) as exc_info:
            merger.add_server(manifest, tmp_path)
        assert "missing a command" in exc_info.value.message
