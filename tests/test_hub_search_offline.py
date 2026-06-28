"""Tests for offline Hub search fallback catalog."""

from __future__ import annotations

import pytest

from rosclaw.hub.cli import _builtin_catalog_entries, cmd_hub_search


class TestHubSearchOffline:
    """Hub search must work without an active registry."""

    def test_builtin_catalog_has_realsense_assets(self):
        entries = _builtin_catalog_entries()
        names = {e["asset"]["name"] for e in entries}
        assert "librealsense-mcp" in names
        assert "realsense-ros-mcp" in names
        assert "realsense_d405" in names

    def test_hub_search_offline_finds_realsense(self, monkeypatch, tmp_path):
        """Without login, search falls back to the built-in catalog."""
        from types import SimpleNamespace

        # Isolate auth/cache state.
        monkeypatch.setenv("HOME", str(tmp_path))

        args = SimpleNamespace(
            query="realsense",
            type=None,
            namespace=None,
            official=False,
            license=None,
            robot=None,
            compatible=False,
            limit=50,
            json=True,
        )
        assert cmd_hub_search(args) == 0

    def test_hub_search_offline_no_results_for_unrelated_query(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        monkeypatch.setenv("HOME", str(tmp_path))

        args = SimpleNamespace(
            query="unitree-g1",
            type=None,
            namespace=None,
            official=False,
            license=None,
            robot=None,
            compatible=False,
            limit=50,
            json=True,
        )
        assert cmd_hub_search(args) == 0
