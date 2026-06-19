"""Tests for ROSClaw Hub SQLite catalog index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.hub.cache import HubCache
from rosclaw.hub.index import CatalogIndex


@pytest.fixture
def catalog_entries():
    """Load the fake registry catalog entries."""
    catalog_path = Path(__file__).parents[1] / "fixtures" / "fake_registry" / "catalog.jsonl"
    entries = []
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


@pytest.fixture
def index(tmp_path, monkeypatch) -> CatalogIndex:
    """Create an empty catalog index in a temporary home."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    return CatalogIndex("http://localhost:8787", HubCache())


def test_index_creates_database(index: CatalogIndex) -> None:
    """The index database file is created on initialization."""
    assert index.db_path.exists()


def test_index_entries_and_count(index: CatalogIndex, catalog_entries) -> None:
    """Entries are inserted and counted correctly."""
    indexed = index.index_entries(catalog_entries)
    assert indexed == len(catalog_entries)
    assert index.count() == len(catalog_entries)


def test_search_by_keyword(index: CatalogIndex, catalog_entries) -> None:
    """Full-text search finds assets by title, summary, tags, etc."""
    index.index_entries(catalog_entries)
    results = index.search("g1")
    names = {r["asset"]["name"] for r in results}
    assert "unitree-g1" in names or "g1-mujoco-basic" in names or "g1-pick-place" in names


def test_search_no_results(index: CatalogIndex, catalog_entries) -> None:
    """Search returns an empty list when nothing matches."""
    index.index_entries(catalog_entries)
    results = index.search("zzzzzz-not-present")
    assert results == []


def test_search_filter_by_type(index: CatalogIndex, catalog_entries) -> None:
    """Structured filters restrict results by asset type."""
    index.index_entries(catalog_entries)
    results = index.search(asset_type="skill")
    assert len(results) == 1
    assert results[0]["asset"]["name"] == "g1-pick-place"


def test_search_filter_official(index: CatalogIndex, catalog_entries) -> None:
    """Official filter selects only official publishers."""
    index.index_entries(catalog_entries)
    results = index.search("", official=True)
    assert len(results) == len(catalog_entries)
    assert all(r["publisher"]["trust_level"] == "official" for r in results)


def test_search_filter_license(index: CatalogIndex, catalog_entries) -> None:
    """License filter restricts by SPDX identifier."""
    index.index_entries(catalog_entries)
    results = index.search(licenses=["CC-BY-SA-4.0"])
    assert len(results) == 1
    assert results[0]["asset"]["name"] == "humanoid-locomotion-patterns"


def test_search_filter_robot(index: CatalogIndex, catalog_entries) -> None:
    """Robot filter matches body kinds and eurdf profiles."""
    index.index_entries(catalog_entries)
    results = index.search(robot="humanoid")
    names = {r["asset"]["name"] for r in results}
    assert "unitree-g1" in names
    assert "g1-pick-place" in names


def test_search_compatible(index: CatalogIndex, catalog_entries) -> None:
    """Compatible filter matches the current platform."""
    index.index_entries(catalog_entries)
    results = index.search(compatible=True)
    # The test runner is Linux x86_64, so all linux-compatible assets match.
    assert len(results) >= 1
    for r in results:
        assert "linux" in json.dumps(r["compatibility"]["os"])
        assert "x86_64" in json.dumps(r["compatibility"]["arch"])


def test_get_by_ref(index: CatalogIndex, catalog_entries) -> None:
    """A specific entry can be retrieved by canonical id."""
    index.index_entries(catalog_entries)
    result = index.get("hardware_mcp:rosclaw:unitree-g1:1.0.0")
    assert result is not None
    assert result["asset"]["name"] == "unitree-g1"


def test_get_missing(index: CatalogIndex, catalog_entries) -> None:
    """Missing entries return None."""
    index.index_entries(catalog_entries)
    assert index.get("skill:rosclaw:missing:9.9.9") is None


def test_clear(index: CatalogIndex, catalog_entries) -> None:
    """Clear removes every indexed entry."""
    index.index_entries(catalog_entries)
    assert index.count() == len(catalog_entries)
    index.clear()
    assert index.count() == 0
