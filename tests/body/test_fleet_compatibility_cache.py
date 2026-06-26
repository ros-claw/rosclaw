"""Tests for FleetCompatibilityCache."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.fleet_cache import FleetCompatibilityCache
from rosclaw.body.service import BodyInstanceService


@pytest.fixture
def multi_body_workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService(workspace=tmp_path).create_or_init(
        robot="unitree-g1", name="g1-a", mode="registry", update_registry=True, switch_active=True
    )
    BodyInstanceService(workspace=tmp_path).create_or_init(
        robot="unitree-g1", name="g1-b", mode="registry", update_registry=True
    )
    return tmp_path


def test_cache_returns_same_report_on_second_call(multi_body_workspace: Path):
    cache = FleetCompatibilityCache(multi_body_workspace, ttl_sec=300)
    report1 = cache.get_or_compute()
    report2 = cache.get_or_compute()

    assert report1 is report2
    assert cache.stats()["hits"] >= 1


def test_cache_invalidates_on_body_change(multi_body_workspace: Path):
    cache = FleetCompatibilityCache(multi_body_workspace, ttl_sec=300)
    report1 = cache.get_or_compute()

    cache.on_body_changed("g1-a")
    report2 = cache.get_or_compute()

    assert report1 is not report2


def test_cache_invalidates_on_active_body_switched(multi_body_workspace: Path):
    cache = FleetCompatibilityCache(multi_body_workspace, ttl_sec=300)
    report1 = cache.get_or_compute()

    cache.on_active_body_switched()
    report2 = cache.get_or_compute()

    assert report1 is not report2


def test_cache_invalidates_on_skill_manifest_change(multi_body_workspace: Path):
    cache = FleetCompatibilityCache(multi_body_workspace, ttl_sec=300)
    report1 = cache.get_or_compute()

    cache.on_skill_manifest_changed()
    report2 = cache.get_or_compute()

    assert report1 is not report2


def test_cache_respects_ttl(multi_body_workspace: Path, monkeypatch):
    cache = FleetCompatibilityCache(multi_body_workspace, ttl_sec=0)
    report1 = cache.get_or_compute()

    # Even with identical state, TTL=0 means immediate expiry.
    report2 = cache.get_or_compute()
    assert report1 is not report2


def test_cache_stats(multi_body_workspace: Path):
    cache = FleetCompatibilityCache(multi_body_workspace, ttl_sec=300)
    cache.get_or_compute()
    cache.get_or_compute()

    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
    assert 0.0 <= stats["hit_rate"] <= 1.0
