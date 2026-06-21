"""Tests for the ROSClaw Hub assets lockfile."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.hub.errors import HubError
from rosclaw.hub.lockfile import (
    AssetsLock,
    LockEntry,
    acquire_assets_lock,
)
from rosclaw.hub.refs import AssetRef


def test_load_empty_lockfile(tmp_path: Path) -> None:
    """Loading a missing lockfile returns an empty instance."""
    lock = AssetsLock.load(tmp_path / "assets.lock")
    assert lock.is_installed("rosclaw://skill/rosclaw/foo@1.0.0") is False
    assert list(lock) == []


def test_add_and_save_roundtrip(tmp_path: Path) -> None:
    """Entries survive a save/load roundtrip."""
    path = tmp_path / "assets.lock"
    lock = AssetsLock(path=path)
    entry = LockEntry(
        ref="rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0",
        source="./fixtures/hub_assets/hardware_mcp_valid",
        asset_dir=str(tmp_path / "installed" / "unitree-g1"),
        depends_on=["rosclaw://provider/rosclaw/g1-driver@1.0.0"],
    )
    lock.add(entry)
    lock.save()

    loaded = AssetsLock.load(path)
    assert loaded.is_installed(entry.ref)
    found = loaded.get(entry.ref)
    assert found is not None
    assert found.source == entry.source
    assert found.asset_dir == entry.asset_dir
    assert found.depends_on == entry.depends_on


def test_remove_existing_and_missing(tmp_path: Path) -> None:
    """Removing returns True only when the entry existed."""
    lock = AssetsLock(path=tmp_path / "assets.lock")
    ref = "rosclaw://skill/rosclaw/pick@1.0.0"
    lock.add(LockEntry(ref=ref, source="src", asset_dir="dst"))
    assert lock.remove(ref) is True
    assert lock.remove(ref) is False


def test_list_installed_is_sorted(tmp_path: Path) -> None:
    """list_installed returns entries ordered by ref."""
    lock = AssetsLock(path=tmp_path / "assets.lock")
    refs = [
        "rosclaw://skill/rosclaw/zebra@1.0.0",
        "rosclaw://skill/rosclaw/alpha@1.0.0",
        "rosclaw://skill/rosclaw/mid@2.0.0",
    ]
    for ref in refs:
        lock.add(LockEntry(ref=ref, source="s", asset_dir="d"))
    assert [e.ref for e in lock.list_installed()] == sorted(refs)


def test_add_from_dict(tmp_path: Path) -> None:
    """add() accepts a serialized dict."""
    lock = AssetsLock(path=tmp_path / "assets.lock")
    lock.add(
        {
            "ref": "rosclaw://digital_twin/rosclaw/world@0.1.0",
            "source": "catalog",
            "asset_dir": "/tmp/dt",
            "lifecycle_status": "installed",
            "health_status": "healthy",
        }
    )
    entry = lock.get("rosclaw://digital_twin/rosclaw/world@0.1.0")
    assert entry is not None
    assert entry.health_status == "healthy"


def test_asset_ref_helper() -> None:
    """LockEntry.asset_ref() parses the stored canonical reference."""
    entry = LockEntry(
        ref="rosclaw://provider/rosclaw/llm@3.0.0",
        source="s",
        asset_dir="d",
    )
    assert entry.asset_ref() == AssetRef(
        type="provider",
        namespace="rosclaw",
        name="llm",
        version="3.0.0",
    )


def test_corrupt_lockfile_raises(tmp_path: Path) -> None:
    """Loading invalid JSON raises a HubError."""
    path = tmp_path / "assets.lock"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(HubError):
        AssetsLock.load(path)


def test_acquire_assets_lock(tmp_path: Path) -> None:
    """acquire_assets_lock can be entered and protects a block."""
    lock_file = tmp_path / "assets.lock"
    with acquire_assets_lock(lock_file):
        # While the lock is held the underlying lock file exists.
        assert lock_file.exists()
