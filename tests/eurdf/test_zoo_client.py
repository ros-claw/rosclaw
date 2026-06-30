"""Tests for ``rosclaw.eurdf.zoo_client``."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.eurdf.zoo_client import (
    E_URDF_ZOO_AVAILABLE,
    EurdfZooClient,
    EurdfZooClientError,
)

pytestmark = pytest.mark.skipif(
    not E_URDF_ZOO_AVAILABLE,
    reason="e_urdf_zoo package is not installed",
)


@pytest.fixture
def zoo_path() -> Path:
    """Path to the project-root e-URDF-Zoo robots directory."""
    return Path(__file__).parent.parent.parent / "e-urdf-zoo" / "robots"


@pytest.fixture
def client(zoo_path: Path, tmp_path: Path) -> EurdfZooClient:
    """Configured client with isolated cache."""
    return EurdfZooClient(zoo_path=zoo_path, cache_dir=tmp_path / "cache")


def test_client_lists_dexhands(client: EurdfZooClient) -> None:
    assets = client.list_assets(category="dexhands")
    ids = [a.id for a in assets]
    assert "dexhands/inspire_hand/right" in ids
    assert "dexhands/ability_hand/left" in ids


def test_client_search_panda(client: EurdfZooClient) -> None:
    results = client.search_assets("panda")
    ids = [a.id for a in results]
    assert "grippers/panda/default" in ids


def test_client_load_manifest_asset(client: EurdfZooClient) -> None:
    asset = client.load("dexhands/inspire_hand/right")
    assert asset.is_manifest
    assert asset.name == "Inspire Hand (Right)"
    assert asset.manifest is not None
    assert asset.manifest.runtime_policy.real_robot_execution_allowed is False


def test_client_get_profile(client: EurdfZooClient) -> None:
    profile = client.get_profile("dexhands/inspire_hand/right")
    assert profile.robot_id == "dexhands/inspire_hand/right"
    assert profile.vendor == "Inspire Robots"
    assert profile.embodiment.dof > 0
    assert len(profile.embodiment.joints) > 0
    assert profile.safety.safety_level == "STRICT"
    forbidden = profile.capability.skill_registry.get("forbidden_capabilities", [])
    assert any(fc["id"] == "fast_full_close" for fc in forbidden)


def test_client_get_eurdf_profile(client: EurdfZooClient) -> None:
    eurdf = client.get_eurdf_profile("dexhands/inspire_hand/right")
    assert eurdf.profile_id == "dexhands/inspire_hand/right"
    assert eurdf.safety.get("safety_level") == "STRICT"
    assert "all" in eurdf.capability_hints


def test_client_pull_and_load_from_cache(client: EurdfZooClient) -> None:
    cached = client.pull("dexhands/inspire_hand/right")
    assert cached.exists()
    assert (cached / "manifest.yaml").exists()

    asset = client.load("dexhands/inspire_hand/right")
    assert asset.is_manifest
    assert "Inspire" in asset.name


def test_client_cache_list(client: EurdfZooClient) -> None:
    client.pull("dexhands/inspire_hand/right")
    entries = client.cache_list()
    assert any(e["asset_id"] == "dexhands/inspire_hand/right" for e in entries)


def test_client_validate(client: EurdfZooClient) -> None:
    report = client.validate("dexhands/inspire_hand/right")
    assert report["asset_id"] == "dexhands/inspire_hand/right"
    assert report["overall"] != "FAIL"


def test_client_missing_asset(client: EurdfZooClient) -> None:
    with pytest.raises(EurdfZooClientError):
        client.load("does/not/exist")
