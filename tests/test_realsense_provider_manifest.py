"""Tests for RealSense LAN provider manifest."""
from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.provider.adapters.generic import GenericProvider
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.registry import ProviderRegistry


MANIFEST_PATH = Path(__file__).parent.parent / "providers" / "cosmos-reason2-lan" / "provider.yaml"


def test_manifest_loads() -> None:
    manifest = ProviderManifest.from_yaml(MANIFEST_PATH)
    assert manifest.name == "cosmos-reason2-lan"
    assert manifest.type == "vlm"
    assert manifest.runtime.backend == "http"
    assert "192.168.1.105:8009" in manifest.runtime.endpoint
    assert "vlm.object_grounding" in manifest.capabilities
    assert "realsense-d405" in manifest.embodiment.supported_robots


@pytest.mark.asyncio
async def test_provider_load_and_health() -> None:
    manifest = ProviderManifest.from_yaml(MANIFEST_PATH)
    registry = ProviderRegistry()
    provider = registry.register(manifest, GenericProvider, auto_load=False)
    await provider.load()
    health = await provider.health()
    assert health["ok"] is True
    assert provider.name == "cosmos-reason2-lan"
    await provider.unload()
