"""Additional coverage tests for provider/core/registry.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rosclaw.provider.core.errors import ProviderNotFoundError
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.registry import ProviderRegistry


def _make_manifest(name: str = "test_provider", type_: str = "llm", capabilities=None):
    m = MagicMock(spec=ProviderManifest)
    m.name = name
    m.type = type_
    m.capabilities = capabilities or ["chat"]
    m.extra = {}
    m.safety = MagicMock(executable=False)
    return m


def _make_provider(name: str = "test_provider", capabilities=None, healthy: bool = True):
    p = MagicMock()
    p.name = name
    p.capabilities = capabilities or ["chat"]
    p._healthy = healthy
    p.manifest = _make_manifest(name, capabilities=capabilities)
    p.load = AsyncMock()
    p.unload = AsyncMock()
    p.health = AsyncMock(return_value={"ok": healthy})
    return p


class TestProviderRegistryRegister:
    def test_register_duplicate_raises(self):
        reg = ProviderRegistry()
        manifest = _make_manifest("dup")

        def factory(m):
            return _make_provider("dup")

        reg.register(manifest, factory)
        with pytest.raises(ProviderNotFoundError, match="already registered"):
            reg.register(manifest, factory)

    def test_register_without_auto_load(self):
        reg = ProviderRegistry()
        manifest = _make_manifest("no_load")
        provider = _make_provider("no_load", healthy=False)

        def factory(m):
            return provider

        result = reg.register(manifest, factory, auto_load=False)

        assert result is provider
        assert provider._healthy is False  # not loaded


class TestProviderRegistryLookups:
    def test_get_not_found(self):
        reg = ProviderRegistry()
        with pytest.raises(ProviderNotFoundError, match="not found"):
            reg.get("missing")

    def test_get_manifest_not_found(self):
        reg = ProviderRegistry()
        with pytest.raises(ProviderNotFoundError, match="Manifest"):
            reg.get_manifest("missing")

    def test_find_by_capability_healthy_only_false(self):
        reg = ProviderRegistry()
        p1 = _make_provider("p1", capabilities=["chat"], healthy=False)
        p2 = _make_provider("p2", capabilities=["chat"], healthy=True)
        reg._providers = {"p1": p1, "p2": p2}
        reg._health = {"p1": {"ok": False}, "p2": {"ok": True}}

        results = reg.find_by_capability("chat", healthy_only=False)
        assert len(results) == 2

    def test_find_by_capability_healthy_only_true(self):
        reg = ProviderRegistry()
        p1 = _make_provider("p1", capabilities=["chat"], healthy=False)
        p2 = _make_provider("p2", capabilities=["chat"], healthy=True)
        reg._providers = {"p1": p1, "p2": p2}
        reg._health = {"p1": {"ok": False}, "p2": {"ok": True}}

        results = reg.find_by_capability("chat", healthy_only=True)
        assert len(results) == 1
        assert results[0].name == "p2"

    def test_find_by_type(self):
        reg = ProviderRegistry()
        p1 = _make_provider("p1", capabilities=["chat"])
        p1.manifest.type = "llm"
        p2 = _make_provider("p2", capabilities=["vision"])
        p2.manifest.type = "vlm"
        reg._providers = {"p1": p1, "p2": p2}

        results = reg.find_by_type("vlm")
        assert len(results) == 1
        assert results[0].name == "p2"


class TestProviderRegistryHealth:
    def test_set_provider_health(self):
        reg = ProviderRegistry()
        p = _make_provider("p1", healthy=False)
        reg._providers = {"p1": p}

        reg.set_provider_health("p1", True, error="")

        assert p._healthy is True
        assert reg._health["p1"]["ok"] is True

    def test_set_provider_health_with_error(self):
        reg = ProviderRegistry()
        p = _make_provider("p1")
        reg._providers = {"p1": p}

        reg.set_provider_health("p1", False, error="timeout")

        assert p._healthy is False
        assert p._load_error == "timeout"

    def test_set_provider_health_missing_provider(self):
        reg = ProviderRegistry()
        # Should not crash even if provider doesn't exist
        reg.set_provider_health("missing", True)

    def test_is_healthy(self):
        reg = ProviderRegistry()
        reg._health = {"p1": {"ok": True}, "p2": {"ok": False}}
        assert reg.is_healthy("p1") is True
        assert reg.is_healthy("p2") is False
        assert reg.is_healthy("missing") is False


class TestProviderRegistryStatistics:
    def test_get_statistics(self):
        reg = ProviderRegistry()
        p1 = _make_provider("p1", healthy=True)
        p1.manifest.type = "llm"
        p2 = _make_provider("p2", healthy=False)
        p2.manifest.type = "llm"
        p3 = _make_provider("p3", healthy=True)
        p3.manifest.type = "vlm"

        reg._providers = {"p1": p1, "p2": p2, "p3": p3}
        reg._manifests = {"p1": p1.manifest, "p2": p2.manifest, "p3": p3.manifest}
        reg._health = {
            "p1": {"ok": True},
            "p2": {"ok": False},
            "p3": {"ok": True},
        }

        stats = reg.get_statistics()
        assert stats["total_providers"] == 3
        assert stats["healthy_providers"] == 2
        assert stats["unhealthy_providers"] == 1
        assert stats["by_type"] == {"llm": 2, "vlm": 1}


class TestProviderRegistryEventBus:
    def test_publish_event_with_bus(self):
        bus = MagicMock()
        bus.publish = MagicMock()
        reg = ProviderRegistry(event_bus=bus)

        reg._publish_event("test.topic", {"key": "val"})

        bus.publish.assert_called_once()
        event = bus.publish.call_args[0][0]
        assert event.topic == "test.topic"

    def test_publish_event_without_bus(self):
        reg = ProviderRegistry(event_bus=None)
        # Should not crash
        reg._publish_event("test.topic", {"key": "val"})


class TestProviderRegistryLifecycle:
    def test_unregister_missing(self):
        reg = ProviderRegistry()
        # Should not crash
        reg.unregister("missing")

    def test_list_providers_empty(self):
        reg = ProviderRegistry()
        assert reg.list_providers() == []

    def test_list_providers(self):
        reg = ProviderRegistry()
        p = _make_provider("p1")
        reg._providers = {"p1": p}
        assert reg.list_providers() == ["p1"]
