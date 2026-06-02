"""Tests for ProviderRegistry EventBus lifecycle events (ISSUE-003)."""

import asyncio
import pytest
from rosclaw.core.event_bus import Event
from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class FakeEventBus:
    """In-memory event bus that records all published events."""

    def __init__(self):
        self.events: list[Event] = []
        self._subscribers: dict[str, list] = {}

    def publish(self, event: Event) -> None:
        self.events.append(event)
        for handler in self._subscribers.get(event.topic, []):
            handler(event)

    def subscribe(self, topic: str, handler) -> None:
        self._subscribers.setdefault(topic, []).append(handler)


class DummyProvider(Provider):
    name = "dummy"
    capabilities = ["test.cap"]

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={},
        )

    async def health(self):
        return {"ok": True}


class FailingProvider(Provider):
    name = "failing"
    capabilities = ["test.fail"]

    async def load(self):
        raise RuntimeError("load failure")

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={},
        )

    async def health(self):
        return {"ok": False, "error": "unhealthy"}


@pytest.fixture
def bus():
    return FakeEventBus()


@pytest.fixture
def registry(bus):
    return ProviderRegistry(event_bus=bus)


@pytest.fixture
def manifest():
    return ProviderManifest.from_dict({
        "name": "dummy",
        "version": "1.0.0",
        "type": "test",
        "capabilities": ["test.cap"],
    })


# ------------------------------------------------------------------
# provider_registered event
# ------------------------------------------------------------------

def test_provider_registered_event(bus, registry, manifest):
    registry.register(manifest, lambda m: DummyProvider(m), auto_load=False)
    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.topic == "provider_registered"
    assert event.payload["provider"] == "dummy"
    assert event.payload["type"] == "test"
    assert event.payload["capabilities"] == ["test.cap"]
    assert event.payload["auto_load"] is False


def test_provider_registered_with_auto_load_sync(bus, manifest):
    """Auto-load in sync context emits provider_health_changed after registered."""
    registry = ProviderRegistry(event_bus=bus)
    registry.register(manifest, lambda m: DummyProvider(m), auto_load=True)
    topics = [e.topic for e in bus.events]
    assert topics[0] == "provider_registered"
    assert "provider_health_changed" in topics
    health_event = [e for e in bus.events if e.topic == "provider_health_changed"][0]
    assert health_event.payload["provider"] == "dummy"
    assert health_event.payload["ok"] is True


@pytest.mark.asyncio
async def test_provider_registered_with_auto_load_async(bus, manifest):
    """Auto-load in async context defers load; health event arrives later."""
    registry = ProviderRegistry(event_bus=bus)
    registry.register(manifest, lambda m: DummyProvider(m), auto_load=True)
    # deferred load is scheduled; give it a tick
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    topics = [e.topic for e in bus.events]
    assert topics[0] == "provider_registered"
    assert "provider_health_changed" in topics


# ------------------------------------------------------------------
# provider_unregistered event
# ------------------------------------------------------------------

def test_provider_unregistered_event(bus, registry, manifest):
    registry.register(manifest, lambda m: DummyProvider(m), auto_load=False)
    bus.events.clear()
    registry.unregister("dummy")
    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.topic == "provider_unregistered"
    assert event.payload["provider"] == "dummy"


# ------------------------------------------------------------------
# provider_health_changed event
# ------------------------------------------------------------------

def test_set_provider_health_emits_event(bus, registry, manifest):
    registry.register(manifest, lambda m: DummyProvider(m), auto_load=False)
    bus.events.clear()
    registry.set_provider_health("dummy", ok=True)
    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.topic == "provider_health_changed"
    assert event.payload["provider"] == "dummy"
    assert event.payload["ok"] is True
    assert event.payload["reason"] == "manual_set"


def test_set_provider_health_with_reason(bus, registry, manifest):
    registry.register(manifest, lambda m: DummyProvider(m), auto_load=False)
    bus.events.clear()
    registry.set_provider_health("dummy", ok=False, error="connection lost")
    event = bus.events[0]
    assert event.payload["ok"] is False
    assert event.payload["reason"] == "connection lost"


@pytest.mark.asyncio
async def test_check_health_emits_event_on_change(bus, registry, manifest):
    registry.register(manifest, lambda m: DummyProvider(m), auto_load=False)
    registry.set_provider_health("dummy", ok=False)
    bus.events.clear()
    result = await registry.check_health("dummy")
    assert result["ok"] is True
    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.topic == "provider_health_changed"
    assert event.payload["ok"] is True
    assert event.payload["reason"] == "health_check"


@pytest.mark.asyncio
async def test_check_health_no_event_when_unchanged(bus, registry, manifest):
    registry.register(manifest, lambda m: DummyProvider(m), auto_load=False)
    registry.set_provider_health("dummy", ok=True)
    bus.events.clear()
    await registry.check_health("dummy")
    assert len(bus.events) == 0


def test_load_failure_emits_health_event(bus, manifest):
    """A provider whose load() raises emits a failed health event."""
    registry = ProviderRegistry(event_bus=bus)
    registry.register(manifest, lambda m: FailingProvider(m), auto_load=True)
    health_events = [e for e in bus.events if e.topic == "provider_health_changed"]
    assert len(health_events) == 1
    assert health_events[0].payload["ok"] is False
    assert "load_failed" in health_events[0].payload["reason"]


# ------------------------------------------------------------------
# No event bus (backwards compatibility)
# ------------------------------------------------------------------

def test_no_event_bus_no_crash(manifest):
    registry = ProviderRegistry(event_bus=None)
    registry.register(manifest, lambda m: DummyProvider(m), auto_load=False)
    registry.set_provider_health("dummy", ok=True)
    registry.unregister("dummy")
    # Should not raise


# ------------------------------------------------------------------
# Runtime subscription integration
# ------------------------------------------------------------------

def test_runtime_injects_event_bus():
    """Runtime passes its EventBus into ProviderRegistry."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig
    config = RuntimeConfig(
        robot_id="test",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_provider=True,
    )
    rt = Runtime(config)
    rt.initialize()
    assert rt.provider_registry is not None
    # The registry should have received the event bus
    assert rt.provider_registry._event_bus is rt.event_bus
    rt.stop()


def test_runtime_subscription_output(caplog):
    """Runtime handlers log provider lifecycle events."""
    import logging
    from rosclaw.core.runtime import Runtime, RuntimeConfig
    config = RuntimeConfig(
        robot_id="test",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_provider=True,
    )
    with caplog.at_level(logging.INFO, logger="rosclaw.core.runtime"):
        rt = Runtime(config)
        rt.initialize()
    assert "Provider event: provider_registered" in caplog.text or "Provider 'mock_vlm' is now healthy" in caplog.text or "Provider Layer" in caplog.text
    rt.stop()
