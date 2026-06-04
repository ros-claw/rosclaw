"""L3: Runtime Plugin tests."""
import pytest
from rosclaw.auto.plugin import AutoPlugin
from rosclaw.auto.config import AutoConfig


class FakeRuntimeContext:
    pass


class FakeEventBus:
    def __init__(self):
        self.subscriptions = {}
        self.published = []
    def subscribe(self, topic, handler):
        self.subscriptions.setdefault(topic, []).append(handler)
    def unsubscribe(self, topic, handler):
        if topic in self.subscriptions:
            self.subscriptions[topic] = [h for h in self.subscriptions[topic] if h != handler]
    def publish(self, event):
        self.published.append(event)


class TestRuntimePlugin:
    """AUTO-RUNTIME-001/002: Plugin lifecycle and event handling."""

    def test_plugin_load_and_setup(self):
        """AUTO-RUNTIME-001: Auto plugin can be loaded and setup."""
        plugin = AutoPlugin(config={"local_store_path": "./.rosclaw_auto_test_plugin"})
        plugin.setup(ctx=FakeRuntimeContext())
        assert plugin.engine is not None
        assert plugin.subscriber is not None
        assert plugin.publisher is not None

    def test_plugin_start_subscribes_events(self):
        """AUTO-RUNTIME-001b: Start subscribes to event bus topics."""
        bus = FakeEventBus()
        plugin = AutoPlugin(config={"local_store_path": "./.rosclaw_auto_test_plugin2"})
        plugin.bind_event_bus(bus)
        plugin.setup()
        plugin.start()
        assert len(bus.subscriptions) >= 1
        assert "rosclaw.practice.failed" in bus.subscriptions

    def test_plugin_stop_unsubscribes_events(self):
        """AUTO-RUNTIME-002: Stop unsubscribes from event bus."""
        bus = FakeEventBus()
        plugin = AutoPlugin(config={"local_store_path": "./.rosclaw_auto_test_plugin3"})
        plugin.bind_event_bus(bus)
        plugin.setup()
        plugin.start()
        assert len(bus.subscriptions) > 0
        plugin.stop()
        for topic, handlers in bus.subscriptions.items():
            assert len(handlers) == 0

    def test_plugin_health(self):
        """AUTO-RUNTIME-003: Health check returns correct status."""
        plugin = AutoPlugin(config={"local_store_path": "./.rosclaw_auto_test_plugin4"})
        plugin.setup()
        h = plugin.health()
        assert h["plugin"] == "rosclaw-auto"
        assert h["version"] == "1.0.0"
        assert h["status"] == "healthy"

    def test_plugin_config_hot_reload(self):
        """AUTO-RUNTIME-004: Config changes are reflected."""
        plugin = AutoPlugin(config={"max_rounds": 10})
        assert plugin.config.max_rounds == 10
        plugin.config.max_rounds = 20
        assert plugin.config.max_rounds == 20
