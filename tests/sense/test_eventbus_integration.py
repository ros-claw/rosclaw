"""Tests for rosclaw.sense EventBus integration."""

from rosclaw.core.event_bus import EventBus
from rosclaw.core.event_topics import EventTopics
from rosclaw.sense.config import SenseConfig
from rosclaw.sense.runtime import SenseRuntime


class TestEventBusIntegration:
    def test_tick_publishes_state_updated(self):
        bus = EventBus()
        captured = []
        bus.subscribe(EventTopics.SENSE_STATE_UPDATED, captured.append)
        runtime = SenseRuntime(
            config=SenseConfig(robot_id="g1", collector="mock", update_hz=0.0),
            event_bus=bus,
        )
        runtime.initialize()
        runtime.tick()
        assert len(captured) == 1
        runtime.stop()

    def test_tick_publishes_body_updated(self):
        bus = EventBus()
        captured = []
        bus.subscribe(EventTopics.SENSE_BODY_UPDATED, captured.append)
        runtime = SenseRuntime(
            config=SenseConfig(robot_id="g1", collector="mock", update_hz=0.0),
            event_bus=bus,
        )
        runtime.initialize()
        runtime.tick()
        assert len(captured) == 1
        runtime.stop()

    def test_hot_knee_publishes_capability_blocked(self):
        from rosclaw.sense.collectors.mock_collector import MockCollector

        bus = EventBus()
        captured = []
        bus.subscribe(EventTopics.SENSE_CAPABILITY_BLOCKED, captured.append)
        runtime = SenseRuntime(
            config=SenseConfig(robot_id="g1", collector="mock", update_hz=0.0),
            event_bus=bus,
        )
        runtime.initialize()
        # Force hot knee scenario
        runtime._collector = MockCollector(robot_id="g1", scenario="hot_knee")
        runtime.tick()
        assert any(e.payload.get("capability") == "kick_ball" for e in captured)
        runtime.stop()
