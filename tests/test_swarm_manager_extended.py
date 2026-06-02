"""Extended tests for SwarmRuntimeManager with EventBus."""

import time

from rosclaw.core.event_bus import EventBus, Event
from rosclaw.swarm.manager import SwarmRuntimeManager


class TestSwarmManagerEventBus:
    def test_init_with_event_bus(self):
        bus = EventBus()
        swarm = SwarmRuntimeManager(event_bus=bus)
        swarm.initialize()
        assert swarm.event_bus is bus
        swarm.stop()

    def test_register_via_event(self):
        bus = EventBus()
        swarm = SwarmRuntimeManager(event_bus=bus)
        swarm.initialize()
        bus.publish(Event(
            topic="swarm.register",
            payload={"agent_id": "bot_a", "capabilities": ["pick"]},
            source="test",
        ))
        time.sleep(0.05)
        assert swarm.agent_count == 1
        assert swarm.get_agent_status("bot_a")["capabilities"] == ["pick"]
        swarm.stop()

    def test_allocate_via_event(self):
        bus = EventBus()
        swarm = SwarmRuntimeManager(event_bus=bus)
        swarm.initialize()
        swarm.register_agent("bot_b", ["place"])

        received = []
        def on_allocate(event):  # noqa: E306
            received.append(event.payload)
        bus.subscribe("swarm.allocate_result", on_allocate)

        bus.publish(Event(
            topic="swarm.allocate",
            payload={"task": {"required_capabilities": ["place"], "id": "t1"}},
            source="test",
        ))
        time.sleep(0.05)
        assert len(received) == 1
        assert received[0]["agent_id"] == "bot_b"
        swarm.stop()

    def test_status_via_event(self):
        bus = EventBus()
        swarm = SwarmRuntimeManager(event_bus=bus)
        swarm.initialize()
        swarm.register_agent("bot_c", ["scan"])

        received = []
        def on_status(event):  # noqa: E306
            received.append(event.payload)
        bus.subscribe("swarm.status_result", on_status)

        bus.publish(Event(
            topic="swarm.status",
            payload={"agent_id": "bot_c"},
            source="test",
        ))
        time.sleep(0.05)
        assert len(received) == 1
        assert received[0]["agent_id"] == "bot_c"
        swarm.stop()

    def test_status_all_agents(self):
        bus = EventBus()
        swarm = SwarmRuntimeManager(event_bus=bus)
        swarm.initialize()
        swarm.register_agent("bot_d", ["scan"])

        received = []
        def on_status(event):  # noqa: E306
            received.append(event.payload)
        bus.subscribe("swarm.status_result", on_status)

        bus.publish(Event(
            topic="swarm.status",
            payload={},
            source="test",
        ))
        time.sleep(0.05)
        assert len(received) == 1
        assert "agents" in received[0]["status"]
        assert "bot_d" in received[0]["status"]["agents"]
        swarm.stop()

    def test_allocate_no_match_via_event(self):
        bus = EventBus()
        swarm = SwarmRuntimeManager(event_bus=bus)
        swarm.initialize()
        swarm.register_agent("bot_e", ["pick"])

        received = []
        def on_allocate(event):  # noqa: E306
            received.append(event.payload)
        bus.subscribe("swarm.allocate_result", on_allocate)

        bus.publish(Event(
            topic="swarm.allocate",
            payload={"task": {"required_capabilities": ["place"], "id": "t2"}},
            source="test",
        ))
        time.sleep(0.05)
        assert len(received) == 1
        assert received[0]["agent_id"] is None
        swarm.stop()

    def test_stop_unsubscribes(self):
        bus = EventBus()
        swarm = SwarmRuntimeManager(event_bus=bus)
        swarm.initialize()
        swarm.stop()
        # Should not raise on stop without event_bus
        swarm2 = SwarmRuntimeManager()
        swarm2.initialize()
        swarm2.stop()
