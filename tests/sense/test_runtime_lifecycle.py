"""Tests for rosclaw.sense.runtime SenseRuntime lifecycle."""

from rosclaw.core.event_bus import EventBus
from rosclaw.core.lifecycle import LifecycleState
from rosclaw.sense.config import SenseConfig
from rosclaw.sense.runtime import SenseRuntime


class TestSenseRuntimeLifecycle:
    def test_initialize_ready(self):
        runtime = SenseRuntime(
            config=SenseConfig(robot_id="g1", collector="mock", update_hz=0.0),
            event_bus=EventBus(),
        )
        runtime.initialize()
        assert runtime.state == LifecycleState.READY
        runtime.stop()

    def test_tick_returns_body_sense(self):
        runtime = SenseRuntime(
            config=SenseConfig(robot_id="g1", collector="mock", update_hz=0.0),
            event_bus=EventBus(),
        )
        runtime.initialize()
        sense = runtime.tick()
        assert sense.robot_id == "g1"
        assert sense.overall_status == "ready"
        runtime.stop()

    def test_get_latest_state(self):
        runtime = SenseRuntime(
            config=SenseConfig(robot_id="g1", collector="mock", update_hz=0.0),
            event_bus=EventBus(),
        )
        runtime.initialize()
        runtime.tick()
        assert runtime.get_latest_state() is not None
        assert runtime.get_latest_sense() is not None
        runtime.stop()

    def test_get_readiness(self):
        runtime = SenseRuntime(
            config=SenseConfig(robot_id="g1", collector="mock", update_hz=0.0),
            event_bus=EventBus(),
        )
        runtime.initialize()
        readiness = runtime.get_readiness(task="observe_scene")
        assert readiness.capabilities["observe_scene"].status == "ready"
        runtime.stop()
