"""Tests for Core modules."""


import pytest

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin, LifecycleState
from rosclaw.core.runtime import Runtime, RuntimeConfig


def test_event_bus_publish_subscribe():
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("test.topic", handler)
    bus.publish(Event(topic="test.topic", payload={"data": 1}, source="test"))
    assert len(received) == 1
    assert received[0].payload["data"] == 1


def test_event_bus_priority():
    bus = EventBus()
    priorities = []

    def handler(event):
        priorities.append(event.priority)

    bus.subscribe("test", handler)
    bus.publish(Event(topic="test", payload={}, priority=EventPriority.HIGH))
    bus.publish(Event(topic="test", payload={}, priority=EventPriority.LOW))
    assert priorities[0] == EventPriority.HIGH
    assert priorities[1] == EventPriority.LOW


def test_event_bus_history():
    bus = EventBus()
    bus.publish(Event(topic="a", payload={}))
    bus.publish(Event(topic="b", payload={}))
    bus.publish(Event(topic="a", payload={}))
    assert len(bus.get_history("a")) == 2
    assert len(bus.get_history()) == 3


def test_event_bus_unsubscribe():
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("test", handler)
    bus.unsubscribe("test", handler)
    bus.publish(Event(topic="test", payload={}))
    assert len(received) == 0


def test_lifecycle_transitions():
    class TestModule(LifecycleMixin):
        def __init__(self):
            super().__init__()
            self.started = False

        def _do_start(self):
            self.started = True

    mod = TestModule()
    assert mod.state == LifecycleState.UNINITIALIZED
    mod.initialize()
    assert mod.state == LifecycleState.READY
    mod.start()
    assert mod.state == LifecycleState.RUNNING
    assert mod.started is True
    mod.stop()
    assert mod.state == LifecycleState.STOPPED


def test_lifecycle_cannot_start_uninitialized():
    class TestModule(LifecycleMixin):
        pass

    mod = TestModule()
    with pytest.raises(RuntimeError):
        mod.start()


def test_runtime_default_config():
    rt = Runtime()
    assert rt.config.robot_id == "rosclaw_default"
    assert rt.config.enable_firewall is True


def test_runtime_status():
    rt = Runtime(RuntimeConfig(enable_firewall=False, enable_memory=False, enable_practice=False))
    rt.initialize()
    status = rt.get_status()
    assert status["robot_id"] == "rosclaw_default"
    assert status["modules"]["firewall"] is False
    rt.stop()


def test_runtime_status_property():
    """Runtime.status property should mirror get_status()."""
    from rosclaw.core.runtime import Runtime
    runtime = Runtime()
    runtime.initialize()
    status = runtime.status
    assert isinstance(status, dict)
    assert "robot_id" in status
    assert "modules" in status
    assert status == runtime.get_status()
    runtime.stop()


# --- Lifecycle coverage: is_running, error_message, pause/resume, exceptions ---

class TestLifecycleCoverage:
    def test_is_running(self):
        class Mod(LifecycleMixin):
            pass

        m = Mod()
        assert m.is_running is False
        m.initialize()
        assert m.is_running is False
        m.start()
        assert m.is_running is True
        m.stop()
        assert m.is_running is False

    def test_error_message_on_init_failure(self):
        class BrokenMod(LifecycleMixin):
            def _do_initialize(self):
                raise ValueError("boom")

        m = BrokenMod()
        with pytest.raises(ValueError, match="boom"):
            m.initialize()
        assert m.state == LifecycleState.ERROR
        assert m.error_message == "boom"

    def test_reinitialize_from_error(self):
        class FlakyMod(LifecycleMixin):
            fail_count = 0

            def _do_initialize(self):
                FlakyMod.fail_count += 1
                if FlakyMod.fail_count == 1:
                    raise RuntimeError("first")

        m = FlakyMod()
        with pytest.raises(RuntimeError):
            m.initialize()
        assert m.state == LifecycleState.ERROR
        m.initialize()
        assert m.state == LifecycleState.READY

    def test_pause_and_resume(self):
        class Mod(LifecycleMixin):
            paused = False
            resumed = False

            def _do_pause(self):
                self.paused = True

            def _do_resume(self):
                self.resumed = True

        m = Mod()
        m.initialize()
        m.start()
        assert m.state == LifecycleState.RUNNING
        m.pause()
        assert m.state == LifecycleState.PAUSED
        assert m.paused is True
        m.resume()
        assert m.state == LifecycleState.RUNNING
        assert m.resumed is True
        m.stop()

    def test_pause_only_from_running(self):
        class Mod(LifecycleMixin):
            pass

        m = Mod()
        m.initialize()
        # Not running yet — pause should be no-op
        m.pause()
        assert m.state == LifecycleState.READY
        m.start()
        m.pause()
        assert m.state == LifecycleState.PAUSED
        m.stop()

    def test_resume_only_from_paused(self):
        class Mod(LifecycleMixin):
            pass

        m = Mod()
        m.initialize()
        m.start()
        # Not paused — resume should be no-op
        m.resume()
        assert m.state == LifecycleState.RUNNING
        m.stop()

    def test_stop_from_ready(self):
        class Mod(LifecycleMixin):
            stopped = False

            def _do_stop(self):
                self.stopped = True

        m = Mod()
        m.initialize()
        assert m.state == LifecycleState.READY
        m.stop()
        assert m.state == LifecycleState.STOPPED
        assert m.stopped is True

    def test_initialize_from_stopped(self):
        class Mod(LifecycleMixin):
            pass

        m = Mod()
        m.initialize()
        m.start()
        m.stop()
        assert m.state == LifecycleState.STOPPED
        m.initialize()
        assert m.state == LifecycleState.READY

    def test_is_ready_property(self):
        class Mod(LifecycleMixin):
            pass

        m = Mod()
        assert m.is_ready is False
        m.initialize()
        assert m.is_ready is True
        m.start()
        assert m.is_ready is True
        m.stop()
        assert m.is_ready is False

    def test_cannot_initialize_from_running(self):
        class Mod(LifecycleMixin):
            pass

        m = Mod()
        m.initialize()
        m.start()
        with pytest.raises(RuntimeError, match="Cannot initialize"):
            m.initialize()

    def test_cannot_initialize_from_initializing(self):
        # Manually set state to INITIALIZING to test the guard
        class Mod(LifecycleMixin):
            pass

        m = Mod()
        m._lifecycle_state = LifecycleState.INITIALIZING
        with pytest.raises(RuntimeError, match="Cannot initialize"):
            m.initialize()
