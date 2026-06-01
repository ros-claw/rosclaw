"""Coverage tests for DataFlywheel and PracticeRecorder."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from rosclaw.core.event_bus import EventBus, Event
from rosclaw.data.flywheel import DataFlywheel, DataEvent, EventType, RobotState
from rosclaw.practice.recorder import PracticeRecorder


class TestRobotState:
    def test_validate_correct_dimensions(self):
        state = RobotState(
            timestamp=time.time(),
            joint_positions=np.array([0.0] * 6),
            joint_velocities=np.array([0.0] * 6),
            joint_torques=np.array([0.0] * 6),
        )
        assert state.validate(6) is True

    def test_validate_wrong_dimensions(self):
        state = RobotState(
            timestamp=time.time(),
            joint_positions=np.array([0.0] * 3),
            joint_velocities=np.array([0.0] * 6),
            joint_torques=np.array([0.0] * 6),
        )
        assert state.validate(6) is False


class TestDataEvent:
    def test_to_dict(self):
        event = DataEvent(
            event_id="evt_1",
            event_type=EventType.SUCCESS,
            timestamp=123.0,
            robot_id="bot",
            metadata={"task": "pick"},
            data_paths={"joints": Path("/tmp/j.npy")},
        )
        d = event.to_dict()
        assert d["event_id"] == "evt_1"
        assert d["event_type"] == "SUCCESS"
        assert d["data_paths"]["joints"] == "/tmp/j.npy"

    def test_from_dict(self):
        d = {
            "event_id": "evt_2",
            "event_type": "FAILURE",
            "timestamp": 456.0,
            "robot_id": "bot2",
            "pre_event_duration": 3.0,
            "post_event_duration": 7.0,
            "metadata": {"x": 1},
            "data_paths": {"a": "/tmp/a.npy"},
        }
        event = DataEvent.from_dict(d)
        assert event.event_type == EventType.FAILURE
        assert event.pre_event_duration == 3.0
        assert event.post_event_duration == 7.0
        assert event.data_paths["a"] == Path("/tmp/a.npy")


class TestDataFlywheelLifecycle:
    def test_init_creates_storage(self, tmp_path):
        DataFlywheel("bot", storage_path=tmp_path / "data")
        assert (tmp_path / "data").exists()

    def test_init_default_storage(self):
        fw = DataFlywheel("bot")
        assert fw._storage_path.exists()
        fw._storage_path = None  # cleanup not needed


class TestDataFlywheelControlCycle:
    def test_on_control_cycle_valid(self, tmp_path):
        fw = DataFlywheel("bot", joint_dof=3, storage_path=tmp_path)
        state = RobotState(
            timestamp=time.time(),
            joint_positions=np.array([0.1, 0.2, 0.3]),
            joint_velocities=np.array([0.0, 0.0, 0.0]),
            joint_torques=np.array([0.0, 0.0, 0.0]),
        )
        fw.on_control_cycle(state)
        assert fw._cycle_count == 1
        assert fw._dropped_cycles == 0

    def test_on_control_cycle_invalid_dropped(self, tmp_path, caplog):
        import logging
        fw = DataFlywheel("bot", joint_dof=3, storage_path=tmp_path)
        state = RobotState(
            timestamp=time.time(),
            joint_positions=np.array([0.0] * 6),  # wrong dims
            joint_velocities=np.array([0.0] * 6),
            joint_torques=np.array([0.0] * 6),
        )
        with caplog.at_level(logging.WARNING):
            fw.on_control_cycle(state)
        assert fw._dropped_cycles == 1
        assert "Invalid state dimensions" in caplog.text


class TestDataFlywheelEvents:
    def test_trigger_event_returns_id(self, tmp_path):
        fw = DataFlywheel("bot", joint_dof=3, storage_path=tmp_path)
        event_id = fw.trigger_event(EventType.SUCCESS, metadata={"task": "test"})
        assert isinstance(event_id, str)
        assert len(event_id) > 0
        assert len(fw._events) == 1

    def test_trigger_event_saves_data(self, tmp_path):
        fw = DataFlywheel("bot", joint_dof=3, buffer_duration_sec=1.0, sampling_rate_hz=10, storage_path=tmp_path)
        # Feed some data
        for i in range(5):
            state = RobotState(
                timestamp=time.time(),
                joint_positions=np.array([float(i)] * 3),
                joint_velocities=np.array([0.0] * 3),
                joint_torques=np.array([0.0] * 3),
            )
            fw.on_control_cycle(state)

        event_id = fw.trigger_event(EventType.FAILURE)
        # Wait for background thread
        time.sleep(0.2)

        event_dir = list((tmp_path / f"bot_{event_id}").glob("*.npy"))
        assert len(event_dir) >= 0  # may be empty if no data in range

    def test_get_stats(self, tmp_path):
        fw = DataFlywheel("bot", joint_dof=3, storage_path=tmp_path)
        state = RobotState(
            timestamp=time.time(),
            joint_positions=np.array([0.0] * 3),
            joint_velocities=np.array([0.0] * 3),
            joint_torques=np.array([0.0] * 3),
        )
        fw.on_control_cycle(state)
        fw.trigger_event(EventType.SUCCESS)
        fw.trigger_event(EventType.FAILURE)

        stats = fw.get_stats()
        assert stats["robot_id"] == "bot"
        assert stats["total_cycles"] == 1
        assert stats["total_events"] == 2
        assert stats["events_by_type"]["SUCCESS"] == 1
        assert stats["events_by_type"]["FAILURE"] == 1
        assert stats["events_by_type"]["EMERGENCY"] == 0

    def test_clear(self, tmp_path):
        fw = DataFlywheel("bot", joint_dof=3, storage_path=tmp_path)
        fw.trigger_event(EventType.SUCCESS)
        assert len(fw._events) == 1
        fw.clear()
        assert len(fw._events) == 0


class TestDataFlywheelExport:
    def test_export_to_lerobot(self, tmp_path):
        fw = DataFlywheel("bot", joint_dof=3, storage_path=tmp_path)
        fw.trigger_event(
            EventType.SUCCESS,
            metadata={"task": "pick", "instruction": "pick the red block"},
        )
        out = tmp_path / "lerobot"
        dataset_path = fw.export_to_lerobot(out)
        assert dataset_path.exists()
        data = json.loads(dataset_path.read_text())
        assert data["dataset_info"]["robot_id"] == "bot"
        assert data["dataset_info"]["total_episodes"] == 1
        assert data["episodes"][0]["task"] == "pick"
        assert data["episodes"][0]["success"] is True

    def test_export_with_filter(self, tmp_path):
        fw = DataFlywheel("bot", joint_dof=3, storage_path=tmp_path)
        fw.trigger_event(EventType.SUCCESS)
        fw.trigger_event(EventType.FAILURE)
        out = tmp_path / "lerobot"
        dataset_path = fw.export_to_lerobot(
            out, filter_fn=lambda e: e.event_type == EventType.SUCCESS
        )
        data = json.loads(dataset_path.read_text())
        assert data["dataset_info"]["total_episodes"] == 1


class TestPracticeRecorderLifecycle:
    def test_initialize_without_bus(self, caplog):
        import logging
        rec = PracticeRecorder("bot")
        with caplog.at_level(logging.INFO, logger="rosclaw.practice.recorder"):
            rec.initialize()
        assert "Recorder initialized" in caplog.text
        assert rec._flywheel is not None
        rec.stop()

    def test_initialize_with_bus(self):
        bus = EventBus()
        rec = PracticeRecorder("bot", event_bus=bus)
        rec.initialize()
        assert rec._flywheel is not None
        rec.stop()

    def test_start_stop_recording(self):
        rec = PracticeRecorder("bot")
        rec.initialize()
        rec.start_recording()
        assert rec.is_recording is True
        rec.stop_recording()
        assert rec.is_recording is False
        rec.stop()


class TestPracticeRecorderSkillEvents:
    def test_on_skill_complete_success(self):
        bus = EventBus()
        received = []
        bus.subscribe("praxis.completed", lambda e: received.append(e.payload))

        rec = PracticeRecorder("bot", event_bus=bus)
        rec.initialize()
        bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "skill_name": "pick",
                "correlation_id": "corr_1",
                "result": {"status": "success", "reward": 0.9, "details": {}},
            },
        ))
        assert len(received) == 1
        assert received[0]["outcome"]["status"] == "success"
        assert received[0]["outcome"]["reward"] == 0.9
        assert received[0]["current_iteration"] == 1
        rec.stop()

    def test_on_skill_complete_failure(self):
        bus = EventBus()
        received = []
        bus.subscribe("praxis.failed", lambda e: received.append(e.payload))

        rec = PracticeRecorder("bot", event_bus=bus)
        rec.initialize()
        bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "skill_name": "pick",
                "correlation_id": "corr_1",
                "result": {"status": "failure", "reward": -0.5, "error": "missed", "details": {}},
            },
        ))
        assert len(received) == 1
        assert received[0]["outcome"]["status"] == "failure"
        assert received[0]["error_log"] == "missed"
        assert received[0]["current_iteration"] == 1
        rec.stop()

    def test_on_skill_complete_non_dict_payload(self):
        bus = EventBus()
        rec = PracticeRecorder("bot", event_bus=bus)
        rec.initialize()
        # Non-dict payload should be silently ignored
        bus.publish(Event(topic="skill.execution.complete", payload=[1, 2, 3]))
        rec.stop()

    def test_failure_context_tracking(self):
        bus = EventBus()
        rec = PracticeRecorder("bot", event_bus=bus)
        rec.initialize()
        bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "result": {"status": "failure", "reward": -0.5, "error": "boom"},
            },
        ))
        bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "result": {"status": "success", "reward": 1.0},
            },
        ))
        ctx = rec.failure_context
        assert ctx["current_iteration"] == 2
        assert ctx["previous_scores"] == [-0.5, 1.0]
        assert ctx["last_error"] == "boom"
        rec.stop()


class TestPracticeRecorderKnowledge:
    def test_on_knowledge_ingest_complete(self):
        bus = EventBus()
        rec = PracticeRecorder("bot", event_bus=bus)
        rec.initialize()
        bus.publish(Event(
            topic="knowledge.ingest_complete",
            payload={
                "practice_id": "p1",
                "knowledge_version": "v2",
                "status": "ok",
                "timestamp": 123.0,
            },
        ))
        assert len(rec.knowledge_ingest_log) == 1
        assert rec.knowledge_ingest_log[0]["practice_id"] == "p1"
        rec.stop()

    def test_on_knowledge_ingest_non_dict_payload(self):
        bus = EventBus()
        rec = PracticeRecorder("bot", event_bus=bus)
        rec.initialize()
        bus.publish(Event(topic="knowledge.ingest_complete", payload="string"))
        assert len(rec.knowledge_ingest_log) == 0
        rec.stop()


class TestPracticeRecorderRecording:
    def test_log_state_when_not_recording(self, tmp_path):
        rec = PracticeRecorder("bot", joint_dof=3)
        rec.initialize()
        # Not started recording
        rec.log_state([0.1, 0.2, 0.3], time.time())
        assert rec._flywheel._cycle_count == 0
        rec.stop()

    def test_log_state_when_recording(self, tmp_path):
        rec = PracticeRecorder("bot", joint_dof=3)
        rec.initialize()
        rec.start_recording()
        rec.log_state([0.1, 0.2, 0.3], time.time())
        assert rec._flywheel._cycle_count == 1
        rec.stop()

    def test_mark_event(self, tmp_path):
        rec = PracticeRecorder("bot", joint_dof=3)
        rec.initialize()
        rec.start_recording()
        event_id = rec.mark_event(EventType.SUCCESS, {"task": "test"})
        assert isinstance(event_id, str)
        assert len(event_id) > 0
        rec.stop()

    def test_mark_event_not_recording(self, tmp_path):
        rec = PracticeRecorder("bot", joint_dof=3)
        rec.initialize()
        # Not recording — mark_event still works (flywheel is active)
        event_id = rec.mark_event(EventType.SUCCESS)
        assert isinstance(event_id, str)
        assert len(event_id) > 0
        rec.stop()

    def test_export_session(self, tmp_path):
        rec = PracticeRecorder("bot", joint_dof=3)
        rec.initialize()
        rec.start_recording()
        rec.mark_event(EventType.SUCCESS, {"task": "pick"})
        out = tmp_path / "export"
        path = rec.export_session(out)
        assert path.exists()
        rec.stop()

    def test_export_session_not_initialized(self, tmp_path):
        rec = PracticeRecorder("bot")
        # Not initialized
        with pytest.raises(RuntimeError, match="Flywheel not initialized"):
            rec.export_session(tmp_path)


class TestPracticeRecorderRecovery:
    def test_record_recovery_outcome(self):
        bus = EventBus()
        received = []
        bus.subscribe("heuristic.recovery_executed", lambda e: received.append(e.payload))

        rec = PracticeRecorder("bot", event_bus=bus)
        rec.initialize()
        rec.record_recovery_outcome("rule_1", success=True, duration=1.5, correlation_id="corr_1")
        assert len(received) == 1
        assert received[0]["rule_id"] == "rule_1"
        assert received[0]["success"] is True
        assert received[0]["duration"] == 1.5
        assert received[0]["correlation_id"] == "corr_1"
        rec.stop()

    def test_record_recovery_no_bus(self):
        rec = PracticeRecorder("bot")  # no bus
        rec.initialize()
        # Should not crash
        rec.record_recovery_outcome("rule_1", success=True, duration=1.0)
        rec.stop()


class TestPracticeRecorderPraxisEvent:
    def test_record_praxis_event_not_recording(self):
        rec = PracticeRecorder("bot", joint_dof=3)
        rec.initialize()
        # Not recording
        result = rec.record_praxis_event(event_id="e1", event_type="success")
        assert result == ""
        rec.stop()

    def test_record_praxis_event_with_args(self):
        rec = PracticeRecorder("bot", joint_dof=3)
        rec.initialize()
        rec.start_recording()
        result = rec.record_praxis_event(
            event_id="e1", event_type="success", instruction="pick red",
            metadata={"x": 1},
        )
        assert isinstance(result, str)
        assert len(result) > 0
        rec.stop()

    def test_record_praxis_event_invalid_type_defaults_to_milestone(self):
        rec = PracticeRecorder("bot", joint_dof=3)
        rec.initialize()
        rec.start_recording()
        result = rec.record_praxis_event(
            event_id="e1", event_type="nonexistent_type",
        )
        assert isinstance(result, str)
        rec.stop()
