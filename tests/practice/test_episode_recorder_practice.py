"""Tests for EpisodeRecorder integration with PracticeCoordinator sessions."""

from __future__ import annotations

import tempfile
import time

from rosclaw.core.event_bus import EventBus
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.episode_recorder import EpisodeRecorder


class _FakeSeekDBBridge:
    def __init__(self):
        self.events = []

    def commit(self, event):
        self.events.append(event)


def test_episode_recorder_finalizes_on_practice_session_finished():
    bus = EventBus()
    bridge = _FakeSeekDBBridge()
    recorder = EpisodeRecorder("test_bot", event_bus=bus, seekdb_bridge=bridge)
    recorder.initialize()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = PracticeConfig(
            robot_id="test_bot",
            task_name="pick cup",
            data_root=tmp,
            sources=SourceConfig(agent=True),
            mock=True,
            event_bus=bus,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(0.3)
        coord.stop()
        recorder.stop()

    assert len(bridge.events) == 1
    event = bridge.events[0]
    assert event.event_id == coord.summary.practice_id
    assert event.event_type == "success"
    assert event.robot_id == "test_bot"
    assert event.agent_instruction == "pick cup"
