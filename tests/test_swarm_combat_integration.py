"""Integration test — Swarm Multi-Agent Combat Task.

Validates the full flow:
  1. Task decomposition
  2. Auction allocation
  3. Consensus agreement
  4. EventBus event publishing
  5. EpisodeRecorder (Practice) artifact generation
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.practice.episode_recorder import EpisodeRecorder
from rosclaw.swarm.coordinator import SwarmCoordinator


class TestSwarmCombatIntegration:
    """End-to-end integration test for collaborative multi-agent tasks."""

    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def coordinator(self, bus):
        return SwarmCoordinator(event_bus=bus)

    @pytest.fixture
    def recorder(self, bus, tmp_path):
        rec = EpisodeRecorder(
            robot_id="swarm_test",
            event_bus=bus,
            artifact_base_dir=str(tmp_path),
        )
        rec.initialize()
        yield rec
        rec.stop()

    def test_full_collaborative_carry_flow(self, bus, coordinator, recorder, tmp_path):
        """G1 + UR5 carry table: decompose → allocate → execute → consensus → record."""
        # Register agents
        coordinator.register_agent(
            "g1", ["skill.locomotion", "skill.pick_and_place"], position=(0.0, 0.0, 0.0)
        )
        coordinator.register_agent(
            "ur5e", ["skill.manipulation", "skill.pick_and_place"], position=(1.0, 0.0, 0.0)
        )

        # Define collaborative task
        task = {
            "id": "carry_table_001",
            "type": "parallel_pick",
            "objects": ["table_001"],
            "target_location": "zone_B",
            "target_position": (5.0, 0.0, 0.0),
            "required_capabilities": ["skill.pick_and_place"],
        }

        # 1. Decomposition
        subtasks = coordinator.decompose_task(task)
        assert len(subtasks) == 1  # single object -> single subtask

        # 2. Auction allocation
        allocation = coordinator.allocate_task(task)
        assert allocation.feasible is True
        assert len(allocation.assignments) == 1
        winner = allocation.assignments[0]["agent_id"]
        assert winner in ("g1", "ur5e")

        # 3. Simulate execution via EventBus
        episode_id = "carry_table_001_sub_0"
        bus.publish(Event(
            topic="skill.execution.start",
            payload={
                "episode_id": episode_id,
                "skill_name": "carry_table_001_sub_0",
                "agent_id": winner,
                "initial_state": {"position": "zone_A"},
            },
            source="test",
            priority=EventPriority.HIGH,
        ))
        bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "episode_id": episode_id,
                "skill_name": "carry_table_001_sub_0",
                "agent_id": winner,
                "final_state": {"position": "zone_B"},
                "duration_sec": 3.0,
            },
            source="test",
            priority=EventPriority.HIGH,
        ))

        # 4. Consensus
        ts = time.time()
        coordinator.propose_state("g1", "task_complete", True, ts)
        coordinator.propose_state("ur5e", "task_complete", True, ts + 0.01)
        assert coordinator.get_consensus("task_complete") is True

        # 5. Terminal praxis event
        bus.publish(Event(
            topic="praxis.completed",
            payload={
                "episode_id": "carry_table_001",
                "outcome": {"success": True, "reward": 1.0},
            },
            source="test",
            priority=EventPriority.HIGH,
        ))

        # Allow recorder to finalize
        time.sleep(0.1)

        # 6. Verify EpisodeRecorder artifacts
        episodes = recorder.list_episodes()
        assert len(episodes) >= 1

        # Verify artifact files exist
        episodes_dir = Path(tmp_path) / "episodes"
        assert episodes_dir.exists()
        for ep_name in episodes_dir.iterdir():
            if ep_name.is_dir():
                assert (ep_name / "metadata.json").exists()
                assert (ep_name / "trajectory.jsonl").exists()

    def test_multi_subtask_allocation(self, bus, coordinator):
        """Multiple objects → multiple subtasks → different agents win."""
        coordinator.register_agent(
            "bot_near", ["skill.pick_and_place"], position=(0.0, 0.0, 0.0)
        )
        coordinator.register_agent(
            "bot_far", ["skill.pick_and_place"], position=(10.0, 0.0, 0.0)
        )

        task = {
            "id": "sort_boxes",
            "type": "parallel_pick",
            "objects": ["box_A", "box_B"],
            "target_location": "shelf",
            "target_position": (0.0, 0.0, 0.0),
            "required_capabilities": ["skill.pick_and_place"],
        }

        allocation = coordinator.allocate_task(task)
        assert allocation.feasible is True
        assert len(allocation.assignments) == 2

        # Both agents should be busy now
        assert coordinator._agents["bot_near"]["status"] == "busy"
        assert coordinator._agents["bot_far"]["status"] == "busy"

        # Verify swarm status reflects active tasks
        status = coordinator.get_swarm_status()
        assert status["active_tasks"] == 1
        assert status["agent_count"] == 2

    def test_consensus_not_reached_insufficient_proposals(self, bus, coordinator):
        """Consensus fails when not enough agents propose."""
        coordinator.register_agent("bot_1", ["skill.pick"])
        coordinator.register_agent("bot_2", ["skill.pick"])

        # Only 1 proposal out of 2 agents → need majority (2)
        coordinator.propose_state("bot_1", "ready", True, time.time())
        assert coordinator.get_consensus("ready") is None

    def test_event_bus_receives_all_swarm_events(self, bus, coordinator):
        """All swarm events are published to EventBus and captured."""
        captured: list[Event] = []
        bus.subscribe("swarm.task_allocated", lambda e: captured.append(e))
        bus.subscribe("swarm.consensus_reached", lambda e: captured.append(e))

        coordinator.register_agent("bot_1", ["skill.pick"])
        task = {"id": "t1", "required_capabilities": ["skill.pick"]}
        coordinator.allocate_task(task)

        coordinator.register_agent("bot_2", ["skill.pick"])
        ts = time.time()
        coordinator.propose_state("bot_1", "state_x", "value_a", ts)
        coordinator.propose_state("bot_2", "state_x", "value_a", ts + 0.01)

        allocated = [e for e in captured if e.topic == "swarm.task_allocated"]
        consensus = [e for e in captured if e.topic == "swarm.consensus_reached"]
        assert len(allocated) == 1
        assert len(consensus) == 1
        assert allocated[0].priority == EventPriority.HIGH
        assert consensus[0].priority == EventPriority.HIGH

    def test_episode_recorder_captures_failure(self, bus, coordinator, recorder, tmp_path):
        """Failed task execution is recorded with failure status."""
        coordinator.register_agent("bot_1", ["skill.pick"])
        task = {"id": "fail_task", "required_capabilities": ["skill.pick"]}
        coordinator.allocate_task(task)

        bus.publish(Event(
            topic="skill.execution.start",
            payload={"episode_id": "fail_task", "skill_name": "fail_task"},
            source="test",
        ))
        bus.publish(Event(
            topic="praxis.failed",
            payload={
                "episode_id": "fail_task",
                "outcome": {"success": False, "reward": -1.0},
                "error_log": "Gripper slipped",
            },
            source="test",
        ))

        time.sleep(0.1)
        episodes = recorder.list_episodes()
        assert len(episodes) >= 1
        fail_episodes = [e for e in episodes if e["status"] == "failure"]
        assert len(fail_episodes) >= 1
