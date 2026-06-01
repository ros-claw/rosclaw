"""Extended tests for swarm coordinator and consensus gaps."""

import pytest

from rosclaw.core.event_bus import EventBus
from rosclaw.swarm.coordinator import SwarmCoordinator
from rosclaw.swarm.consensus import RaftLikeConsensus


class TestSwarmCoordinatorExtended:
    def test_deregister_agent(self):
        sc = SwarmCoordinator()
        sc.register_agent("bot_1", ["pick"])
        assert "bot_1" in sc._agents
        sc.deregister_agent("bot_1")
        assert "bot_1" not in sc._agents
        # Idempotent
        sc.deregister_agent("bot_1")

    def test_decompose_task_sequential_assembly(self):
        sc = SwarmCoordinator()
        subtasks = sc.decompose_task({
            "id": "assemble",
            "type": "sequential_assembly",
            "steps": [
                {"capabilities": ["insert"]},
                {"capabilities": ["screw"]},
            ],
        })
        assert len(subtasks) == 2
        assert subtasks[0]["type"] == "assembly_step"
        assert subtasks[1]["type"] == "assembly_step"

    def test_request_bids_non_idle_agent(self):
        sc = SwarmCoordinator()
        sc.register_agent("bot_1", ["pick"])
        sc._agents["bot_1"]["status"] = "busy"
        bids = sc.request_bids({"required_capabilities": ["pick"]})
        assert bids == []

    def test_request_bids_with_distance(self):
        sc = SwarmCoordinator()
        sc.register_agent("bot_1", ["pick"], position=(0.0, 0.0, 0.0))
        bids = sc.request_bids({
            "required_capabilities": ["pick"],
            "target_position": (3.0, 4.0, 0.0),
        })
        assert len(bids) == 1
        # cost = 1.0 + distance(0,0 to 3,4) = 1.0 + 5.0 = 6.0
        assert bids[0].cost == pytest.approx(6.0)

    def test_allocate_task_publishes_event(self):
        bus = EventBus()
        received = []
        def handler(event):  # noqa: E306
            received.append(event)
        bus.subscribe("swarm.task_allocated", handler)

        sc = SwarmCoordinator(event_bus=bus)
        sc.register_agent("bot_1", ["pick"])
        allocation = sc.allocate_task({
            "id": "task_1",
            "type": "single",
            "required_capabilities": ["pick"],
        })
        assert allocation.feasible is True
        assert len(received) == 1
        assert received[0].payload["agent_id"] == "bot_1"

    def test_propose_state_consensus_reached(self):
        bus = EventBus()
        received = []
        def handler(event):  # noqa: E306
            received.append(event)
        bus.subscribe("swarm.consensus_reached", handler)

        sc = SwarmCoordinator(event_bus=bus)
        sc.register_agent("bot_1", ["pick"])
        sc.register_agent("bot_2", ["place"])
        sc.register_agent("bot_3", ["scan"])

        sc.propose_state("bot_1", "target_pos", (1.0, 2.0), 1.0)
        sc.propose_state("bot_2", "target_pos", (1.1, 2.1), 2.0)
        assert len(received) == 1
        assert received[0].payload["key"] == "target_pos"


class TestRaftLikeConsensusExtended:
    def test_vote_rejected_old_timestamp(self):
        rc = RaftLikeConsensus("a1", ["a2", "a3"], quorum=2)
        rc.set_leader(True)
        rc.propose("x", 10, 10.0)
        # Follower vote with older timestamp should be rejected
        accepted = rc.vote("x", "a2", 20, 5.0)
        assert accepted is False

    def test_check_commit_no_entry(self):
        rc = RaftLikeConsensus("a1", ["a2", "a3"])
        assert rc.check_commit("nonexistent") is False
