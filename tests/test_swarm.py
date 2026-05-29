"""Comprehensive tests for rosclaw.swarm — SwarmRuntimeManager, SwarmCoordinator, RaftLikeConsensus.

Target coverage: 70%+ for coordinator.py and consensus.py.
"""

from __future__ import annotations

import time

import pytest

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.swarm.coordinator import AgentBid, SwarmCoordinator, TaskAllocation
from rosclaw.swarm.consensus import ConsensusEntry, Proposal, RaftLikeConsensus
from rosclaw.swarm.manager import SwarmRuntimeManager


# ==================================================================
# SwarmRuntimeManager (existing coverage preserved)
# ==================================================================


class TestSwarmRuntimeManager:
    def test_swarm_register_agent(self):
        swarm = SwarmRuntimeManager()
        swarm.initialize()
        swarm.register_agent("bot_1", ["pick", "place"])
        assert swarm.agent_count == 1
        status = swarm.get_agent_status("bot_1")
        assert status["status"] == "idle"
        swarm.stop()

    def test_swarm_allocate_task(self):
        swarm = SwarmRuntimeManager()
        swarm.initialize()
        swarm.register_agent("bot_1", ["pick"])
        agent = swarm.allocate_task({"required_capabilities": ["pick"], "id": "task_1"})
        assert agent == "bot_1"
        swarm.stop()

    def test_swarm_allocate_no_match(self):
        swarm = SwarmRuntimeManager()
        swarm.initialize()
        swarm.register_agent("bot_1", ["pick"])
        agent = swarm.allocate_task({"required_capabilities": ["place"]})
        assert agent is None
        swarm.stop()


# ==================================================================
# SwarmCoordinator — Task Decomposition
# ==================================================================


class TestSwarmCoordinatorDecomposition:
    def test_decompose_single_task(self):
        coord = SwarmCoordinator()
        task = {"id": "t1", "type": "single", "required_capabilities": ["skill.pick"]}
        subs = coord.decompose_task(task)
        assert len(subs) == 1
        assert subs[0]["id"] == "t1"

    def test_decompose_parallel_pick(self):
        coord = SwarmCoordinator()
        task = {
            "id": "t2",
            "type": "parallel_pick",
            "objects": ["cup", "plate", "fork"],
            "target_location": "sink",
        }
        subs = coord.decompose_task(task)
        assert len(subs) == 3
        assert subs[0]["type"] == "pick_and_place"
        assert subs[0]["location"] == "sink"
        assert subs[0]["required_capabilities"] == ["skill.pick_and_place"]

    def test_decompose_sequential_assembly(self):
        coord = SwarmCoordinator()
        task = {
            "id": "t3",
            "type": "sequential_assembly",
            "steps": [
                {"name": "attach_A", "capabilities": ["skill.screw"]},
                {"name": "attach_B", "capabilities": ["skill.glue"]},
            ],
        }
        subs = coord.decompose_task(task)
        assert len(subs) == 2
        assert subs[0]["type"] == "assembly_step"
        assert subs[0]["required_capabilities"] == ["skill.screw"]
        assert subs[1]["required_capabilities"] == ["skill.glue"]

    def test_decompose_defaults_to_single(self):
        """Unknown task types default to single-task pass-through."""
        coord = SwarmCoordinator()
        task = {"id": "t_unknown", "type": "weird_type", "foo": "bar"}
        subs = coord.decompose_task(task)
        assert len(subs) == 1
        assert subs[0]["foo"] == "bar"


# ==================================================================
# SwarmCoordinator — Agent Lifecycle
# ==================================================================


class TestSwarmCoordinatorAgentLifecycle:
    def test_register_and_deregister_agent(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick"], position=(1.0, 2.0, 0.0))
        assert "bot_1" in coord._agents
        assert coord._agents["bot_1"]["position"] == (1.0, 2.0, 0.0)
        coord.deregister_agent("bot_1")
        assert "bot_1" not in coord._agents
        # Idempotent deregister
        coord.deregister_agent("bot_1")

    def test_register_multiple_agents(self):
        coord = SwarmCoordinator()
        for i in range(5):
            coord.register_agent(f"bot_{i}", ["skill.pick"])
        assert len(coord._agents) == 5


# ==================================================================
# SwarmCoordinator — Auction / Bidding
# ==================================================================


class TestSwarmCoordinatorAuction:
    def test_request_bids_basic(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick_and_place"])
        task = {
            "id": "t1",
            "required_capabilities": ["skill.pick_and_place"],
            "target_position": (0.0, 0.0, 0.0),
        }
        bids = coord.request_bids(task)
        assert len(bids) == 1
        assert bids[0].agent_id == "bot_1"
        assert bids[0].cost == pytest.approx(1.0)

    def test_request_bids_distance_penalty(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_near", ["skill.pick"], position=(0.0, 0.0, 0.0))
        coord.register_agent("bot_far", ["skill.pick"], position=(3.0, 4.0, 0.0))  # dist=5
        task = {
            "id": "t1",
            "required_capabilities": ["skill.pick"],
            "target_position": (0.0, 0.0, 0.0),
        }
        bids = coord.request_bids(task)
        assert len(bids) == 2
        near = next(b for b in bids if b.agent_id == "bot_near")
        far = next(b for b in bids if b.agent_id == "bot_far")
        assert near.cost == pytest.approx(1.0)
        assert far.cost == pytest.approx(6.0)  # 1.0 + 5.0

    def test_request_bids_filters_busy_agents(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_idle", ["skill.pick"])
        coord.register_agent("bot_busy", ["skill.pick"])
        coord._agents["bot_busy"]["status"] = "busy"
        task = {"id": "t1", "required_capabilities": ["skill.pick"]}
        bids = coord.request_bids(task)
        assert len(bids) == 1
        assert bids[0].agent_id == "bot_idle"

    def test_request_bids_filters_missing_capabilities(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick"])
        task = {"id": "t1", "required_capabilities": ["skill.place", "skill.pick"]}
        bids = coord.request_bids(task)
        assert len(bids) == 0

    def test_request_bids_no_target_position(self):
        """When target_position is absent, cost remains 1.0 base."""
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick"], position=(10.0, 10.0, 10.0))
        task = {"id": "t1", "required_capabilities": ["skill.pick"]}
        bids = coord.request_bids(task)
        assert len(bids) == 1
        assert bids[0].cost == pytest.approx(1.0)

    def test_allocate_task_best_bidder_wins(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_cheap", ["skill.pick"], position=(0.0, 0.0, 0.0))
        coord.register_agent("bot_expensive", ["skill.pick"], position=(10.0, 0.0, 0.0))
        task = {
            "id": "t1",
            "required_capabilities": ["skill.pick"],
            "target_position": (0.0, 0.0, 0.0),
        }
        result = coord.allocate_task(task)
        assert result.feasible is True
        assert result.assignments[0]["agent_id"] == "bot_cheap"

    def test_allocate_task_no_capable_agent(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick"])
        task = {"id": "t_fail", "required_capabilities": ["skill.place"]}
        result = coord.allocate_task(task)
        assert result.feasible is False
        assert "No capable agent" in result.reason
        assert len(result.assignments) == 0

    def test_allocate_task_multiple_subtasks(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick_and_place"])
        coord.register_agent("bot_2", ["skill.pick_and_place"])
        task = {
            "id": "t_multi",
            "type": "parallel_pick",
            "objects": ["obj_A", "obj_B"],
            "target_location": "drop_zone",
            "required_capabilities": ["skill.pick_and_place"],
        }
        result = coord.allocate_task(task)
        assert result.feasible is True
        assert len(result.assignments) == 2
        assigned_agents = {a["agent_id"] for a in result.assignments}
        assert assigned_agents.issubset({"bot_1", "bot_2"})

    def test_allocate_task_marks_agents_busy(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick"])
        task = {"id": "t1", "required_capabilities": ["skill.pick"]}
        coord.allocate_task(task)
        assert coord._agents["bot_1"]["status"] == "busy"
        assert coord._agents["bot_1"]["current_task"] is not None

    def test_allocate_task_publishes_event(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("swarm.task_allocated", lambda e: received.append(e))

        coord = SwarmCoordinator(event_bus=bus)
        coord.register_agent("bot_1", ["skill.pick"])
        task = {"id": "t1", "required_capabilities": ["skill.pick"]}
        coord.allocate_task(task)

        assert len(received) == 1
        assert received[0].topic == "swarm.task_allocated"
        assert received[0].payload["agent_id"] == "bot_1"
        assert received[0].priority == EventPriority.HIGH


# ==================================================================
# SwarmCoordinator — Consensus helpers
# ==================================================================


class TestSwarmCoordinatorConsensus:
    def test_propose_state_and_get_consensus(self):
        coord = SwarmCoordinator()
        for i in range(3):
            coord.register_agent(f"bot_{i}", ["skill.pick"])
        ts = time.time()
        coord.propose_state("bot_0", "obj_loc", "table_A", ts)
        coord.propose_state("bot_1", "obj_loc", "table_A", ts + 0.1)
        assert coord.get_consensus("obj_loc") == "table_A"

    def test_propose_state_not_enough_proposals(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_0", ["skill.pick"])
        coord.register_agent("bot_1", ["skill.pick"])
        coord.propose_state("bot_0", "obj_loc", "table_A", time.time())
        # Only 1 proposal out of 2 agents -> need majority (2//2+1 = 2)
        assert coord.get_consensus("obj_loc") is None

    def test_get_consensus_missing_key(self):
        coord = SwarmCoordinator()
        assert coord.get_consensus("nonexistent") is None

    def test_propose_state_publishes_event(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("swarm.consensus_reached", lambda e: received.append(e))

        coord = SwarmCoordinator(event_bus=bus)
        for i in range(3):
            coord.register_agent(f"bot_{i}", ["skill.pick"])
        ts = time.time()
        coord.propose_state("bot_0", "target", "goal_A", ts)
        coord.propose_state("bot_1", "target", "goal_A", ts + 0.01)

        assert len(received) == 1
        assert received[0].topic == "swarm.consensus_reached"
        assert received[0].payload["key"] == "target"
        assert received[0].payload["value"] == "goal_A"


# ==================================================================
# SwarmCoordinator — Status
# ==================================================================


class TestSwarmCoordinatorStatus:
    def test_get_swarm_status_empty(self):
        coord = SwarmCoordinator()
        status = coord.get_swarm_status()
        assert status["agent_count"] == 0
        assert status["active_tasks"] == 0
        assert status["agents"] == []
        assert status["tasks"] == []
        assert status["consensus_keys"] == []

    def test_get_swarm_status_with_agents_and_tasks(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick"])
        coord.register_agent("bot_2", ["skill.place"])
        task = {"id": "t1", "required_capabilities": ["skill.pick"]}
        coord.allocate_task(task)
        coord.propose_state("bot_1", "loc", "here", time.time())

        status = coord.get_swarm_status()
        assert status["agent_count"] == 2
        assert status["active_tasks"] == 1
        assert len(status["agents"]) == 2
        assert status["agents"][0]["status"] == "busy"
        assert status["agents"][1]["status"] == "idle"
        assert len(status["tasks"]) == 1
        assert status["tasks"][0]["assignments"] == 1
        assert status["consensus_keys"] == ["loc"]


# ==================================================================
# RaftLikeConsensus — Leader / Follower basics
# ==================================================================


class TestRaftLikeConsensusBasics:
    def test_default_quorum(self):
        c = RaftLikeConsensus("a", ["b", "c", "d"])
        assert c.quorum == 2  # 3 peers -> 3//2+1 = 2

    def test_custom_quorum(self):
        c = RaftLikeConsensus("a", ["b", "c"], quorum=1)
        assert c.quorum == 1

    def test_set_leader_increments_term(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        assert c._term == 0
        c.set_leader(True)
        assert c._term == 1
        c.set_leader(True)
        assert c._term == 2

    def test_set_follower_does_not_increment_term(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        c.set_leader(True)
        assert c._term == 1
        c.set_leader(False)
        assert c._term == 1

    def test_leader_can_propose(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        c.set_leader(True)
        assert c.propose("key", "value", time.time()) is True

    def test_follower_cannot_propose(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        c.set_leader(False)
        assert c.propose("key", "value", time.time()) is False

    def test_propose_creates_entry(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        c.set_leader(True)
        c.propose("key", "value", time.time())
        assert "key" in c._entries
        assert len(c._entries["key"].proposals) == 1


# ==================================================================
# RaftLikeConsensus — Voting
# ==================================================================


class TestRaftLikeConsensusVoting:
    def test_vote_accepts_first_proposal(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        assert c.vote("key", "b", "value", time.time()) is True

    def test_vote_rejects_stale_timestamp(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        t_old = time.time()
        c.vote("key", "b", "v1", t_old)
        # Same agent tries to vote with older timestamp -> rejected
        assert c.vote("key", "a", "v2", t_old - 1.0) is False

    def test_vote_accepts_newer_timestamp(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        t_old = time.time()
        c.vote("key", "b", "v1", t_old)
        assert c.vote("key", "a", "v2", t_old + 1.0) is True

    def test_vote_rejects_older_timestamp_same_term(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        c._term = 5
        t = time.time()
        c.vote("key", "b", "v1", t)
        # older timestamp, same term -> rejected
        assert c.vote("key", "a", "v2", t - 1.0) is False


# ==================================================================
# RaftLikeConsensus — Commit / Quorum
# ==================================================================


class TestRaftLikeConsensusCommit:
    def test_check_commit_true_on_quorum(self):
        c = RaftLikeConsensus("a", ["b", "c"], quorum=2)
        t = time.time()
        c.vote("key", "a", "v1", t)
        c.vote("key", "b", "v1", t)
        assert c.check_commit("key") is True
        assert c.get("key") == "v1"

    def test_check_commit_false_below_quorum(self):
        c = RaftLikeConsensus("a", ["b", "c", "d"], quorum=3)
        c.vote("key", "a", "v1", time.time())
        assert c.check_commit("key") is False
        assert c.get("key") is None

    def test_check_commit_uses_latest_value(self):
        c = RaftLikeConsensus("a", ["b", "c"], quorum=2)
        t = time.time()
        c.vote("key", "a", "old", t)
        c.vote("key", "b", "new", t + 10.0)
        assert c.check_commit("key") is True
        assert c.get("key") == "new"

    def test_check_commit_sets_agreed_at(self):
        c = RaftLikeConsensus("a", ["b", "c"], quorum=2)
        t = time.time()
        c.vote("key", "a", "v1", t)
        c.vote("key", "b", "v1", t)
        c.check_commit("key")
        assert c._entries["key"].agreed_at == t

    def test_check_commit_nonexistent_key(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        assert c.check_commit("missing") is False

    def test_get_nonexistent_key(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        assert c.get("missing") is None

    def test_get_uncommitted_key(self):
        c = RaftLikeConsensus("a", ["b", "c"], quorum=3)
        c.vote("key", "a", "v1", time.time())
        assert c.get("key") is None

    def test_get_all_committed_empty(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        assert c.get_all_committed() == {}

    def test_get_all_committed_partial(self):
        c = RaftLikeConsensus("a", ["b", "c"], quorum=2)
        t = time.time()
        c.vote("k1", "a", "v1", t)
        c.vote("k1", "b", "v1", t)
        c.check_commit("k1")
        c.vote("k2", "a", "v2", t)  # uncommitted
        committed = c.get_all_committed()
        assert committed == {"k1": "v1"}

    def test_multiple_keys_independent_commit(self):
        c = RaftLikeConsensus("a", ["b", "c"], quorum=2)
        t = time.time()
        c.vote("k1", "a", "v1", t)
        c.vote("k1", "b", "v1", t)
        c.vote("k2", "a", "v2", t)
        c.vote("k2", "b", "v2", t)
        assert c.check_commit("k1") is True
        assert c.check_commit("k2") is True
        assert c.get_all_committed() == {"k1": "v1", "k2": "v2"}


# ==================================================================
# RaftLikeConsensus — Multi-term scenarios
# ==================================================================


class TestRaftLikeConsensusTerms:
    def test_higher_term_overrides(self):
        c = RaftLikeConsensus("a", ["b", "c"])
        c._term = 1
        t = time.time()
        c.vote("key", "a", "old", t)
        # Now term increases (leader re-elected)
        c._term = 2
        c.vote("key", "b", "new", t + 1.0)
        # The newer term should be accepted regardless of timestamp logic
        assert len(c._entries["key"].proposals) == 2


# ==================================================================
# Integration — Multi-robot scenarios
# ==================================================================


class TestMultiRobotIntegration:
    def test_three_robot_consensus_on_target(self):
        """Three robots vote on a shared target location; quorum=2 commits."""
        c = RaftLikeConsensus("bot_1", ["bot_2", "bot_3"], quorum=2)
        t = time.time()
        c.vote("target", "bot_1", "shelf_A", t)
        c.vote("target", "bot_2", "shelf_A", t + 0.1)
        assert c.check_commit("target") is True
        assert c.get("target") == "shelf_A"

    def test_three_robot_split_vote_no_commit(self):
        """Only one vote -> no commit with quorum=2."""
        c = RaftLikeConsensus("bot_1", ["bot_2", "bot_3"], quorum=2)
        c.vote("target", "bot_1", "shelf_A", time.time())
        assert c.check_commit("target") is False
        assert c.get("target") is None

    def test_task_decomposition_then_consensus(self):
        """Full flow: decompose task -> allocate -> agents reach consensus on result."""
        coord = SwarmCoordinator()
        coord.register_agent("ur5", ["skill.pick_and_place"], position=(0.0, 0.0, 0.0))
        coord.register_agent("g1", ["skill.pick_and_place"], position=(2.0, 0.0, 0.0))

        task = {
            "id": "carry_table",
            "type": "parallel_pick",
            "objects": ["leg_1", "leg_2"],
            "target_location": "assembly_zone",
            "required_capabilities": ["skill.pick_and_place"],
            "target_position": (1.0, 0.0, 0.0),
        }

        allocation = coord.allocate_task(task)
        assert allocation.feasible is True
        assert len(allocation.assignments) == 2

        # Simulate agents reporting completion via consensus
        for agent_id in [a["agent_id"] for a in allocation.assignments]:
            coord.propose_state(agent_id, "carry_complete", True, time.time())

        assert coord.get_consensus("carry_complete") is True

    def test_conflict_resolution_closest_wins(self):
        """Two capable agents compete; closest one wins the auction."""
        coord = SwarmCoordinator()
        coord.register_agent("near", ["skill.pick"], position=(1.0, 0.0, 0.0))
        coord.register_agent("far", ["skill.pick"], position=(10.0, 0.0, 0.0))
        task = {
            "id": "t1",
            "required_capabilities": ["skill.pick"],
            "target_position": (0.0, 0.0, 0.0),
        }
        result = coord.allocate_task(task)
        assert result.assignments[0]["agent_id"] == "near"

    def test_multi_robot_sequential_assembly_consensus(self):
        """Sequential assembly with consensus on each step completion."""
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.screw"])
        coord.register_agent("bot_2", ["skill.glue"])

        task = {
            "id": "assemble_chair",
            "type": "sequential_assembly",
            "steps": [
                {"name": "screw_leg", "capabilities": ["skill.screw"]},
                {"name": "glue_pad", "capabilities": ["skill.glue"]},
            ],
        }

        allocation = coord.allocate_task(task)
        assert allocation.feasible is True
        assert len(allocation.assignments) == 2

        # Consensus that assembly is complete
        for agent_id in ["bot_1", "bot_2"]:
            coord.propose_state(agent_id, "assembly_done", True, time.time())
        assert coord.get_consensus("assembly_done") is True

    def test_coordinator_with_event_bus_full_flow(self):
        bus = EventBus()
        events: list[Event] = []

        def capture(event: Event) -> None:
            events.append(event)

        bus.subscribe("swarm.task_allocated", capture)
        bus.subscribe("swarm.consensus_reached", capture)

        coord = SwarmCoordinator(event_bus=bus)
        coord.register_agent("bot_1", ["skill.pick"])
        coord.register_agent("bot_2", ["skill.pick"])

        task = {"id": "t1", "required_capabilities": ["skill.pick"]}
        coord.allocate_task(task)

        coord.propose_state("bot_1", "ready", True, time.time())
        coord.propose_state("bot_2", "ready", True, time.time() + 0.01)

        allocated = [e for e in events if e.topic == "swarm.task_allocated"]
        consensus = [e for e in events if e.topic == "swarm.consensus_reached"]
        assert len(allocated) == 1
        assert len(consensus) == 1
        assert allocated[0].priority == EventPriority.HIGH
        assert consensus[0].priority == EventPriority.HIGH


# ==================================================================
# Dataclass behaviour
# ==================================================================


class TestDataclasses:
    def test_agent_bid_defaults(self):
        bid = AgentBid(agent_id="a", task_id="t", cost=1.0)
        assert bid.capabilities == []
        assert bid.estimated_duration_sec == 0.0

    def test_task_allocation_defaults(self):
        alloc = TaskAllocation(task_id="t")
        assert alloc.assignments == []
        assert alloc.feasible is True
        assert alloc.reason == ""

    def test_proposal_defaults(self):
        p = Proposal(agent_id="a", key="k", value="v", timestamp=1.0)
        assert p.term == 0

    def test_consensus_entry_defaults(self):
        e = ConsensusEntry(key="k")
        assert e.agreed_value is None
        assert e.agreed_at == 0.0
        assert e.proposals == []
        assert e.quorum_size == 1
