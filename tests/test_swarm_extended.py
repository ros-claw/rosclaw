"""Extended tests for rosclaw.swarm coordinator and consensus."""

import time

import pytest

from rosclaw.swarm.coordinator import AgentBid, SwarmCoordinator, TaskAllocation
from rosclaw.swarm.consensus import RaftLikeConsensus


class TestSwarmCoordinator:
    def test_register_agent(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick", "skill.place"])
        assert "bot_1" in coord._agents
        assert coord._agents["bot_1"]["status"] == "idle"

    def test_decompose_single_task(self):
        coord = SwarmCoordinator()
        task = {"id": "t1", "type": "single", "required_capabilities": ["skill.pick"]}
        subs = coord.decompose_task(task)
        assert len(subs) == 1

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

    def test_allocate_task_success(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick_and_place"])
        task = {
            "id": "t3",
            "type": "single",
            "required_capabilities": ["skill.pick_and_place"],
        }
        result = coord.allocate_task(task)
        assert result.feasible is True
        assert result.assignments[0]["agent_id"] == "bot_1"
        assert coord._agents["bot_1"]["status"] == "busy"

    def test_allocate_task_no_agent(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick"])
        task = {"id": "t4", "required_capabilities": ["skill.place"]}
        result = coord.allocate_task(task)
        assert result.feasible is False

    def test_consensus_propose(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick"])
        coord.register_agent("bot_2", ["skill.place"])
        coord.register_agent("bot_3", ["skill.pick"])
        coord.propose_state("bot_1", "object_location", "table_center", time.time())
        coord.propose_state("bot_2", "object_location", "table_center", time.time())
        assert coord.get_consensus("object_location") == "table_center"

    def test_get_swarm_status(self):
        coord = SwarmCoordinator()
        coord.register_agent("bot_1", ["skill.pick"])
        status = coord.get_swarm_status()
        assert status["agent_count"] == 1


class TestRaftLikeConsensus:
    def test_leader_propose(self):
        consensus = RaftLikeConsensus("bot_1", ["bot_2", "bot_3"])
        consensus.set_leader(True)
        assert consensus.propose("target", "table", time.time()) is True

    def test_follower_cannot_propose(self):
        consensus = RaftLikeConsensus("bot_2", ["bot_1", "bot_3"])
        consensus.set_leader(False)
        assert consensus.propose("target", "table", time.time()) is False

    def test_quorum_commit(self):
        consensus = RaftLikeConsensus("bot_1", ["bot_2", "bot_3"], quorum=2)
        t = time.time()
        consensus.vote("target", "bot_1", "table", t)
        consensus.vote("target", "bot_2", "table", t)
        assert consensus.check_commit("target") is True
        assert consensus.get("target") == "table"

    def test_no_quorum(self):
        consensus = RaftLikeConsensus("bot_1", ["bot_2", "bot_3", "bot_4"], quorum=3)
        consensus.vote("target", "bot_1", "table", time.time())
        assert consensus.check_commit("target") is False
        assert consensus.get("target") is None

    def test_get_all_committed(self):
        consensus = RaftLikeConsensus("bot_1", ["bot_2"], quorum=2)
        t = time.time()
        consensus.vote("k1", "bot_1", "v1", t)
        consensus.vote("k1", "bot_2", "v1", t)
        consensus.check_commit("k1")
        consensus.vote("k2", "bot_1", "v2", t)
        consensus.vote("k2", "bot_2", "v2", t)
        consensus.check_commit("k2")
        committed = consensus.get_all_committed()
        assert committed["k1"] == "v1"
        assert committed["k2"] == "v2"
