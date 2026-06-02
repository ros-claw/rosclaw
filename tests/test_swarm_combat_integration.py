"""Integration test for Swarm multi-agent combat task."""
from rosclaw.swarm.coordinator import SwarmCoordinator
from rosclaw.core.event_bus import EventBus


class TestSwarmCombat:
    def test_swarm_task_decomposition(self):
        bus = EventBus()
        swarm = SwarmCoordinator(event_bus=bus)
        result = swarm.decompose_task({
            'id': 'combat_001',
            'type': 'parallel_pick',
            'objects': ['target_a', 'target_b'],
            'target_location': 'base'
        })
        assert len(result) == 2

    def test_swarm_consensus(self):
        from rosclaw.swarm.consensus import RaftLikeConsensus
        c = RaftLikeConsensus('agent_1', ['agent_1', 'agent_2', 'agent_3'], quorum=2)
        c.set_leader(True)
        assert c.propose('target', [1.0, 2.0], 1000.0)
