def test_hash_ignores_turn_in_flight(tmp_path):
    """turn_in_flight 翻转不得改变 context_hash（真实模型验收轮实测）。"""
    import asyncio

    from tests.agentd.test_pi_tool_bridge import _setup
    async def main():
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.pi_bridge.context import build_embodied_context
        from rosclaw.agentd.pi_bridge.context_lease import context_hash_of
        e1 = build_embodied_context(service, mission.mission_id)
        e2 = build_embodied_context(service, mission.mission_id)
        e2.self_state["turn_in_flight"] = not e1.self_state.get("turn_in_flight", False)
        assert context_hash_of(e1) == context_hash_of(e2)
        await service.close()
    asyncio.run(main())


async def test_ros_context_hash_preserves_readiness_and_ignores_capture_identity(tmp_path):
    from copy import deepcopy

    from rosclaw.agentd.pi_bridge.context import build_embodied_context
    from rosclaw.agentd.pi_bridge.context_lease import context_hash_of
    from tests.agentd.test_pi_tool_bridge import _setup

    service, mission = await _setup(tmp_path)
    try:
        first = build_embodied_context(service, mission.mission_id)
        first.self_state["ros_observations"] = {
            "status": "OBSERVED", "snapshot_hash": "capture1", "captured_at": "time1",
            "admission_facts": {"body_hash": "bound", "readiness": {"coverage.execute": "AVAILABLE"}},
        }
        second = deepcopy(first)
        second.self_state["ros_observations"].update(snapshot_hash="capture2", captured_at="time2")
        assert context_hash_of(first) == context_hash_of(second)
        second.self_state["ros_observations"]["admission_facts"]["readiness"]["coverage.execute"] = "BLOCKED"
        assert context_hash_of(first) != context_hash_of(second)
        second = deepcopy(first)
        second.self_state["ros_observations"]["admission_facts"]["body_hash"] = "changed"
        assert context_hash_of(first) != context_hash_of(second)
    finally:
        await service.close()
