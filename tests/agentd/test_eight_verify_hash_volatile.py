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
