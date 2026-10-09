"""Actual agentd dispatcher refusal; no model, host process or robot actions."""

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.connectors.ros.context.memory_worker_effects import (
    memory_worker_host_operation_refused,
)
from tests.agentd.test_pi_tool_bridge import _request, _setup


@pytest.mark.parametrize("mode", ["M0", "M1", "M2", "MALFORMED"])
@pytest.mark.parametrize(
    "tool",
    [
        "rosclaw_process_start",
        "rosclaw_process_status",
        "rosclaw_process_output",
        "rosclaw_process_stop",
        "rosclaw_stop_operation",
    ],
)
async def test_registered_worker_refuses_actual_bridge_before_task_or_operation_access(
    tmp_path, monkeypatch, mode, tool
):
    service, mission = await _setup(tmp_path)
    try:
        service._config.raw["agent"] = {"ros_expert": {"memory_intervention": {"mode": mode}}}
        dispatcher = PiToolDispatcher(service)
        monkeypatch.setattr(
            dispatcher,
            "_ensure_task_for_effect",
            lambda *a: pytest.fail("host process reached task admission"),
        )
        for method in ("start", "get", "events_since", "cancel"):
            monkeypatch.setattr(
                service._operation_manager,
                method,
                lambda *a, **kw: pytest.fail("host operation accessed"),
            )
        response = await dispatcher.execute(
            _request(tool, mission=mission.mission_id, arguments={"command": "true"})
        )
        assert not response.ok
        assert response.error_code == "MEMORY_WORKER_HOST_OPERATION_DISABLED"
        assert service._task_kernel.active_task_for(mission.mission_id, "pi_1") is None
        events = service.events_replay(mission.mission_id, limit=50)
        assert any(e.type.value == "tool.effect_resolved" for e in events)
        assert any(
            e.type.value == "tool.completed" and e.payload.get("ok") is False for e in events
        )
    finally:
        await service.close()


@pytest.mark.parametrize("declaration", [{}, [], "malformed", False, {"mode": "M0"}])
def test_configured_invalid_source_cannot_bypass_canonical_host_effect(declaration):
    raw = {"agent": {"ros_expert": {"memory_intervention": declaration}}}
    assert memory_worker_host_operation_refused(
        raw, tool_name="rosclaw_execute", effect_class="HOST_PROCESS"
    )


@pytest.mark.parametrize(
    "raw", [{}, {"agent": {}}, {"agent": {"ros_expert": {"memory_intervention": None}}}]
)
def test_default_profile_not_restricted(raw):
    assert not memory_worker_host_operation_refused(
        raw, tool_name="rosclaw_process_start", effect_class="HOST_PROCESS"
    )


@pytest.mark.parametrize("effect", ["READ_ONLY", "COMPUTE", "SIMULATED_EFFECT", "PHYSICAL_EFFECT"])
def test_worker_restriction_does_not_grant_or_replace_other_effect_policies(effect):
    raw = {"agent": {"ros_expert": {"memory_intervention": {"mode": "M1"}}}}
    assert not memory_worker_host_operation_refused(
        raw, tool_name="rosclaw_execute", effect_class=effect
    )


@pytest.mark.parametrize("mode", ["M0", "M1", "M2"])
@pytest.mark.parametrize("tool", ["rosclaw_execute", "rosclaw_compute", "rosclaw_observe"])
async def test_canonical_host_effect_cannot_escape_through_generic_entry(
    tmp_path, monkeypatch, mode, tool
):
    from rosclaw.contracts.agent.capability import CapabilityDescriptorV2, CapabilityEffectV1

    service, mission = await _setup(tmp_path)
    try:
        service._config.raw["agent"] = {"ros_expert": {"memory_intervention": {"mode": mode}}}
        service._tool_catalog.register_capability(
            CapabilityDescriptorV2(
                capability_id="test_host_process",
                source="test:no-executor",
                effect=CapabilityEffectV1(**{"class": "HOST_PROCESS"}),
            )
        )

        async def discovered():
            return None

        async def forbidden(*args):
            pytest.fail("generic host effect reached downstream dispatch")

        monkeypatch.setattr(service, "_ensure_mcp_discovered", discovered)
        dispatcher = PiToolDispatcher(service)
        monkeypatch.setattr(dispatcher, "_dispatch", forbidden)
        response = await dispatcher.execute(
            _request(
                tool,
                mission=mission.mission_id,
                arguments={"capability_id": "test_host_process"},
            )
        )
        assert not response.ok
        assert response.error_code == "MEMORY_WORKER_HOST_OPERATION_DISABLED"
        events = service.events_replay(mission.mission_id, limit=50)
        effect = [e for e in events if e.type.value == "tool.effect_resolved"][-1]
        assert effect.payload["effect_class"] == "HOST_PROCESS"
        assert effect.payload["source"] == "capability:test_host_process"
    finally:
        await service.close()


async def test_default_profile_still_routes_existing_process_tool(tmp_path, monkeypatch):
    from rosclaw.contracts.pi.tool_request import PiToolResultV1

    service, mission = await _setup(tmp_path)
    try:
        dispatcher = PiToolDispatcher(service)
        called = []

        async def handler(request):
            called.append(request.tool_name)
            return PiToolResultV1(
                request_id=request.request_id, ok=True, status="COMPLETED", summary="route only"
            )

        monkeypatch.setattr(dispatcher, "_process_start", handler)
        response = await dispatcher.execute(
            _request(
                "rosclaw_process_start", mission=mission.mission_id, arguments={"command": "true"}
            )
        )
        assert response.ok and called == ["rosclaw_process_start"]
    finally:
        await service.close()


async def test_registered_worker_still_requires_original_session_binding(tmp_path):
    service, mission = await _setup(tmp_path)
    try:
        service._config.raw["agent"] = {"ros_expert": {"memory_intervention": {"mode": "M0"}}}
        response = await PiToolDispatcher(service).execute(
            _request("rosclaw_process_start", session="unbound", mission=mission.mission_id)
        )
        assert not response.ok and response.error_code == "SESSION_UNBOUND"
    finally:
        await service.close()


async def test_preexperiment_cached_host_reply_cannot_bypass_new_gate(tmp_path, monkeypatch):
    from rosclaw.contracts.pi.tool_request import PiToolResultV1

    service, mission = await _setup(tmp_path)
    try:
        dispatcher = PiToolDispatcher(service)

        async def previous_handler(request):
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary="SYNTHETIC_PRIVATE_HOST_OP_CANARY",
            )

        monkeypatch.setattr(dispatcher, "_process_start", previous_handler)
        request = _request(
            "rosclaw_process_start", mission=mission.mission_id, arguments={"command": "true"}
        )
        assert (await dispatcher.execute(request)).ok
        original = service._store.connection.execute(
            "SELECT response_json FROM pi_tool_idempotency WHERE idempotency_key=?",
            (request.idempotency_key,),
        ).fetchone()["response_json"]
        service._config.raw["agent"] = {"ros_expert": {"memory_intervention": {"mode": "M0"}}}
        refused = await dispatcher.execute(request)
        assert refused.error_code == "MEMORY_WORKER_CACHE_SCOPE_MISMATCH"
        assert "PRIVATE_HOST_OP_CANARY" not in refused.model_dump_json()
        assert (
            service._store.connection.execute(
                "SELECT response_json FROM pi_tool_idempotency WHERE idempotency_key=?",
                (request.idempotency_key,),
            ).fetchone()["response_json"]
            == original
        )
    finally:
        await service.close()


@pytest.mark.parametrize("change", [None, "tool", "arguments", "policy", "session", "mission"])
async def test_worker_cache_replays_only_same_request_and_policy(tmp_path, change):
    service, mission = await _setup(tmp_path)
    try:
        service._config.raw["agent"] = {"ros_expert": {"memory_intervention": {"mode": "M1"}}}
        dispatcher = PiToolDispatcher(service)
        request = _request("rosclaw_status", mission=mission.mission_id)
        first = await dispatcher.execute(request)
        assert first.ok
        if change == "tool":
            request.tool_name = "rosclaw_process_output"
        elif change == "arguments":
            request.arguments = {"changed": True}
        elif change == "policy":
            service._config.raw["agent"]["ros_expert"]["memory_intervention"]["mode"] = "M2"
        elif change == "session":
            request.pi_session_id = "foreign"
        elif change == "mission":
            request.mission_id = "foreign"
        second = await dispatcher.execute(request)
        if change is None:
            assert second.model_dump() == first.model_dump()
        else:
            assert not second.ok and second.error_code == "MEMORY_WORKER_CACHE_SCOPE_MISMATCH"
    finally:
        await service.close()
