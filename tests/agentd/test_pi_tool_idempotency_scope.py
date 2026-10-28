"""A transport replay cannot leak a different caller's cached tool result."""

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


@pytest.mark.parametrize(
    "change", ["mission_id", "pi_session_id", "tool_name", "arguments", "caller_uid"]
)
async def test_idempotency_key_does_not_replace_request_ownership(tmp_path, change):
    service, mission = await _setup(tmp_path)
    try:
        dispatcher = PiToolDispatcher(service)
        request = _request("rosclaw_status", mission=mission.mission_id, idem="shared_key")
        original = await dispatcher.execute(request, caller_uid=1000)
        assert original.ok
        changes = {
            "mission_id": "other_mission",
            "pi_session_id": "other_session",
            "tool_name": "rosclaw_process_stop",
            "arguments": {"operation_id": "other_op"},
        }
        altered = (
            request.model_copy(update={change: changes[change]})
            if change != "caller_uid"
            else request
        )
        result = await dispatcher.execute(
            altered, caller_uid=1001 if change == "caller_uid" else 1000
        )
        assert not result.ok and result.error_code == "IDEMPOTENCY_CONFLICT"
        assert original.summary not in result.summary
        replay = await dispatcher.execute(request, caller_uid=1000)
        assert replay.model_dump() == original.model_dump()
    finally:
        await service.close()


async def test_same_intent_replay_ignores_cosmetic_context_refresh(tmp_path):
    service, mission = await _setup(tmp_path)
    try:
        dispatcher = PiToolDispatcher(service)
        request = _request("rosclaw_status", mission=mission.mission_id, idem="same_intent")
        original = await dispatcher.execute(request)

        async def should_not_execute(_):
            raise AssertionError("An idempotent result must not execute another effect")

        dispatcher._execute_validated = should_not_execute
        replay = await dispatcher.execute(
            request.model_copy(
                update={
                    "requested_at": "later",
                    "context_revision": 999,
                    "context_lease_id": "new_context",
                }
            )
        )
        assert replay.model_dump() == original.model_dump()
    finally:
        await service.close()


async def test_legacy_unbound_cache_is_unknown_without_replay_or_overwrite(tmp_path):
    service, mission = await _setup(tmp_path)
    try:
        dispatcher = PiToolDispatcher(service)
        request = _request("rosclaw_status", mission=mission.mission_id, idem="legacy_key")
        original = await dispatcher.execute(request)
        conn = service._store.connection
        legacy = original.model_dump_json()
        conn.execute(
            "UPDATE pi_tool_idempotency SET response_json=? WHERE idempotency_key=?",
            (legacy, request.idempotency_key),
        )
        conn.commit()

        async def should_not_execute(_):
            raise AssertionError("Unknown historical ownership does not authorize replay")

        dispatcher._execute_validated = should_not_execute
        result = await dispatcher.execute(request)
        assert not result.ok and result.status == "UNKNOWN" and result.reconciliation_required
        assert result.error_code == "IDEMPOTENCY_BINDING_UNKNOWN"
        assert (
            conn.execute(
                "SELECT response_json FROM pi_tool_idempotency WHERE idempotency_key=?",
                (request.idempotency_key,),
            ).fetchone()["response_json"]
            == legacy
        )
    finally:
        await service.close()


async def test_inflight_retry_and_conflict_never_execute_a_second_effect(tmp_path):
    import asyncio

    from rosclaw.contracts.pi.tool_request import PiToolResultV1

    service, mission = await _setup(tmp_path)
    first_task = None
    release = asyncio.Event()
    try:
        entered = asyncio.Event()
        calls = []

        async def delayed(request):
            calls.append(request)
            entered.set()
            await release.wait()
            return PiToolResultV1(request_id=request.request_id, ok=True, status="COMPLETED")

        dispatcher = PiToolDispatcher(service)
        dispatcher._execute_validated = delayed
        request = _request("rosclaw_status", mission=mission.mission_id, idem="inflight_key")
        first_task = asyncio.create_task(dispatcher.execute(request))
        await asyncio.wait_for(entered.wait(), 1)
        # A duplicate must return promptly without waiting on or duplicating the effect.
        retry = await asyncio.wait_for(PiToolDispatcher(service).execute(request), 1)
        assert retry.status == "UNKNOWN" and retry.reconciliation_required
        conflict = await PiToolDispatcher(service).execute(
            request.model_copy(update={"mission_id": "foreign"})
        )
        assert conflict.error_code == "IDEMPOTENCY_CONFLICT"
        assert len(calls) == 1
        release.set()
        original = await first_task
        replay = await dispatcher.execute(request)
        assert replay.model_dump() == original.model_dump()
    finally:
        release.set()
        if first_task is not None:
            await first_task
        await service.close()


async def test_interrupted_original_attempt_stays_unknown_after_dispatcher_restart(tmp_path):
    import asyncio

    service, mission = await _setup(tmp_path)
    task = None
    try:
        entered = asyncio.Event()

        async def interrupted(_):
            entered.set()
            await asyncio.Event().wait()

        dispatcher = PiToolDispatcher(service)
        dispatcher._execute_validated = interrupted
        request = _request("rosclaw_status", mission=mission.mission_id, idem="interrupted_key")
        task = asyncio.create_task(dispatcher.execute(request))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        restarted = PiToolDispatcher(service)

        async def forbidden(_):
            raise AssertionError("Interrupted effects must not be re-executed")

        restarted._execute_validated = forbidden
        result = await restarted.execute(request)
        assert not result.ok and result.status == "UNKNOWN" and result.reconciliation_required
        assert result.error_code == "IDEMPOTENCY_IN_FLIGHT"
    finally:
        if task is not None and not task.done():
            task.cancel()
        await service.close()
