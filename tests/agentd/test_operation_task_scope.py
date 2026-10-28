"""An Operation ID is not authority over another mission or harness task."""

import asyncio
import sys

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


async def start_owned_process(service, mission, session):
    kernel = service._task_kernel
    kernel.persist_input(
        mission_id=mission.mission_id,
        session_ref=session,
        message_id="scope_" + session,
        text="Run a finite fixture process",
    )
    bound = kernel.ensure_task_for_effect(
        mission_id=mission.mission_id,
        session_ref=session,
        backend_native_id=session,
        cwd="",
        mode="SIMULATION",
        body_id=mission.body_binding.body_id,
    )
    operation = await service._operation_manager.start(
        task_id=bound["task_id"],
        attempt_id="main",
        kind="process",
        argv=[
            sys.executable,
            "-u",
            "-c",
            "import time; print('FOREIGN_TASK_OUTPUT', flush=True); time.sleep(20)",
        ],
    )
    for _ in range(100):
        events = service._operation_manager.events_since(bound["task_id"], 0)
        if any(e["event_type"] == "operation.output" for e in events):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("Fixture process produced no actual stdout")
    return operation


@pytest.mark.parametrize(
    "tool",
    [
        "rosclaw_process_status",
        "rosclaw_process_output",
        "rosclaw_process_stop",
        "rosclaw_stop_operation",
    ],
)
@pytest.mark.parametrize("scope", ["other_mission", "other_session"])
async def test_foreign_operation_is_not_read_or_cancel_authority(tmp_path, tool, scope):
    service, mission = await _setup(tmp_path)
    try:
        foreign = (
            service.create_mission("Foreign process owner") if scope == "other_mission" else mission
        )
        operation = await start_owned_process(service, foreign, "pi_foreign")
        result = await PiToolDispatcher(service).execute(
            _request(
                tool,
                mission=mission.mission_id,
                arguments={"operation_id": operation["operation_id"]},
            )
        )
        assert result.ok is False
        assert result.error_code == "NOT_FOUND"
        assert "FOREIGN_TASK_OUTPUT" not in result.summary
        assert service._operation_manager.get(operation["operation_id"])["state"] == "RUNNING"
    finally:
        await service.close()


@pytest.mark.parametrize(
    "tool",
    [
        "rosclaw_process_status",
        "rosclaw_process_output",
        "rosclaw_process_stop",
        "rosclaw_stop_operation",
    ],
)
@pytest.mark.parametrize("historical_binding", [False, True])
async def test_exact_owner_keeps_query_and_cleanup_after_active_task_changes(
    tmp_path, tool, historical_binding
):
    service, mission = await _setup(tmp_path)
    try:
        operation = await start_owned_process(service, mission, "pi_1")
        if historical_binding:
            service._store.connection.execute(
                "UPDATE task_session_bindings SET active=0 WHERE task_id=?",
                (operation["task_id"],),
            )
        result = await PiToolDispatcher(service).execute(
            _request(
                tool,
                mission=mission.mission_id,
                arguments={"operation_id": operation["operation_id"]},
            )
        )
        assert result.ok
        if tool in ["rosclaw_process_stop", "rosclaw_stop_operation"]:
            assert result.status == "CANCELLED"
            assert service._operation_manager.get(operation["operation_id"])["state"] == "CANCELLED"
        else:
            assert result.status == "RUNNING"
            if tool == "rosclaw_process_output":
                assert "FOREIGN_TASK_OUTPUT" in result.summary
        binding = service._store.connection.execute(
            "SELECT active FROM task_session_bindings WHERE task_id=?",
            (operation["task_id"],),
        ).fetchone()
        assert bool(binding["active"]) is not historical_binding
    finally:
        await service.close()
