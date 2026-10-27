"""All user cancellation entrypoints distinguish requests from proved terminals."""

from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.server import PiBridgeServer
from tests.agentd.test_pi_tool_bridge import _setup


@pytest.mark.parametrize(
    "method",
    [
        "pi.input.persist",
        "pi.session.interrupt",
        "pi.task.cancel",
        "mission.cancel",
        "native.fail_safe",
        "ui.cancel",
    ],
)
@pytest.mark.parametrize("state", ["RUNNING", "LOST"])
async def test_unproved_operation_is_not_counted_as_cancelled(
    tmp_path: Path, monkeypatch, method, state
):
    service, mission = await _setup(tmp_path)
    conn = service._store.connection
    task_id = "task_unproved_stop"
    conn.execute(
        "INSERT INTO tasks (task_id,mission_id,root_goal,mode,workspace_path,state,"
        "active_revision,locale,created_at,updated_at) "
        "VALUES (?,?,'t','SIMULATION','/tmp/ws','EXECUTING',1,'zh','now','now')",
        (task_id, mission.mission_id),
    )
    conn.execute(
        "INSERT INTO operations (operation_id,task_id,attempt_id,kind,state,resumable,revision,started_at) "
        "VALUES ('op_unproved',?,'','process',?,0,1,'now')",
        (task_id, state),
    )
    conn.commit()
    monkeypatch.setattr(
        service._task_kernel,
        "active_task_for_session",
        lambda *_: service._task_kernel.get_task(task_id),
    )
    try:
        if method == "mission.cancel":
            report = await service.cancel(mission.mission_id)
        elif method == "native.fail_safe":
            from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
            from tests.agentd.test_pi_tool_bridge import _request

            result = await PiToolDispatcher(service)._dispatch(
                _request("rosclaw_fail_safe", mission=mission.mission_id)
            )
            assert result.ok is False and result.status == "UNKNOWN"
            assert result.reconciliation_required is True
            report = result.details
        elif method == "ui.cancel":
            from rosclaw.agentd.ui.command_service import CommandService
            from rosclaw.contracts.ui.commands import CommandRequestV1

            command = CommandService(service)
            result = await command._handlers["turn.cancel"](
                CommandRequestV1(
                    request_id="cancel",
                    idempotency_key="cancel",
                    command_name="cancel",
                    mission_id=mission.mission_id,
                )
            )
            assert result.ok is False
            assert result.error_code == "OPERATION_CANCEL_UNCONFIRMED"
            assert "待确认" in result.message
            report = result.data
        else:
            bridge = PiBridgeServer(service, tmp_path / "run/pi-bridge.sock")
            result = await bridge._dispatch(
                "user:local:1000",
                1,
                method,
                {
                    "token": service.control_token,
                    "mission_id": mission.mission_id,
                    "task_id": task_id,
                    "session_ref": "pi_1",
                    "message_id": "stop_unknown",
                    "text": "别做了",
                },
            )
            assert result["ok"]
            if method == "pi.input.persist":
                assert result["turn_disposition"]["suppress_model_turn"] is True
                report = result["turn_disposition"]["cancel_report"]
            else:
                report = result
        assert report["operations_cancelled"] == 0
        assert report["operations_cancel_requested"] == 1
        assert report["operations_cancel_unconfirmed"] == 1
        assert report["operation_states"]["op_unproved"] == "LOST"
        assert report["reconciliation_required"] is True
        assert service._operation_manager.get("op_unproved")["state"] == "LOST"
    finally:
        await service.close()
