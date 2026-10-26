"""Lost response is not failed physics: real journal/txn, fixture transport."""

from unittest.mock import AsyncMock, patch

import pytest

from rosclaw.agentd.action_dispatch import HandlerOutcome
from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService
from rosclaw.agentd.pi_bridge.action_txn import ActionTxnStore
from rosclaw.contracts.agent.agent_event import AgentEventType
from tests.agentd.test_pi_tool_bridge import _setup
from tests.agentd.test_seven3_policy_grant import _propose


@pytest.mark.parametrize(
    "late_result", ["COMPLETED", "FAILED", "CANCELLED", "EXPIRED", "unverified", "other_mission"]
)
async def test_pending_receipt_reconciles_exactly_without_redispatch(tmp_path, late_result):
    service, mission = await _setup(tmp_path)
    try:
        request, card = await _propose(
            service,
            mission,
            "ur5e.move_to_pose",
            idem="lost-result",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        admission = ActionAdmissionService(service)
        transport = AsyncMock(
            return_value=HandlerOutcome(
                text="response pending",
                terminal_receipt=False,
                evidence_ref="receipt://owned_action",
            )
        )
        with patch("rosclaw.agentd.action_dispatch.request_action", transport):
            first = await admission.execute(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
            assert first["status"] == "UNKNOWN"
            assert first["reconciliation_required"] is True
            assert first["terminal"] is False
            # Persist the delayed result, then query the SAME transaction.
            await service._events.append(
                "other" if late_result == "other_mission" else mission.mission_id,
                AgentEventType.RECEIPT_RECEIVED,
                {
                    "action_id": "owned_action",
                    "verified": late_result != "unverified",
                    "final_state": "COMPLETED"
                    if late_result in ["unverified", "other_mission"]
                    else late_result,
                },
            )
            # A delayed older progress update cannot hide a committed terminal.
            await service._events.append(
                mission.mission_id,
                AgentEventType.RECEIPT_RECEIVED,
                {"action_id": "owned_action", "final_state": "RUNNING", "verified": False},
            )
            second = await admission.execute(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        assert transport.await_count == 1
        assert second["status"] == (
            "UNKNOWN" if late_result in ["unverified", "other_mission"] else late_result
        )
        assert second["executed"] is (late_result == "COMPLETED")
        if late_result == "COMPLETED":
            assert second["action_id"] == "owned_action"
    finally:
        await service.close()


async def test_transport_exception_retains_unknown_and_never_replays(tmp_path):
    service, mission = await _setup(tmp_path)
    try:
        request, card = await _propose(
            service,
            mission,
            "ur5e.move_to_pose",
            idem="lost-stream",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        admission = ActionAdmissionService(service)
        transport = AsyncMock(
            side_effect=ConnectionError("response stream lost after possible acceptance")
        )
        with patch("rosclaw.agentd.action_dispatch.request_action", transport):
            first = await admission.execute(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
            second = await admission.execute(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        assert first["status"] == second["status"] == "UNKNOWN"
        assert first["txn_id"] == second["txn_id"]
        assert transport.await_count == 1
        txn = ActionTxnStore(service._store.connection).get(first["txn_id"])
        assert txn.state == "RECEIPT_PENDING"
    finally:
        await service.close()


async def test_delayed_receipt_after_service_restart_is_reconciled_without_dispatch(tmp_path):
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.service import AgentService
    from tests.agentd.test_pi_tool_bridge import MockModelGateway, _turn, mock_profile

    service, mission = await _setup(tmp_path)
    request, card = await _propose(
        service,
        mission,
        "ur5e.move_to_pose",
        idem="restart-result",
        args={"x": 0.35, "y": 0.25, "z": 0.45},
    )
    with patch(
        "rosclaw.agentd.action_dispatch.request_action",
        AsyncMock(
            return_value=HandlerOutcome(
                text="pending", terminal_receipt=False, evidence_ref="receipt://restart_owned"
            )
        ),
    ):
        initial = await ActionAdmissionService(service).execute(
            card["approval_id"], request=request, caller_pid=1, caller_uid=1000
        )
    assert initial["status"] == "UNKNOWN"
    await service.close()
    reopened = AgentService(
        load_agent_config(tmp_path / "config.yaml"),
        tmp_path,
        gateway=MockModelGateway(mock_profile(), [_turn()] * 4),
    )
    try:
        await reopened._events.append(
            mission.mission_id,
            AgentEventType.RECEIPT_RECEIVED,
            {"action_id": "restart_owned", "final_state": "COMPLETED", "verified": True},
        )
        dispatch = AsyncMock()
        with patch("rosclaw.agentd.action_dispatch.request_action", dispatch):
            result = ActionAdmissionService(reopened).reconcile(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        assert result["status"] == "COMPLETED"
        assert result["txn_id"] == initial["txn_id"]
        assert result["action_id"] == "restart_owned"
        dispatch.assert_not_called()
    finally:
        await reopened.close()
