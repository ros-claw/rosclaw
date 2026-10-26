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
    "late_result",
    [
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "EXPIRED",
        "TIMED_OUT",
        "ORPHANED",
        "BLOCKED",
        "DEGRADED",
        "unverified",
        "other_mission",
    ],
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
                    "action_id": first["action_id"],
                    "receipt_id": "owned_action",
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
                {
                    "action_id": first["action_id"],
                    "receipt_id": "owned_action",
                    "final_state": "RUNNING",
                    "verified": False,
                },
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
            assert second["action_id"] == first["action_id"]
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


@pytest.mark.parametrize("terminal", ["TIMED_OUT", "ORPHANED", "BLOCKED", "DEGRADED"])
async def test_interrupted_dispatch_reconciles_authoritative_terminal_without_replay(
    tmp_path, terminal
):
    import asyncio

    service, mission = await _setup(tmp_path)
    try:
        request, card = await _propose(
            service,
            mission,
            "ur5e.move_to_pose",
            idem="interrupted-dispatch",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        admission = ActionAdmissionService(service)
        transport = AsyncMock(side_effect=asyncio.CancelledError("cognitive call interrupted"))
        with (
            patch("rosclaw.agentd.action_dispatch.request_action", transport),
            pytest.raises(asyncio.CancelledError),
        ):
            await admission.execute(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        txn = ActionTxnStore(service._store.connection).get_by_approval(card["approval_id"])
        assert txn.state == "DISPATCHING" and txn.action_id and txn.receipt_id
        await service._events.append(
            mission.mission_id,
            AgentEventType.RECEIPT_RECEIVED,
            {
                "action_id": txn.action_id,
                "receipt_id": txn.receipt_id,
                "final_state": terminal,
                "verified": False,
            },
        )
        with patch("rosclaw.agentd.action_dispatch.request_action", AsyncMock()) as dispatch:
            result = admission.reconcile(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        assert result["status"] == terminal and result["executed"] is False
        assert result["terminal"] is True and result["reconciliation_required"] is False
        assert result["action_id"] == txn.action_id and result["txn_id"] == txn.txn_id
        dispatch.assert_not_called()
    finally:
        await service.close()


async def test_ids_persist_before_actual_sim_transport_and_lost_result_can_reconcile(tmp_path):
    service, mission = await _setup(tmp_path)
    try:
        request, card = await _propose(
            service,
            mission,
            "ur5e.move_to_pose",
            idem="persist-before-send",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        captured = {}

        async def lose_response(*args, **kwargs):
            txn = ActionTxnStore(service._store.connection).get_by_approval(card["approval_id"])
            assert txn.state == "DISPATCHING"
            assert txn.action_id and txn.receipt_id and txn.action_id != txn.receipt_id
            captured.update(action_id=txn.action_id, receipt_id=txn.receipt_id)
            raise ConnectionError("accepted, but response stream disconnected")

        admission = ActionAdmissionService(service)
        # Exercise the actual dispatch and SIM executor, replacing only MCP IO.
        with patch("rosclaw.agentd.sim_executor.SimActionChannel._call_tool", lose_response):
            initial = await admission.execute(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        assert initial["status"] == "UNKNOWN"
        assert initial["action_id"] == captured["action_id"]
        assert initial["receipt_id"] == captured["receipt_id"]
        await service._events.append(
            mission.mission_id,
            AgentEventType.RECEIPT_RECEIVED,
            {**captured, "final_state": "COMPLETED", "verified": True},
        )
        with patch("rosclaw.agentd.action_dispatch.request_action", AsyncMock()) as dispatch:
            final = admission.reconcile(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        assert final["status"] == "COMPLETED"
        assert final["action_id"] == captured["action_id"]
        dispatch.assert_not_called()
    finally:
        await service.close()


async def test_daemon_transport_retains_preallocated_identity_and_rejects_substitution():
    from rosclaw.agentd.action_channel import ActionChannelError
    from tests.agentd.test_nonreal_action_session_lifecycle import Client, channel, kwargs

    client = Client()
    transport = channel(client)
    await transport.request_nonreal_action(**kwargs(), action_id="persisted-before-IO")
    assert client.actions[0].action_id == "persisted-before-IO"
    client.request_action = lambda envelope: {"action_id": "unrelated-action"}
    with pytest.raises(ActionChannelError, match="changed the submitted action identity"):
        await transport.request_nonreal_action(**kwargs(), action_id="second-owned-ID")


@pytest.mark.parametrize(
    "case",
    [
        "COMPLETED",
        "TIMED_OUT",
        "ORPHANED",
        "BLOCKED",
        "unverified",
        "wrong_body",
        "wrong_mode",
        "running",
        "SYNTHETIC",
    ],
)
async def test_complete_response_loss_queries_original_authority_without_any_dispatch(
    tmp_path, case
):
    from unittest.mock import Mock

    from rosclaw.agentd.action_channel import DaemonActionChannel

    service, mission = await _setup(tmp_path)
    try:
        request, card = await _propose(
            service,
            mission,
            "ur5e.move_to_pose",
            idem="authority-readback",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        daemon = Mock()
        service._action_channel = DaemonActionChannel(
            daemon,
            actor_id="agent",
            body_id=mission.body_binding.body_id,
            body_hash=mission.body_binding.effective_body_hash,
        )
        admission = ActionAdmissionService(service)
        with patch(
            "rosclaw.agentd.action_dispatch.request_action",
            AsyncMock(side_effect=ConnectionError("lost entire response")),
        ):
            initial = await admission.execute(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        action_id = initial["action_id"]
        assert initial["status"] == "UNKNOWN" and action_id
        daemon.get_action_status.return_value = {
            "action_id": action_id,
            "state": "RUNNING" if case == "running" else "FINISHED",
        }
        terminal = (
            "COMPLETED"
            if case in ["unverified", "wrong_body", "wrong_mode", "running", "SYNTHETIC"]
            else case
        )
        daemon.get_execution_receipt.return_value = {
            "receipt": {
                "action_id": action_id,
                "body_id": "foreign" if case == "wrong_body" else mission.body_binding.body_id,
                "body_snapshot_hash": mission.body_binding.effective_body_hash,
                "capability_id": "ur5e.move_to_pose",
                "execution_mode": "REAL" if case == "wrong_mode" else "SIMULATION",
                "trust_level": "SYNTHETIC" if case == "SYNTHETIC" else "SIMULATED",
                "final_state": terminal,
                "verified": case == "COMPLETED",
                "evidence_level": "TASK_VERIFIED" if case == "COMPLETED" else "REQUESTED",
            }
        }
        with patch("rosclaw.agentd.action_dispatch.request_action", AsyncMock()) as dispatch:
            result = await admission.reconcile_authority(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        expected = (
            "UNKNOWN"
            if case in ["unverified", "wrong_body", "wrong_mode", "running", "SYNTHETIC"]
            else case
        )
        assert result["status"] == expected
        assert result["action_id"] == action_id and result["txn_id"] == initial["txn_id"]
        assert result["executed"] is (case == "COMPLETED")
        assert result["side_effects_dispatched"] is False
        daemon.get_action_status.assert_called_once_with(action_id)
        if case != "running":
            daemon.get_execution_receipt.assert_called_once_with(action_id)
        daemon.request_action.assert_not_called()
        daemon.create_session.assert_not_called()
        dispatch.assert_not_called()
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
            {
                "action_id": initial["action_id"],
                "receipt_id": "restart_owned",
                "final_state": "COMPLETED",
                "verified": True,
            },
        )
        dispatch = AsyncMock()
        with patch("rosclaw.agentd.action_dispatch.request_action", dispatch):
            result = ActionAdmissionService(reopened).reconcile(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        assert result["status"] == "COMPLETED"
        assert result["txn_id"] == initial["txn_id"]
        assert result["action_id"] == initial["action_id"]
        dispatch.assert_not_called()
    finally:
        await reopened.close()


async def test_pending_shortcut_cannot_reconcile_another_mission_or_session(tmp_path):
    from rosclaw.agentd.pi_bridge.action_admission import ActionRequestContext
    from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
    from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

    service, mission = await _setup(tmp_path)
    try:
        request, card = await _propose(
            service,
            mission,
            "ur5e.move_to_pose",
            idem="owned-pending",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        admission = ActionAdmissionService(service)
        with patch(
            "rosclaw.agentd.action_dispatch.request_action",
            AsyncMock(
                return_value=HandlerOutcome(
                    text="pending", terminal_receipt=False, evidence_ref="receipt://owned"
                )
            ),
        ):
            result = await admission.execute(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        assert result["status"] == "UNKNOWN"
        other = service.create_mission("separate owned session; no physical task")
        bindings = SessionBindingStore(service._store.connection)
        bindings.bind(
            pi_session_id="other",
            pi_session_path="",
            mission_id=other.mission_id,
            body_id=other.body_binding.body_id,
            execution_mode="SIMULATION",
            created_by="user:local:1000",
        )
        bindings.acquire_lease(
            mission_id=other.mission_id, pi_session_id="other", owner_pid=2, owner_uid=1000
        )
        other_request = ActionRequestContext(
            "other",
            other.mission_id,
            0,
            other.body_binding.effective_body_hash,
            "SIMULATION",
            "read-another",
            "not-used",
        )
        with pytest.raises(ToolBridgeError, match="ownership mismatch"):
            await admission.execute(
                card["approval_id"], request=other_request, caller_pid=2, caller_uid=1000
            )
        with pytest.raises(ToolBridgeError, match="ownership mismatch"):
            admission.reconcile(
                card["approval_id"], request=other_request, caller_pid=2, caller_uid=1000
            )
    finally:
        await service.close()


async def test_new_idempotency_key_cannot_bypass_pending_physical_result(tmp_path):
    from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

    service, mission = await _setup(tmp_path)
    try:
        request, card = await _propose(
            service,
            mission,
            "ur5e.move_to_pose",
            idem="first-action",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        admission = ActionAdmissionService(service)
        with patch(
            "rosclaw.agentd.action_dispatch.request_action",
            AsyncMock(
                return_value=HandlerOutcome(
                    text="pending", terminal_receipt=False, evidence_ref="receipt://first_owned"
                )
            ),
        ):
            first = await admission.execute(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        assert first["status"] == "UNKNOWN"
        with pytest.raises(ToolBridgeError, match="unresolved"):
            await _propose(
                service,
                mission,
                "ur5e.move_to_pose",
                idem="attempted-replay-as-new-key",
                args={"x": 0.35, "y": 0.25, "z": 0.45},
            )
        await service._events.append(
            mission.mission_id,
            AgentEventType.RECEIPT_RECEIVED,
            {
                "action_id": first["action_id"],
                "receipt_id": "first_owned",
                "final_state": "COMPLETED",
                "verified": True,
            },
        )
        assert (
            admission.reconcile(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )["status"]
            == "COMPLETED"
        )
        _, new_card = await _propose(
            service,
            mission,
            "ur5e.move_to_pose",
            idem="explicit-new-decision-after-reconciliation",
            args={"x": 0.30, "y": 0.25, "z": 0.45},
        )
        assert new_card["approval_id"] != card["approval_id"]
    finally:
        await service.close()
