"""A cancel request cannot prove native termination or physical stopping.

Protocol fixtures only: no ROS transport, model, simulator, or robot is opened.
"""

import asyncio
import json
import sqlite3

import pytest

from rosclaw.task_kernel.operation_manager import OperationManager
from tests.agentd.test_p1b3_ros2_action import FakeActionClient, _conn, _task


class Peer(FakeActionClient):
    def __init__(self, *, confirm=False, cancel_error=False):
        super().__init__()
        self.active = False
        self.confirm = confirm
        self.cancel_error = cancel_error

    def send_goal(self, **kwargs):
        super().send_goal(**kwargs)
        self.active = True

    def cancel_goal(self, goal_id):
        super().cancel_goal(goal_id)
        if self.cancel_error:
            raise OSError("fixture cancel transport failed")
        if self.confirm:
            self.active = False
            self.emit_result(5, {"native_cancelled": True})

    def emit_result(self, status, values):
        if status in {4, 5, 6}:
            self.active = False
        super().emit_result(status, values)


def manager(tmp_path, *, grace_s=0.01):
    connection = _conn(tmp_path)
    _task(connection)
    mgr = OperationManager(None, connection)
    original = mgr._cancel_grace

    async def short_grace(operation_id, reason):
        await original(operation_id, reason, grace_s=grace_s)

    mgr._cancel_grace = short_grace
    return mgr, connection


async def start(mgr, client):
    return await mgr.start_action(
        task_id="task_1",
        attempt_id="",
        action="/fixture",
        action_type="test/action/Fixture",
        args={},
        client=client,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_error", [False, True])
async def test_missing_native_cancel_result_remains_unknown_not_cancelled(tmp_path, cancel_error):
    mgr, conn = manager(tmp_path)
    peer = Peer(cancel_error=cancel_error)
    operation = await start(mgr, peer)
    await mgr.cancel(operation["operation_id"], reason="requested-stop")
    row = await mgr.wait(operation["operation_id"], timeout=1)
    assert peer.active is True  # A real native result has NOT been supplied.
    assert row["state"] == "LOST"
    assert row["cancel_reason"] == "requested-stop"
    assert row["failure_code"] == "cancel_unconfirmed"
    evidence = json.loads(row["result_ref"])
    assert evidence["goal_id"] == operation["goal_id"]
    assert evidence["native_terminal_status"] is None
    assert evidence["cancel_confirmed"] is False
    assert evidence["physical_stop_verified"] is None
    conn.close()


@pytest.mark.asyncio
async def test_shutdown_cancels_and_joins_owned_native_action_before_journal_close(tmp_path):
    mgr, conn = manager(tmp_path)
    peer = Peer(confirm=True)
    operation = await start(mgr, peer)
    try:
        await asyncio.gather(mgr.close(), mgr.close())
        assert peer.cancelled == [operation["goal_id"]]
        assert peer.active is False
        assert mgr.get(operation["operation_id"])["state"] == "CANCELLED"
        assert mgr.get(operation["operation_id"])["cancel_reason"] == "supervisor_shutdown"
    finally:
        # Baseline failure must not leave a fixture callback pending on a closed DB.
        if peer.active:
            peer.emit_result(5, {})
            await asyncio.sleep(0)
        conn.close()


@pytest.mark.asyncio
async def test_native_admission_after_supervisor_close_never_sends_a_goal(tmp_path):
    mgr, conn = manager(tmp_path)
    peer = Peer()
    await mgr.close()
    try:
        with pytest.raises(RuntimeError, match="closing"):
            await start(mgr, peer)
        assert peer.sent_goals == []
        assert conn.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 0
    finally:
        if peer.active:
            peer.emit_result(5, {})
            await asyncio.sleep(0)
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [False, True])
async def test_close_owns_pending_admission_even_if_its_requester_disconnects(tmp_path, disconnect):
    mgr, conn = manager(tmp_path)
    peer = Peer(confirm=True)
    entered, release = asyncio.Event(), asyncio.Event()
    original = mgr._start_action_owned

    async def delayed(**kwargs):
        entered.set()
        await release.wait()
        return await original(**kwargs)

    mgr._start_action_owned = delayed
    request = asyncio.create_task(start(mgr, peer))
    await entered.wait()
    if disconnect:
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    closing = asyncio.create_task(mgr.close())
    await asyncio.sleep(0)
    assert not closing.done() and not peer.sent_goals
    release.set()
    await closing
    if not disconnect:
        await request
    assert len(peer.sent_goals) == 1
    assert peer.cancelled == [peer.sent_goals[0]["goal_id"]]
    assert peer.active is False
    assert conn.execute("SELECT state FROM operations").fetchone()[0] == "CANCELLED"
    conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("confirm", [False, True])
async def test_send_reply_failure_keeps_the_goal_owned_and_never_replays_it(tmp_path, confirm):
    mgr, conn = manager(tmp_path)

    class LostReply(Peer):
        def send_goal(self, **kwargs):
            super().send_goal(**kwargs)
            raise OSError("fixture lost send reply after remote acceptance")

    peer = LostReply(confirm=confirm)
    operation = await start(mgr, peer)
    assert operation["state"] == "DEGRADED"
    assert operation["failure_code"] == "action_admission_unknown"
    assert peer.active is True
    await mgr.close()
    row = mgr.get(operation["operation_id"])
    assert len(peer.sent_goals) == 1
    assert peer.cancelled == [operation["goal_id"]]
    assert row["state"] == ("CANCELLED" if confirm else "LOST")
    if not confirm:
        assert peer.active is True
        assert json.loads(row["result_ref"])["physical_stop_verified"] is None
    conn.close()
    # A callback from a closed provider cannot write to an already closed DB.
    peer.emit_result(5, {})
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_late_success_does_not_claim_cancel_confirmation_and_is_not_lost_from_audit(tmp_path):
    mgr, conn = manager(tmp_path)
    peer = Peer()
    operation = await start(mgr, peer)
    await mgr.cancel(operation["operation_id"], reason="original_cancel")
    peer.emit_result(4, {"work_completed": True})
    row = await mgr.wait(operation["operation_id"], timeout=1)
    assert row["state"] == "LOST" and row["cancel_reason"] == "original_cancel"
    evidence = json.loads(row["result_ref"])
    assert evidence["native_terminal_status"] == 4
    assert evidence["cancel_confirmed"] is False
    recorded = conn.execute(
        "SELECT payload_json FROM task_events WHERE event_type='operation.native_result'"
    ).fetchall()
    assert len(recorded) == 1
    assert json.loads(recorded[0][0])["status"] == 4
    peer.emit_result(5, {})
    await asyncio.sleep(0)
    assert mgr.get(operation["operation_id"])["state"] == "LOST"  # primary terminal immutable
    native_statuses = [
        json.loads(row[0])["status"]
        for row in conn.execute(
            "SELECT payload_json FROM task_events WHERE event_type='operation.native_result' ORDER BY seq"
        ).fetchall()
    ]
    assert native_statuses == [4, 5]
    conn.close()


@pytest.mark.asyncio
async def test_actual_agent_service_closes_native_goals_before_closing_its_journal(tmp_path):
    from tests.agentd.test_pi_tool_bridge import _setup

    service, mission = await _setup(tmp_path)
    task = service._task_kernel.ensure_task_for_effect(
        mission_id=mission.mission_id, session_ref="pi_1", backend_native_id="", cwd=str(tmp_path)
    )
    peer = Peer(confirm=True)
    operation = await service._operation_manager.start_action(
        task_id=task["task_id"],
        attempt_id="",
        action="/fixture",
        action_type="test/action/Fixture",
        args={},
        client=peer,
    )
    await service.close()
    assert peer.active is False and peer.cancelled == [operation["goal_id"]]
    conn = sqlite3.connect("file:" + str(tmp_path / "agentd/missions.db") + "?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT state FROM operations WHERE operation_id=?", (operation["operation_id"],)
        ).fetchone()
        assert row[0] == "CANCELLED"
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("send_raises", [False, True])
async def test_actual_action_client_keeps_ambiguous_send_owned_for_cancel(tmp_path, send_raises):
    """Real product client + OperationManager, fixture transport only."""
    import queue

    from rosclaw.connectors.ros.action_client import Ros2ActionClient
    from rosclaw.connectors.ros.transport.base import RosTransportResult

    class AmbiguousTransport:
        def __init__(self):
            self.messages = []
            self.incoming = queue.Queue()
            self.active = False

        def send(self, message):
            self.messages.append(message.copy())
            if message["op"] == "send_action_goal":
                self.active = True
                if send_raises:
                    raise OSError("accepted goal but write acknowledgement lost")
                return RosTransportResult(ok=False, error="ambiguous write failure")
            assert message["op"] == "cancel_action_goal"
            self.active = False
            self.incoming.put(
                RosTransportResult(
                    ok=True,
                    data={
                        "op": "action_result",
                        "id": message["id"],
                        "status": 5,
                        "values": {"cancelled": True},
                    },
                )
            )
            return RosTransportResult(ok=True)

        def receive(self, timeout_sec=None):
            try:
                return self.incoming.get(timeout=min(timeout_sec or 0.01, 0.01))
            except queue.Empty:
                return None

        def close(self):
            pass

    mgr, conn = manager(tmp_path, grace_s=0.25)
    transport = AmbiguousTransport()
    client = Ros2ActionClient(transport)
    try:
        operation = await start(mgr, client)
        assert operation["state"] == "DEGRADED"
        assert transport.active is True
        await mgr.close()
        assert [m["op"] for m in transport.messages] == [
            "send_action_goal",
            "cancel_action_goal",
        ]
        assert transport.messages[1]["id"] == operation["goal_id"]
        assert transport.active is False
        assert mgr.get(operation["operation_id"])["state"] == "CANCELLED"
    finally:
        client.close()
        if client._listener:
            await asyncio.to_thread(client._listener.join, 1)
        conn.close()
