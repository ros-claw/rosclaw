"""Real bridge handler admission with fixture-backed receipts; no worker starts."""

import asyncio
import os

import pytest

from rosclaw.agentd.pi_bridge.server import PiBridgeServer
from rosclaw.agentd.turn_store import TurnStore
from tests.agentd.test_pi_tool_bridge import _setup


@pytest.mark.parametrize(
    "bad", [{}, {"operation_id": "foreign"}, {"turn_id": False}, {"task_id": 7}]
)
def test_owned_cancel_requires_complete_tuple(tmp_path, bad):
    async def run():
        service, _ = await _setup(tmp_path)
        try:
            result = await PiBridgeServer(service, tmp_path / "unused.sock")._dispatch(
                f"uid:{os.getuid()}",
                os.getpid(),
                "pi.op.cancel_owned",
                {"token": service.control_token, **bad},
            )
            assert result == {"ok": False, "code": "OWNERSHIP_REQUIRED"}
        finally:
            await service.close()

    asyncio.run(run())


def test_foreign_peer_rejected_before_operation_lookup(tmp_path):
    async def run():
        service, mission = await _setup(tmp_path)
        try:
            result = await PiBridgeServer(service, tmp_path / "unused.sock")._dispatch(
                "uid:1000",
                os.getpid(),
                "pi.op.cancel_owned",
                {
                    "token": service.control_token,
                    "mission_id": mission.mission_id,
                    "session_ref": "pi_1",
                    "task_id": "foreign_task",
                    "operation_id": "foreign_operation",
                    "turn_id": "foreign_turn",
                },
            )
            assert result == {"ok": False, "code": "CALLER_MISMATCH"}
            # OperationManager.get returns {} for absent IDs, not None.
            assert not service._operation_manager.get("foreign_operation")
        finally:
            await service.close()

    asyncio.run(run())


def test_receipt_bound_cancel_admission_and_rejections(tmp_path, monkeypatch):
    async def run():
        service, mission = await _setup(tmp_path)
        try:
            bindings = PiBridgeServer(service, tmp_path / "unused.sock")
            # Re-acquire for THIS process; the store replaces a lease for the
            # same bound session and issues a new token (no forged peer PID).
            bindings._bindings.acquire_lease(
                mission_id=mission.mission_id,
                pi_session_id="pi_1",
                owner_pid=os.getpid(),
                owner_uid=os.getuid(),
            )
            task = service._task_kernel.ensure_task_for_effect(
                mission_id=mission.mission_id,
                session_ref="pi_1",
                backend_native_id="test",
                cwd=str(tmp_path),
                mode="SIMULATION",
                body_id=mission.body_binding.body_id,
                explicit_goal="test cancel",
            )
            task_id = task["task_id"]
            revision = service._task_kernel.get_task(task_id)["active_revision"]
            turn = TurnStore(service._store.connection).record(
                pi_session_id="pi_1",
                mission_id=mission.mission_id,
                text="start",
            )
            params = {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "task_id": task_id,
                "operation_id": "op_fixture",
                "turn_id": turn["turn_id"],
            }
            service._ui_operation_receipts = {
                "op_fixture": {
                    **{
                        k: params[k]
                        for k in ("mission_id", "session_ref", "task_id", "operation_id", "turn_id")
                    },
                    "revision": revision,
                }
            }
            ops = service._operation_manager
            calls = []
            monkeypatch.setattr(
                ops,
                "get",
                lambda op: {"task_id": task_id, "state": "RUNNING"} if op == "op_fixture" else {},
            )

            async def cancel_many(ids, *, reason):
                calls.append((ids, reason))
                return {"ok": True, "cancelled": ids}

            monkeypatch.setattr(ops, "cancel_many", cancel_many)

            async def dispatch(p=params, principal=None, pid=None):
                return await bindings._dispatch(
                    principal or f"uid:{os.getuid()}",
                    os.getpid() if pid is None else pid,
                    "pi.op.cancel_owned",
                    p,
                )

            # A legacy/no-turn session must not resolve a submitted tuple.
            service._store.connection.execute(
                "DELETE FROM user_turns WHERE pi_session_id = ?", ("pi_1",)
            )
            assert (await dispatch())["code"] == "OWNERSHIP_STALE"
            assert calls == []
            turn = TurnStore(service._store.connection).record(
                pi_session_id="pi_1",
                mission_id=mission.mission_id,
                text="start again",
            )
            params["turn_id"] = turn["turn_id"]
            service._ui_operation_receipts["op_fixture"]["turn_id"] = turn["turn_id"]
            assert (await dispatch())["ok"] is True
            assert calls == [(["op_fixture"], "user_interrupt")]
            for changed, expected in [
                ({"mission_id": "other"}, "CALLER_MISMATCH"),
                ({"session_ref": "foreign"}, "CALLER_MISMATCH"),
                ({"task_id": "other"}, "OWNERSHIP_MISMATCH"),
                ({"operation_id": "other"}, "OWNERSHIP_MISMATCH"),
                ({"turn_id": "other"}, "OWNERSHIP_MISMATCH"),
            ]:
                assert (await dispatch({**params, **changed}))["code"] == expected
            assert (await dispatch(principal="uid:1000" if os.getuid() != 1000 else "uid:9999"))[
                "code"
            ] == "CALLER_MISMATCH"
            assert (await dispatch(pid=os.getpid() + 1))["code"] == "CALLER_MISMATCH"
            TurnStore(service._store.connection).record(
                pi_session_id="pi_1",
                mission_id=mission.mission_id,
                text="next turn",
            )
            assert (await dispatch())["code"] == "OWNERSHIP_STALE"
            assert len(calls) == 1
        finally:
            await service.close()

    asyncio.run(run())


def test_natural_success_is_quiescent_only_for_exact_receipt(tmp_path):
    """A real short-lived operation completes before the owned UI cancel RPC."""
    import sys

    async def run():
        service, mission = await _setup(tmp_path)
        try:
            bridge = PiBridgeServer(service, tmp_path / "unused.sock")
            bridge._bindings.acquire_lease(
                mission_id=mission.mission_id,
                pi_session_id="pi_1",
                owner_pid=os.getpid(),
                owner_uid=os.getuid(),
            )
            task = service._task_kernel.ensure_task_for_effect(
                mission_id=mission.mission_id,
                session_ref="pi_1",
                backend_native_id="test",
                cwd=str(tmp_path),
                mode="SIMULATION",
                body_id=mission.body_binding.body_id,
                explicit_goal="natural completion",
            )
            turn = TurnStore(service._store.connection).record(
                pi_session_id="pi_1", mission_id=mission.mission_id, text="first"
            )
            ops = service._operation_manager
            op = await ops.start(
                task_id=task["task_id"],
                attempt_id="main",
                kind="process",
                argv=[sys.executable, "-c", "pass"],
                cwd=str(tmp_path),
            )
            op_id = op["operation_id"]

            async def finished():
                while ops.get(op_id)["state"] not in {"SUCCEEDED", "FAILED", "LOST"}:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(finished(), timeout=5)
            assert ops.get(op_id)["state"] == "SUCCEEDED"
            assert not ops._pid_alive(ops.get(op_id)["pid"])
            params = {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "task_id": task["task_id"],
                "operation_id": op_id,
                "turn_id": turn["turn_id"],
            }
            service._ui_operation_receipts = {
                op_id: {
                    **{
                        k: params[k]
                        for k in ("mission_id", "session_ref", "task_id", "operation_id", "turn_id")
                    },
                    "revision": service._task_kernel.get_task(task["task_id"])["active_revision"],
                }
            }

            async def cancel(p):
                return await bridge._dispatch(
                    f"uid:{os.getuid()}", os.getpid(), "pi.op.cancel_owned", p
                )

            assert await cancel(params) == {
                "ok": True,
                "code": "ALREADY_SUCCEEDED",
                "operations_cancelled": 0,
            }
            assert (await cancel({**params, "operation_id": "foreign"}))[
                "code"
            ] == "OWNERSHIP_MISMATCH"
            assert ops.get(op_id)["state"] == "SUCCEEDED"
            # A genuine second durable turn supersedes the old receipt even
            # if the previous operation really finished; no stale success.
            TurnStore(service._store.connection).record(
                pi_session_id="pi_1", mission_id=mission.mission_id, text="second"
            )
            assert (await cancel(params))["code"] == "OWNERSHIP_STALE"
        finally:
            await service.close()

    asyncio.run(run())
