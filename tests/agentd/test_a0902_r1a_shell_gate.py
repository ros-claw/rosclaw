"""0902 审计 R1-a 红测试：Approval Broker（shell 降级授权）——
删除全局环境变量授权方案的正式路径。

0902 实证：用户已在会话里明确回答"允许！"，系统仍要求其退出、
export ROSCLAW_ALLOW_UNSANDBOXED_SHELL=1、重启——这不是可接受的
产品授权。

正确语义（审计 §5.2）：确认卡只给 允许一次/本任务允许/拒绝；
批准后 Runtime 立即继续原操作；grant 绑定 task+revision+scope。

闭环断言：
1. request → PENDING；decide(允许一次) → 该请求行消费即授权
   （standing grant 不落）；
2. decide(本任务允许) → standing grant 落账 → check 命中；revision
   变化后不再命中（语义变化 = 重新询问）；
3. decide(拒绝) → check 不命中；
4. 全局环境变量不再是正式路径（workspace-pack 的 bash 降级路径
   不读 ROSCLAW_ALLOW_UNSANDBOXED_SHELL——TS 侧断言在
   p06-os-isolation 测试更新）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _kernel(tmp_path: Path):
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, tmp_path)


def _bind(kernel, tmp_path: Path) -> str:
    bound = kernel.bind_message(
        mission_id="m1", session_ref="s1", backend_native_id="s1",
        message_id="msg_1", text="写一个 note.txt", cwd=str(tmp_path),
    )
    return str(bound["task_id"])


class TestShellGateBroker:
    def test_allow_once_consumes_request_no_standing_grant(
        self, tmp_path: Path
    ) -> None:
        from rosclaw.agentd.shell_gate import ShellGateBroker

        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path)
        broker = ShellGateBroker(kernel)
        req = broker.request(
            task_id=task_id, revision=1, mission_id="m1", session_ref="s1",
            scope="shell.unsandboxed",
        )
        assert req["status"] == "PENDING"
        broker.decide(req["request_id"], "allow_once")
        status = broker.status(req["request_id"])
        assert status["status"] == "APPROVED_ONCE"
        # 一次允许不落 standing grant——下一次同类操作要重新问。
        assert not broker.check(task_id=task_id, revision=1)

    def test_allow_task_grants_and_revision_invalidates(
        self, tmp_path: Path
    ) -> None:
        from rosclaw.agentd.shell_gate import ShellGateBroker

        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path)
        broker = ShellGateBroker(kernel)
        req = broker.request(
            task_id=task_id, revision=1, mission_id="m1", session_ref="s1",
            scope="shell.unsandboxed",
        )
        broker.decide(req["request_id"], "allow_task")
        assert broker.check(task_id=task_id, revision=1)
        # revision 变化（用户改需求）→ 不再命中（重新询问）。
        kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_2", text="改一下内容", cwd=str(tmp_path),
        )
        assert not broker.check(task_id=task_id, revision=2)

    def test_deny_no_grant(self, tmp_path: Path) -> None:
        from rosclaw.agentd.shell_gate import ShellGateBroker

        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path)
        broker = ShellGateBroker(kernel)
        req = broker.request(
            task_id=task_id, revision=1, mission_id="m1", session_ref="s1",
            scope="shell.unsandboxed",
        )
        broker.decide(req["request_id"], "deny")
        assert not broker.check(task_id=task_id, revision=1)
        assert broker.status(req["request_id"])["status"] == "DENIED"

    def test_pending_is_fail_closed(self, tmp_path: Path) -> None:
        """PENDING/未知请求不授权（fail closed）。"""
        from rosclaw.agentd.shell_gate import ShellGateBroker

        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path)
        broker = ShellGateBroker(kernel)
        broker.request(
            task_id=task_id, revision=1, mission_id="m1", session_ref="s1",
            scope="shell.unsandboxed",
        )
        assert not broker.check(task_id=task_id, revision=1)

    def test_terminal_task_grant_expires(self, tmp_path: Path) -> None:
        """任务终态后 standing grant 不再命中（grant 是活任务的）。"""
        from rosclaw.agentd.shell_gate import ShellGateBroker

        kernel = _kernel(tmp_path)
        task_id = _bind(kernel, tmp_path)
        broker = ShellGateBroker(kernel)
        req = broker.request(
            task_id=task_id, revision=1, mission_id="m1", session_ref="s1",
            scope="shell.unsandboxed",
        )
        broker.decide(req["request_id"], "allow_task")
        assert broker.check(task_id=task_id, revision=1)
        kernel.transition(task_id, "SUCCEEDED", reason="done")
        assert not broker.check(task_id=task_id, revision=1)


class TestShellGateBridge:
    """bridge 接线层：pi.shell_gate.* 经 PiBridgeServer._dispatch 可达。"""

    async def test_bridge_request_decide_check_flow(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        kernel = service._task_kernel
        bound = kernel.bind_message(
            mission_id=mission.mission_id, session_ref="pi_1",
            backend_native_id="pi_1", message_id="msg_1",
            text="写一个 note.txt", cwd=str(tmp_path),
        )
        task_id = str(bound["task_id"])

        async def call(method: str, params: dict) -> dict:
            return await bridge._dispatch(
                "user:local:1000", 1, method,
                {"token": service.control_token, **params},
            )

        # 无 grant → check 不命中。
        r = await call("pi.shell_gate.check", {
            "task_id": task_id, "revision": 1, "scope": "shell.unsandboxed",
        })
        assert r.get("ok") and r.get("granted") is False
        # request → PENDING。
        r = await call("pi.shell_gate.request", {
            "task_id": task_id, "revision": 1,
            "mission_id": mission.mission_id, "session_ref": "pi_1",
            "scope": "shell.unsandboxed",
        })
        assert r.get("ok")
        req_id = str(r["request"]["request_id"])
        assert r["request"]["status"] == "PENDING"
        # decide 本任务允许 → standing grant → check 命中。
        r = await call("pi.shell_gate.decide", {
            "request_id": req_id, "decision": "allow_task",
        })
        assert r.get("ok") and r["request"]["status"] == "APPROVED_TASK"
        r = await call("pi.shell_gate.check", {
            "task_id": task_id, "revision": 1, "scope": "shell.unsandboxed",
        })
        assert r.get("granted") is True
        # 非法决定 → 带错返回（不崩）。
        r = await call("pi.shell_gate.decide", {
            "request_id": req_id, "decision": "bogus",
        })
        assert r.get("ok") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
