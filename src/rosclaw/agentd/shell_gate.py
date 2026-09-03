"""ShellGateBroker（0902 审计 R1-a，§5.2）——shell 降级授权的
Runtime 批准面。

0902 实证：用户说"允许！"系统仍要求 export 全局环境变量并重启。
正确语义：确认卡（允许一次/本任务允许/拒绝）→ grant 绑定
task+revision+scope（shell.unsandboxed）→ Runtime 立即继续原操作。

不变量：
- PENDING/未知 = fail closed；
- 允许一次 = 该 request 行消费（不落 standing grant）；
- 本任务允许 = standing grant（task+revision+scope 唯一）——revision
  变化（语义变化）或任务终态后不再命中；
- 全程进 missions.db（append-only 审计面）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rosclaw.contracts.common import new_id

SCOPE_UNSANDBOXED_SHELL = "shell.unsandboxed"

_DECISIONS = frozenset({"allow_once", "allow_task", "deny"})
_STATUS_BY_DECISION = {
    "allow_once": "APPROVED_ONCE",
    "allow_task": "APPROVED_TASK",
    "deny": "DENIED",
}


class ShellGateBroker:
    """shell 降级授权的 Runtime 权威（agentd 进程内，与 kernel 同
    连接——grant 进同一账本）。"""

    def __init__(self, kernel: Any) -> None:
        self._kernel = kernel
        self._conn = kernel._conn

    def request(self, *, task_id: str, revision: int, mission_id: str,
                session_ref: str, scope: str) -> dict[str, Any]:
        """登记 PENDING 授权请求（幂等——同 task+revision+scope 有
        PENDING 时复用，不重复弹卡）。"""
        existing = self._conn.execute(
            "SELECT * FROM shell_gate_requests WHERE task_id = ? "
            "AND revision = ? AND scope = ? AND status = 'PENDING' "
            "ORDER BY created_at DESC LIMIT 1",
            (task_id, revision, scope),
        ).fetchone()
        if existing is not None:
            return dict(existing)
        request_id = new_id("shg")
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO shell_gate_requests (request_id, task_id, "
            "revision, mission_id, session_ref, scope, status, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?)",
            (request_id, task_id, revision, mission_id, session_ref,
             scope, now),
        )
        return {
            "request_id": request_id, "task_id": task_id,
            "revision": revision, "mission_id": mission_id,
            "session_ref": session_ref, "scope": scope,
            "status": "PENDING", "created_at": now, "decided_at": None,
        }

    def status(self, request_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM shell_gate_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return {"request_id": request_id, "status": "UNKNOWN"}
        return dict(row)

    def decide(self, request_id: str, decision: str) -> dict[str, Any]:
        """用户决定（允许一次/本任务允许/拒绝）——立即生效。"""
        if decision not in _DECISIONS:
            raise ValueError(f"unknown decision {decision!r}")
        row = self._conn.execute(
            "SELECT * FROM shell_gate_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown request {request_id!r}")
        if row["status"] != "PENDING":
            # 幂等：重复决定返回现状（不改变第一次决定）。
            return dict(row)
        now = datetime.now(UTC).isoformat()
        status = _STATUS_BY_DECISION[decision]
        self._conn.execute(
            "UPDATE shell_gate_requests SET status = ?, decided_at = ? "
            "WHERE request_id = ?",
            (status, now, request_id),
        )
        if decision == "allow_task":
            # standing grant：task+revision+scope 唯一。
            self._conn.execute(
                "INSERT OR IGNORE INTO shell_grants (grant_id, task_id, "
                "revision, mission_id, scope, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (new_id("shg_grant"), str(row["task_id"]),
                 int(row["revision"]), str(row["mission_id"]),
                 str(row["scope"]), now),
            )
        updated = self.status(request_id)
        return updated

    def check(self, *, task_id: str, revision: int,
              scope: str = SCOPE_UNSANDBOXED_SHELL) -> bool:
        """standing grant 命中判定：revision 一致 + 任务仍活跃
        （终态任务的 grant 不再命中）。"""
        task = self._kernel.get_task(task_id)
        if task is None:
            return False
        from rosclaw.task_kernel.service import TASK_TERMINAL

        if str(task["state"]) in TASK_TERMINAL:
            return False
        if int(task["active_revision"]) != revision:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM shell_grants WHERE task_id = ? AND revision = ? "
            "AND scope = ? LIMIT 1",
            (task_id, revision, scope),
        ).fetchone()
        return row is not None


__all__ = ["SCOPE_UNSANDBOXED_SHELL", "ShellGateBroker"]
