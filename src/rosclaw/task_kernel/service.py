"""TaskKernel（PR-H2，ADR-0012，总纲 v2 §9）——任务唯一权威。

不变量（代码/DB 约束实现，不靠 prompt）：
- 一个用户目标 = 一个 root task（同一 mission+session 有活跃 task 时，
  新消息是 revision，不是新 task——除非显式 /new 或已 /done）；
- 一个 task 全生命周期固定一个 workspace；
- 一个 task 同时只有一个 active primary Harness Session（DB 唯一索引）；
- message_id 幂等（重放不重复建 task/revision）；
- 不用 LLM/字符串相似度决定 task 身份——身份来自输入事务；
- 终态只由 Verifier/用户动作写入（H4 接管 task_finish）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.contracts.common import new_id

#: root task 状态机（§9.4）：ACTIVE 子态 + TERMINAL。
TASK_ACTIVE = frozenset({
    "RUNNING",
    "WAITING_OPERATION",
    "WAITING_INPUT",
    "WAITING_PERMISSION",
    "PAUSED",
    "VERIFYING",
    "RECOVERING",
})
TASK_TERMINAL = frozenset({"SUCCEEDED", "BLOCKED", "FAILED", "CANCELLED"})
TASK_STATES = TASK_ACTIVE | TASK_TERMINAL


class TaskKernel:
    """tasks/revisions/bindings 的事务权威（agentd 进程内）。"""

    def __init__(self, conn: sqlite3.Connection, home: Path) -> None:
        self._conn = conn
        self._home = Path(home)

    # --------------------------------------------------------------
    # 输入事务（§9.3 Root Task 绑定算法）
    # --------------------------------------------------------------
    def bind_message(
        self,
        *,
        mission_id: str,
        session_ref: str,
        backend_native_id: str,
        message_id: str,
        text: str,
        cwd: str,
        mode: str = "SIMULATION",
        body_id: str = "",
        locale: str = "auto",
        force_new: bool = False,
    ) -> dict[str, Any]:
        """用户消息 → root task 绑定（原子）。返回
        {task_id, revision, created_task, workspace_path, replayed}。

        - message_id 已存在 → 返回既有绑定（重放幂等）；
        - 无活跃 task 或 force_new → 创建 task+workspace+revision 1+
          primary binding；
        - 有活跃 task → revision+1（同一 task/workspace/session）。
        """
        now = datetime.now(UTC).isoformat()
        # 1. 重放幂等：message_id 唯一约束是兜底，先查。
        existing = self._conn.execute(
            "SELECT task_id, revision FROM task_revisions "
            "WHERE user_message_id = ?",
            (message_id,),
        ).fetchone()
        if existing is not None:
            task = self.get_task(str(existing["task_id"]))
            assert task is not None
            return {
                "task_id": task["task_id"],
                "revision": int(existing["revision"]),
                "created_task": False,
                "replayed": True,
                "workspace_path": task["workspace_path"],
                "state": task["state"],
            }
        # 2. 活跃 task（本 mission+session 的 primary binding）。
        active = self._conn.execute(
            "SELECT t.* FROM tasks t JOIN task_session_bindings b "
            "ON b.task_id = t.task_id "
            "WHERE t.mission_id = ? AND b.session_ref = ? AND b.active = 1 "
            "AND b.role = 'primary' "
            "ORDER BY t.created_at DESC LIMIT 1",
            (mission_id, session_ref),
        ).fetchone()
        if active is not None and active["state"] in TASK_TERMINAL:
            active = None  # 已终态的 task 不再收 revision
        if active is not None and force_new:
            self._conn.execute(
                "UPDATE task_session_bindings SET active = 0 "
                "WHERE task_id = ? AND session_ref = ?",
                (active["task_id"], session_ref),
            )
            active = None
        if active is not None:
            revision = int(active["active_revision"]) + 1
            self._conn.execute(
                "INSERT INTO task_revisions (task_id, revision, "
                "user_message_id, goal_delta, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (active["task_id"], revision, message_id, text, now),
            )
            self._conn.execute(
                "UPDATE tasks SET active_revision = ?, updated_at = ? "
                "WHERE task_id = ?",
                (revision, now, active["task_id"]),
            )
            self._emit(active["task_id"], "task.revised",
                       {"revision": revision, "delta": text[:200]},
                       session_ref=session_ref)
            return {
                "task_id": str(active["task_id"]),
                "revision": revision,
                "created_task": False,
                "replayed": False,
                "workspace_path": str(active["workspace_path"]),
                "state": str(active["state"]),
            }
        # 3. 新 root task：workspace 固定 + revision 1 + primary binding。
        task_id = new_id("task")
        workspace = self._home / "tasks" / task_id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        for sub in ("artifacts", "checkpoints", "logs", "snapshots"):
            (self._home / "tasks" / task_id / sub).mkdir(parents=True, exist_ok=True)
        self._conn.execute(
            "INSERT INTO tasks (task_id, mission_id, root_goal, mode, body_id, "
            "workspace_path, state, active_revision, locale, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING', 1, ?, ?, ?)",
            (task_id, mission_id, text, mode, body_id, str(workspace),
             locale, now, now),
        )
        self._conn.execute(
            "INSERT INTO task_revisions (task_id, revision, user_message_id, "
            "goal_delta, created_at) VALUES (?, 1, ?, ?, ?)",
            (task_id, message_id, text, now),
        )
        # harness session 登记（backend_native_id 幂等）。
        self._conn.execute(
            "INSERT OR IGNORE INTO harness_sessions (session_ref, backend_id, "
            "backend_native_id, cwd, state, created_at) "
            "VALUES (?, 'pi', ?, ?, 'ACTIVE', ?)",
            (session_ref, backend_native_id, cwd, now),
        )
        self._conn.execute(
            "INSERT INTO task_session_bindings (task_id, session_ref, role, "
            "active, created_at) VALUES (?, ?, 'primary', 1, ?)",
            (task_id, session_ref, now),
        )
        (self._home / "tasks" / task_id / "manifest.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "revisions": 1,
                    "workspace": str(workspace),
                    "harness_backend": "pi",
                    "harness_session_ref": session_ref,
                    "created_at": now,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        self._emit(task_id, "task.started", {"goal": text[:200]},
                   session_ref=session_ref)
        return {
            "task_id": task_id,
            "revision": 1,
            "created_task": True,
            "replayed": False,
            "workspace_path": str(workspace),
            "state": "RUNNING",
        }

    # --------------------------------------------------------------
    # 查询/状态
    # --------------------------------------------------------------
    def get_task(self, task_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_tasks(self, mission_id: str = "") -> list[dict]:
        if mission_id:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE mission_id = ? "
                "ORDER BY created_at DESC",
                (mission_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
        return [dict(r) for r in rows]

    def transition(self, task_id: str, state: str, *, reason: str = "") -> None:
        """状态迁移（终态不可逆——迟到事件不得覆盖 CANCELLED 等）。"""
        assert state in TASK_STATES, f"unknown task state {state}"
        row = self.get_task(task_id)
        if row is None:
            raise ValueError(f"unknown task {task_id!r}")
        if row["state"] in TASK_TERMINAL:
            if row["state"] != state:
                import logging

                logging.getLogger(__name__).warning(
                    "task %s 已终态 %s——拒绝覆盖为 %s",
                    task_id, row["state"], state,
                )
            return
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE tasks SET state = ?, updated_at = ?, "
            "terminal_reason = COALESCE(NULLIF(?, ''), terminal_reason), "
            "accepted_at = CASE WHEN ? = 'SUCCEEDED' THEN ? ELSE accepted_at END "
            "WHERE task_id = ?",
            (state, now, reason, state, now if state == "SUCCEEDED" else "",
             task_id),
        )
        self._emit(task_id, "task.state_changed",
                   {"state": state, "reason": reason[:200]})
        if state in TASK_TERMINAL:
            self._emit(task_id, "task.terminal",
                       {"state": state, "reason": reason[:200]})

    def active_task_for(self, mission_id: str, session_ref: str) -> dict | None:
        row = self._conn.execute(
            "SELECT t.* FROM tasks t JOIN task_session_bindings b "
            "ON b.task_id = t.task_id "
            "WHERE t.mission_id = ? AND b.session_ref = ? AND b.active = 1 "
            "AND b.role = 'primary' "
            "ORDER BY t.created_at DESC LIMIT 1",
            (mission_id, session_ref),
        ).fetchone()
        if row is None or row["state"] in TASK_TERMINAL:
            return None
        return dict(row)

    def _emit(self, task_id: str, event_type: str, payload: dict,
              *, session_ref: str = "") -> None:
        self._conn.execute(
            "INSERT INTO task_events (task_id, session_ref, event_type, "
            "payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, session_ref or None, event_type,
             json.dumps(payload, ensure_ascii=False),
             datetime.now(UTC).isoformat()),
        )

    # --------------------------------------------------------------
    # Artifact 登记 + 验收（PR-H4，§12：终态由 Verifier 决定）
    # --------------------------------------------------------------
    def register_artifact(
        self, *, task_id: str, path: str, media_type: str,
        producer_operation_id: str = "",
    ) -> dict[str, Any]:
        """登记交付物：实读文件算 sha256/size（不存在的文件拒绝登记）。
        登记才进交付列表——模型口头提到不算。"""
        file = Path(path)
        if not file.exists():
            raise ValueError(f"artifact 不存在: {path}")
        content = file.read_bytes()
        artifact_id = new_id("art")
        now = datetime.now(UTC).isoformat()
        record = {
            "artifact_id": artifact_id,
            "task_id": task_id,
            "path": str(file),
            "media_type": media_type,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
        self._conn.execute(
            "INSERT INTO artifacts (artifact_id, task_id, path, media_type, "
            "sha256, size_bytes, producer_operation_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (artifact_id, task_id, str(file), media_type, record["sha256"],
             len(content), producer_operation_id or None, now),
        )
        self._emit(task_id, "artifact.created",
                   {"artifact_id": artifact_id, "path": str(file),
                    "bytes": len(content)})
        return record

    def finish_task(
        self, *, task_id: str, summary: str, artifact_ids: list[str],
        acceptance: dict | None = None,
    ) -> dict[str, Any]:
        """FinishRequest（§12.1）：验收真跑 → SUCCEEDED / REPAIR_REQUIRED。
        终态幂等（重放不重复验证、不覆盖）。"""
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id!r}")
        if task["state"] in TASK_TERMINAL:
            # 幂等：SUCCEEDED 重放返回既有终态。
            return {
                "status": task["state"],
                "already_terminal": True,
                "verification_id": "",
            }
        artifacts = [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM artifacts WHERE task_id = ? AND "
                f"artifact_id IN ({','.join('?' * max(len(artifact_ids), 1))})",
                (task_id, *artifact_ids),
            ).fetchall()
        ] if artifact_ids else []
        from rosclaw.task_kernel.verifier import verdict_for

        verdict = verdict_for(
            artifacts=artifacts,
            acceptance=acceptance or {},
            workspace=Path(task["workspace_path"]),
            summary=summary,
        )
        now = datetime.now(UTC).isoformat()
        if verdict["status"] == "PASS":
            verification_id = new_id("vrf")
            self._conn.execute(
                "INSERT INTO verifications (verification_id, task_id, "
                "revision, status, checks_json, evidence_json, created_at) "
                "VALUES (?, ?, ?, 'PASS', ?, ?, ?)",
                (verification_id, task_id, int(task["active_revision"]),
                 json.dumps({"checks": verdict["checks"]},
                            ensure_ascii=False),
                 json.dumps({"artifact_ids": artifact_ids},
                            ensure_ascii=False),
                 now),
            )
            self._emit(task_id, "verification.completed",
                       {"verification_id": verification_id, "status": "PASS",
                        "checks": verdict["checks"]})
            self.transition(task_id, "SUCCEEDED", reason="verification_passed")
            return {"status": "SUCCEEDED", "verification_id": verification_id}
        # REPAIR_REQUIRED：回同一 session（task 保持活跃——修复不新建）。
        self._emit(task_id, "verification.completed",
                   {"status": "FAIL", "failures": verdict["failures"]})
        return {
            "status": "REPAIR_REQUIRED",
            "failures": verdict["failures"],
            "checks": verdict["checks"],
        }

    def block_task(
        self, *, task_id: str, reason_code: str, detail: str,
        recovery: list[str] | None = None,
    ) -> None:
        """task_blocked（§12.1）：稳定原因码 + 恢复动作。"""
        self._emit(task_id, "task.state_changed",
                   {"state": "BLOCKED", "reason_code": reason_code,
                    "recovery": recovery or []})
        self.transition(
            task_id, "BLOCKED",
            reason=f"{reason_code}: {detail}"[:300],
        )
