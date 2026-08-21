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
        workspace_root: str = "",
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
        if (
            active is not None
            and active["state"] in TASK_TERMINAL
            and not (
                active["state"] == "SUCCEEDED"
                and not active["user_accepted_at"]
            )
        ):
            active = None  # 已终态且（非 SUCCEEDED 或已被用户接受）
        if active is not None and active["state"] == "SUCCEEDED":
            # PR-N0：SUCCEEDED 但未经 /done 接受 → 用户修正消息重开
            # 同一任务（revision+1、状态回 RUNNING、旧 verification
            # 立即作废——幽灵成功熔断）。
            task_id = str(active["task_id"])
            revision = int(active["active_revision"]) + 1
            now = datetime.now(UTC).isoformat()
            self._conn.execute(
                "INSERT INTO task_revisions (task_id, revision, "
                "user_message_id, goal_delta, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, revision, message_id, text, now),
            )
            self._conn.execute(
                "UPDATE tasks SET active_revision = ?, state = 'RUNNING', "
                "updated_at = ? WHERE task_id = ?",
                (revision, now, task_id),
            )
            superseded = self._conn.execute(
                "UPDATE verifications SET status = 'SUPERSEDED' "
                "WHERE task_id = ? AND status = 'PASS'",
                (task_id,),
            ).rowcount
            self._emit(task_id, "verification.superseded",
                       {"count": superseded, "reason": "user_rejected"},
                       session_ref=session_ref)
            self._emit(task_id, "task.revised",
                       {"revision": revision, "delta": text[:200],
                        "reopened_from": "SUCCEEDED"},
                       session_ref=session_ref)
            return {
                "task_id": task_id,
                "revision": revision,
                "created_task": False,
                "replayed": False,
                "reopened": True,
                "workspace_path": str(active["workspace_path"]),
                "state": "RUNNING",
            }
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
        #    PR-N1：workspace_path = 调用方解析的真实工作根（用户项目/
        #    default workspace——ActiveTaskContext 唯一事实源）；缺省才
        #    回落 home/tasks/<id>/workspace（兼容存量调用）。
        task_id = new_id("task")
        workspace = (
            Path(workspace_root).resolve()
            if workspace_root
            else self._home / "tasks" / task_id / "workspace"
        )
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
        producer: str = "model:tool",
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """登记交付物：实读文件算 sha256/size（不存在的文件拒绝登记）。
        登记才进交付列表——模型口头提到不算。

        PR-N0：producer 区分受信管道（'kernel:<pipeline>'——内核内部
        登记）与模型工具调用（'model:<tool>'）；相对路径只按任务
        workspace 根解析（禁止按 session cwd 猜——cwd 分裂是事故
        根因之一），找不到时报错带实际解析根。"""
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id!r}")
        workspace = Path(str(task["workspace_path"])).resolve()
        file = Path(path)
        if not file.is_absolute():
            file = workspace / file
        file = file.resolve()
        if not file.exists():
            raise ValueError(
                f"artifact 不存在: {path}（解析根: {workspace}）"
            )
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
        # N4.1：模型自产证据标 EXPERIMENTAL——通过 qualification 前
        # 不当正式能力证据（N 调整方案 §二）。
        meta = dict(metadata or {})
        if producer.startswith("model:"):
            meta.setdefault("evidence_tier", "EXPERIMENTAL")
        meta_json = json.dumps(meta, ensure_ascii=False)
        self._conn.execute(
            "INSERT INTO artifacts (artifact_id, task_id, path, media_type, "
            "sha256, size_bytes, producer_operation_id, producer, "
            "metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (artifact_id, task_id, str(file), media_type, record["sha256"],
             len(content), producer_operation_id or None, producer,
             meta_json, now),
        )
        self._emit(task_id, "artifact.created",
                   {"artifact_id": artifact_id, "path": str(file),
                    "bytes": len(content)})
        record["metadata_json"] = meta_json
        return record

    def finish_task(
        self, *, task_id: str, summary: str, artifact_ids: list[str],
    ) -> dict[str, Any]:
        """FinishRequest（§12.1）：验收真跑 → SUCCEEDED / REPAIR_REQUIRED。
        终态幂等（重放不重复验证、不覆盖——返回原 receipt id）。

        PR-N0 熔断：
        - 验收条件只读任务创建时冻结值（模型收尾不得改规则）；
        - 机器人行为任务（body_id 非空）必须含受信管道证据
          （kernel 内部登记的产物）——模型自产证据不算数。
        """
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id!r}")
        if task["state"] in TASK_TERMINAL:
            # 幂等：SUCCEEDED 重放返回既有终态 + 原 receipt。
            prior = self._conn.execute(
                "SELECT verification_id FROM verifications WHERE task_id = ? "
                "AND status = 'PASS' ORDER BY rowid DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            return {
                "status": task["state"],
                "already_terminal": True,
                "verification_id": str(prior["verification_id"]) if prior else "",
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

        rev_row = self._conn.execute(
            "SELECT acceptance_json FROM task_revisions WHERE task_id = ? "
            "AND revision = ?",
            (task_id, int(task["active_revision"])),
        ).fetchone()
        frozen = json.loads(str(rev_row["acceptance_json"])) if rev_row else {}
        # N4.1：资源证明比对——产物元数据的 resource 块 ↔ 当前权威
        # manifest（resolver）↔ 内容 digest。producer 只是来源身份，
        # 不代替资源证明。
        provenance_failures: list[str] = []
        embodiment_used = self.task_used_embodiment(task_id)
        if embodiment_used:
            robot_id = str(task["body_id"]).removeprefix("sim/")
            resource_proofs = []
            for art in artifacts:
                meta = json.loads(str(art.get("metadata_json") or "{}"))
                resource = meta.get("resource") or {}
                if resource:
                    resource_proofs.append(resource)
            if not resource_proofs:
                provenance_failures.append(
                    "RESOURCE_PROVENANCE_MISSING: 行为任务产物无资源证明"
                )
            else:
                from rosclaw.cognition.resolver import resolve_resource

                product_root = Path(__file__).resolve().parents[3]
                manifest = resolve_resource(
                    "robot", robot_id, product_root=product_root
                )
                if manifest is None:
                    provenance_failures.append(
                        f"RESOURCE_PROVENANCE_MISSING: 无 {robot_id} 权威 "
                        "manifest 可比对"
                    )
                else:
                    expected_digest = manifest.get("digests", {}).get(
                        "mjcf", ""
                    )
                    for proof in resource_proofs:
                        if proof.get("resource_id") != f"robot:{robot_id}":
                            provenance_failures.append(
                                "RESOURCE_ID_MISMATCH: "
                                f"{proof.get('resource_id')} != "
                                f"robot:{robot_id}"
                            )
                        if proof.get("quality") != "PRODUCTION" or (
                            proof.get("canonical") is not True
                        ):
                            provenance_failures.append(
                                "NON_CANONICAL_RESOURCE: "
                                f"quality={proof.get('quality')}"
                            )
                        if expected_digest and (
                            proof.get("model_digest") != expected_digest
                        ):
                            provenance_failures.append(
                                "RESOURCE_DIGEST_MISMATCH: 实际加载模型 "
                                "与权威 manifest 摘要不符"
                            )
        trusted_present = any(
            str(a.get("producer") or "").startswith("kernel:")
            for a in artifacts
        )
        verdict = verdict_for(
            artifacts=artifacts,
            acceptance=frozen,
            workspace=Path(task["workspace_path"]),
            summary=summary,
            require_trusted_evidence=embodiment_used,
            trusted_evidence_present=trusted_present,
            extra_failures=provenance_failures,
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

    #: 具身执行工具（用了这些 = 行为任务——受信证据规则才武装）。
    _EMBODIMENT_TOOLS = frozenset({
        "rosclaw_task", "rosclaw_execute", "rosclaw_request_action",
    })

    def note_tool_use(self, task_id: str, tool_name: str) -> None:
        """具身工具使用落账（dispatcher 在 _execute_validated 调用）——
        N4.1：行为任务的判定依据是实际用了具身执行工具，不是
        body 存在（编码任务绑着机器人 body 也是编码任务）。"""
        if tool_name in self._EMBODIMENT_TOOLS:
            self._emit(task_id, "task.tool_used", {"tool": tool_name})

    def task_used_embodiment(self, task_id: str) -> bool:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM task_events WHERE task_id = ? "
            "AND event_type = 'task.tool_used'",
            (task_id,),
        ).fetchone()
        return bool(row and row["c"] > 0)

    def set_acceptance(self, task_id: str, acceptance: dict) -> None:
        """验收条件在任务创建/修订时冻结（PR-N0）——finish 不接受
        模型临时传入的新规则。"""
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id!r}")
        self._conn.execute(
            "UPDATE task_revisions SET acceptance_json = ? "
            "WHERE task_id = ? AND revision = ?",
            (json.dumps(acceptance, ensure_ascii=False), task_id,
             int(task["active_revision"])),
        )
        self._emit(task_id, "acceptance.frozen",
                   {"revision": int(task["active_revision"])})

    def accept_task(self, task_id: str) -> None:
        """/done：用户接受（PR-N0）——SUCCEEDED 永久关闭；此后新消息
        开新任务，不污染已接受结果。"""
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id!r}")
        if task["state"] != "SUCCEEDED":
            raise ValueError(
                f"任务未验收通过（{task['state']}）——不能接受"
            )
        if task["user_accepted_at"]:
            return  # 幂等
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE tasks SET user_accepted_at = ? WHERE task_id = ?",
            (now, task_id),
        )
        self._emit(task_id, "task.accepted", {"accepted_at": now})

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
