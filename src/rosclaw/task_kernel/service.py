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
from rosclaw.task_kernel.run_store import ensure_run, run_dir, zone_of

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
    def persist_input(
        self, *, mission_id: str, session_ref: str, message_id: str,
        text: str, force_new: bool = False,
    ) -> dict[str, Any]:
        """P0-C（0824 总纲 §6.1）：输入先落会话，不立即创建 Task。

        message_id 幂等（重发/重放返回既有 input）；问候/解释/
        只读查询永远只走这条路——tasks=0 直到首个 effectful call
        或显式 /goal。"""
        existing = self._conn.execute(
            "SELECT * FROM user_inputs WHERE message_id = ?", (message_id,),
        ).fetchone()
        if existing is not None:
            return dict(existing)
        input_id = new_id("inp")
        digest = hashlib.sha256(text.encode()).hexdigest()
        state = "FORCE_NEW" if force_new else "PERSISTED"
        self._conn.execute(
            "INSERT INTO user_inputs (input_id, mission_id, session_ref, "
            "message_id, text, text_digest, delivery_state, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (input_id, mission_id, session_ref, message_id, text,
             f"sha256:{digest}", state, datetime.now(UTC).isoformat()),
        )
        return {
            "input_id": input_id, "mission_id": mission_id,
            "session_ref": session_ref, "message_id": message_id,
            "text": text, "text_digest": f"sha256:{digest}",
            "task_id": None, "delivery_state": "PERSISTED",
        }

    def ensure_task_for_effect(
        self, *, mission_id: str, session_ref: str, backend_native_id: str,
        cwd: str, mode: str = "SIMULATION", body_id: str = "",
        explicit_goal: str = "",
    ) -> dict[str, Any]:
        """P0-C（0824 总纲 §6.2）：首个 effectful call 的原子 admission。

        以 session 最新未附着输入为动机：未附着 → 建 task/新
        revision（沿用 bind_message 的全部身份语义）并回写
        input.task_id；已附着 → 直接返回该 task（同一动机输入的
        连续 effectful call 不重复 bump revision）。"""
        row = self._conn.execute(
            "SELECT * FROM user_inputs WHERE mission_id = ? AND "
            "session_ref = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (mission_id, session_ref),
        ).fetchone()
        if row is None and not explicit_goal:
            raise ValueError(
                "INPUT_MOTIVATION_MISSING: 无持久化输入——effectful "
                "call 缺少动机输入，不猜目标"
            )
        if row is not None and row["task_id"]:
            task = self.get_task(str(row["task_id"]))
            if task is not None:
                return {
                    "task_id": str(task["task_id"]),
                    "revision": int(task["active_revision"]),
                    "created_task": False,
                    "replayed": False,
                    "workspace_path": str(task["workspace_path"]),
                    "state": str(task["state"]),
                }
        bound = self.bind_message(
            mission_id=mission_id,
            session_ref=session_ref,
            backend_native_id=backend_native_id,
            message_id=(
                str(row["message_id"]) if row is not None
                else f"goal_{new_id('msg')}"
            ),
            text=explicit_goal or str(row["text"]),
            cwd=cwd, mode=mode, body_id=body_id,
            # 任务 workspace = 调用方解析的工作根（与 pi.task.bind
            # 同一语义——不传则回落 home/tasks/<id>/workspace）。
            workspace_root=cwd,
            # /newtask：该输入被显式要求开新任务。
            force_new=(
                row is not None and row["delivery_state"] == "FORCE_NEW"
            ),
        )
        if row is not None:
            self._conn.execute(
                "UPDATE user_inputs SET task_id = ? WHERE message_id = ?",
                (str(bound["task_id"]), str(row["message_id"])),
            )
        return bound

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
        # /newtask（FORCE_NEW）必须最先判定——先于 SUCCEEDED 重开：
        # 否则"显式开新任务"被旧任务的重开语义吞掉（h2 旅程实证：
        # FORCE_NEW 输入落成旧任务 revision 3）。
        if active is not None and force_new:
            self._conn.execute(
                "UPDATE task_session_bindings SET active = 0 "
                "WHERE task_id = ? AND session_ref = ?",
                (active["task_id"], session_ref),
            )
            active = None
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
            ensure_run(self._home, task_id, revision)  # WP-8
            # P1-C1：修订也冻结新 revision 的 TaskSpecV2（goal_delta
            # 是修订文本——旧 revision 的 spec 不被覆盖）。
            from rosclaw.task_kernel.task_spec import compile_task_spec

            revised_spec = compile_task_spec(
                task_id=task_id, revision=revision, goal_text=text,
                body_id=str(active["body_id"] or ""),
                mode=str(active["mode"] or "SIMULATION"),
                acceptance_spec_id="",
                language=str(active["locale"] or "") if active["locale"] != "auto" else "",
            )
            self._conn.execute(
                "INSERT INTO task_revisions (task_id, revision, "
                "user_message_id, goal_delta, task_spec_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, revision, message_id, text,
                 revised_spec.model_dump_json(), now),
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
        if active is not None:
            revision = int(active["active_revision"]) + 1
            ensure_run(self._home, str(active["task_id"]), revision)  # WP-8
            # P1-C1 + R0-1.5（金丝雀实证）：普通修订分支也必须冻结
            # 新 revision 的 TaskSpecV2——此前只有"SUCCEEDED 重开"
            # 分支冻结，活跃任务修订的 spec 为空（TaskExecutionService
            # 无法路由——TASK_SPEC_MISSING）。
            from rosclaw.task_kernel.task_spec import compile_task_spec

            revised_spec = compile_task_spec(
                task_id=str(active["task_id"]), revision=revision,
                goal_text=text,
                body_id=str(active["body_id"] or ""),
                mode=str(active["mode"] or "SIMULATION"),
                acceptance_spec_id="",
                language=(
                    str(active["locale"] or "") if active["locale"] != "auto" else ""
                ),
            )
            self._conn.execute(
                "INSERT INTO task_revisions (task_id, revision, "
                "user_message_id, goal_delta, task_spec_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(active["task_id"]), revision, message_id, text,
                 revised_spec.model_dump_json(), now),
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
        # WP-8：运行目录四区（scratch/outputs/evidence/logs）——
        # 项目源码不再当任务垃圾场。
        ensure_run(self._home, task_id, 1)
        self._conn.execute(
            "INSERT INTO tasks (task_id, mission_id, root_goal, mode, body_id, "
            "workspace_path, state, active_revision, locale, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING', 1, ?, ?, ?)",
            (task_id, mission_id, text, mode, body_id, str(workspace),
             locale, now, now),
        )
        # P1-C1：TaskSpecV2 随 revision 1 冻结（intent/subjects/
        # constraints 工单——root_goal 之外的契约视图）。
        from rosclaw.task_kernel.task_spec import compile_task_spec

        task_spec = compile_task_spec(
            task_id=task_id, revision=1, goal_text=text,
            body_id=body_id, mode=mode, acceptance_spec_id="",
            language=locale if locale != "auto" else "",
        )
        self._conn.execute(
            "INSERT INTO task_revisions (task_id, revision, user_message_id, "
            "goal_delta, task_spec_json, created_at) VALUES (?, 1, ?, ?, ?, ?)",
            (task_id, message_id, text,
             task_spec.model_dump_json(), now),
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
        if row is None:
            return None
        task = dict(row)
        # WP-8：当前 revision 的运行目录（模型/界面可查）。
        task["run_dir"] = str(
            run_dir(self._home, task_id, int(task["active_revision"]))
        )
        return task

    def active_task_for_session(
        self, mission_id: str, session_ref: str
    ) -> dict | None:
        """WP-8：session 的活跃 task + 运行目录/四区（pi.context
        接线——模型每轮知道写哪里）。"""
        row = self._conn.execute(
            "SELECT t.* FROM tasks t JOIN task_session_bindings b "
            "ON b.task_id = t.task_id "
            "WHERE t.mission_id = ? AND b.session_ref = ? AND b.active = 1 "
            "AND b.role = 'primary' "
            "ORDER BY t.created_at DESC LIMIT 1",
            (mission_id, session_ref),
        ).fetchone()
        if row is None:
            return None
        task = self.get_task(str(row["task_id"]))
        assert task is not None
        run = ensure_run(
            self._home, str(task["task_id"]), int(task["active_revision"])
        )
        return {
            "task_id": str(task["task_id"]),
            "state": str(task["state"]),
            "revision": int(task["active_revision"]),
            "run_dir": run["run_dir"],
            "zones": run["zones"],
        }

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

    def latest_task_for(self, mission_id: str, session_ref: str) -> dict | None:
        """P0-C：session 最近 task（含刚终态）——/activity /logs
        /artifacts 展示最近任务的活动账本（终态不抹掉历史）。"""
        row = self._conn.execute(
            "SELECT t.* FROM tasks t JOIN task_session_bindings b "
            "ON b.task_id = t.task_id "
            "WHERE t.mission_id = ? AND b.session_ref = ? "
            "ORDER BY t.created_at DESC LIMIT 1",
            (mission_id, session_ref),
        ).fetchone()
        return dict(row) if row else None

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
        # WP-8：运行区纪律——scratch 是草稿区，不得登记为交付物；
        # outputs/evidence 登记记录 zone（交付/证据可归因）。
        zone = zone_of(
            self._home, task_id, int(task["active_revision"]), file
        )
        if zone == "scratch":
            raise ValueError(
                f"SCRATCH_NOT_DELIVERABLE: {file} 在 scratch 草稿区——"
                "草稿不是交付物；请把最终交付物写入 outputs/ 再登记"
            )
        # P0-E：evidence 区是 kernel-only——模型不能自己写
        # verify_*.json 冒充受信证据（producer 身份来自登记调用方，
        # 不接受文件内容自述）。
        if zone == "evidence" and not producer.startswith("kernel:"):
            raise ValueError(
                f"EVIDENCE_KERNEL_ONLY: {file} 在 evidence 受信区——"
                "该目录只接受内核管道登记；模型的交付物请写 outputs/"
            )
        # P0-E：幂等 upsert——同 task 同内容（sha256）返回既有
        # ArtifactRef（同文件登记十次仍只有一个引用；内容变化才
        # 产生新引用——内容寻址，不是路径寻址）。
        digest = hashlib.sha256(content).hexdigest()
        existing = self._conn.execute(
            "SELECT * FROM artifacts WHERE task_id = ? AND sha256 = ?",
            (task_id, digest),
        ).fetchone()
        if existing is not None:
            record = dict(existing)
            record["metadata_json"] = str(existing["metadata_json"])
            record["idempotent_replay"] = True
            return record
        artifact_id = new_id("art")
        now = datetime.now(UTC).isoformat()
        record = {
            "artifact_id": artifact_id,
            "task_id": task_id,
            "path": str(file),
            "media_type": media_type,
            "sha256": digest,
            "size_bytes": len(content),
        }
        # N4.1：模型自产证据标 EXPERIMENTAL——通过 qualification 前
        # 不当正式能力证据（N 调整方案 §二）。
        meta = dict(metadata or {})
        if zone:
            meta["zone"] = zone
            record["zone"] = zone
        if producer.startswith("model:"):
            meta.setdefault("evidence_tier", "EXPERIMENTAL")
        # WP-4：血缘图——带 render receipt 的交付物在登记时推导并
        # 打戳（digest 从 receipt 文件实算，task/revision 打当前
        # 活跃值；不是调用方自述）。
        lineage = meta.get("lineage")
        if lineage is not None and lineage.get("kind") == "preview_2d":
            # 2D 预演（COMMAND_REPLAY 可视化）——血缘到 trace 即可，
            # 无 render receipt；登记时打 task/revision 戳。
            lineage["task_id"] = task_id
            lineage["revision"] = int(task["active_revision"])
            meta["lineage"] = lineage
        elif lineage is not None:
            receipt_path = Path(str(lineage.get("render_receipt_path", "")))
            if not receipt_path.exists():
                raise ValueError(
                    f"LINEAGE_UNREADABLE: render receipt 不存在: "
                    f"{receipt_path}"
                )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            lineage["render_receipt_digest"] = "sha256:" + hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()
            lineage["input_trace_digest"] = str(
                receipt.get("input_trace_digest", "")
            )
            lineage["trace_id"] = str(
                lineage.get("trace_id") or receipt_path.parent.name
            )
            lineage["task_id"] = task_id
            lineage["revision"] = int(task["active_revision"])
            meta["lineage"] = lineage
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
        # 0901 P0-2：登记即投影——outputs/ 是即时投影视图（不等
        # coordinator PASS；PARTIAL/FAIL 任务的已产出交付物也必须
        # 可打开）。投影失败绝不阻断登记（投影是视图不是真相）。
        from rosclaw.task_kernel.projection import project_deliverables

        try:
            project_deliverables(self, task_id)
        except Exception:  # noqa: BLE001 - 投影是视图，失败不阻断登记
            import logging

            logging.getLogger("rosclaw.projection").warning(
                "project-on-register failed for %s", task_id, exc_info=True,
            )
        return record

    def finish_task(
        self, *, task_id: str, summary: str, artifact_ids: list[str],
        grade: str = "", tracking_max_error_m: float | None = None,
    ) -> dict[str, Any]:
        """FinishRequest（§12.1）：验收真跑 → SUCCEEDED / REPAIR_REQUIRED。
        终态幂等（重放不重复验证、不覆盖——返回原 receipt id）。

        PR-N0 熔断：
        - 验收条件只读任务创建时冻结值（模型收尾不得改规则）；
        - 机器人行为任务（body_id 非空）必须含受信管道证据
          （kernel 内部登记的产物）——模型自产证据不算数。

        P0-5（0827 审计）：grade（PASS / PASS_NEAR_LIMIT——≥90%
        阈值占用的诚实分级）与 tracking_max_error_m 由调用方（受信
        管道）申报并随验收行持久化——误差事实留账，Coordinator
        呈现分级而不是一律 PASS。
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
        # 0901 P0-5：具身任务的终态权威——手拼低层 capability（裸
        # compute/observe/simulate 直调）的产物不得发布终态。实证：
        # 0901 第二轮模型手拼 simulate→trace 落账，turn_end 的
        # consider 直接把任务收成 PASS·DELIVERED（渲染/验证还在
        # 后面跑）。
        # 判别边界（0827 全量回归实证）：body_id 非空只是会话绑定
        # （chat 默认绑 body——纯写文件任务也带 body_id，不得误伤）；
        # 真正的"具身实效"是任务实际触碰了具身面：embodiment 工具
        # 事件（task.tool_used）或 kernel 能力/仿真产物
        # （kernel:capability:*/kernel:sim* producer）。
        # 受信执行证据三选一：PlanGraph plan.node 事件（确定性链）、
        # Operator 链 COMPLETED txn、或 kernel 核验血缘的产物
        # （登记时 kernel 计算的 receipt/trace digest——受信 sim
        # 管道跑过的密码学证明）。
        capability_touched = self._conn.execute(
            "SELECT COUNT(*) AS n FROM artifacts WHERE task_id = ? AND "
            "(producer LIKE 'kernel:capability:%' OR "
            "producer LIKE 'kernel:sim%')",
            (task_id,),
        ).fetchone()
        embodied_in_effect = bool(task.get("body_id")) and (
            embodiment_used or int(capability_touched["n"]) > 0
        )
        authority_failure: list[str] = []
        if embodied_in_effect:
            plan_events = self._conn.execute(
                "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
                "AND event_type LIKE 'plan.node_%'",
                (task_id,),
            ).fetchone()
            binding = self._conn.execute(
                "SELECT session_ref FROM task_session_bindings "
                "WHERE task_id = ? AND role = 'primary' LIMIT 1",
                (task_id,),
            ).fetchone()
            receipts = self._conn.execute(
                "SELECT COUNT(*) AS n FROM action_txns "
                "WHERE mission_id = ? AND pi_session_id = ? "
                "AND state = 'COMPLETED'",
                (str(task["mission_id"]),
                 str(binding["session_ref"]) if binding else ""),
            ).fetchone()
            lineage_rows = self._conn.execute(
                "SELECT metadata_json FROM artifacts WHERE task_id = ? "
                "AND metadata_json LIKE '%\"lineage\"%'",
                (task_id,),
            ).fetchall()
            # 血缘两态都算受信执行证据（登记时 kernel 打戳/核验）：
            # render 血缘（receipt digest 实算）与 preview_2d 血缘
            # （trace 引用 + task/revision 打戳——WP-4 产品路径）。
            # 裸手拼产物（无 lineage 元数据）两者皆无。
            lineage_present = any(
                ((json.loads(str(r["metadata_json"]) or "{}")
                  .get("lineage") or {}).get("render_receipt_digest"))
                or (
                    (json.loads(str(r["metadata_json"]) or "{}")
                     .get("lineage") or {}).get("kind") == "preview_2d"
                    and (json.loads(str(r["metadata_json"]) or "{}")
                         .get("lineage") or {}).get("trace_id")
                )
                for r in lineage_rows
            )
            if (not int(plan_events["n"]) and not int(receipts["n"])
                    and not lineage_present):
                authority_failure.append(
                    "PLAN_AUTHORITY_MISSING: 具身任务缺受信执行证据"
                    "（PlanGraph plan.node 事件 / Operator 链 COMPLETED "
                    "txn / kernel 核验血缘产物 三选一）——手拼低层 "
                    "capability 不得发布终态"
                )
        if embodiment_used:
            # P0-G：canonical alias 唯一权威换算（不再手写前缀）。
            from rosclaw.cognition.alias import canonical_resource_id

            robot_id = canonical_resource_id(
                str(task["body_id"])
            ).removeprefix("robot:")
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
        # WP-4：Evidence Graph 遍历——行为任务的媒体交付物（受信
        # 声明）必须有完整血缘：receipt digest 实算一致、renderer
        # 输入确实是该 trace、revision 不跨（不拼接）、模型手工登记
        # 不得升级为 TRUSTED。
        graph_failures: list[str] = []
        if embodiment_used:
            for art in artifacts:
                producer = str(art.get("producer") or "")
                media = str(art.get("media_type") or "")
                if not media.startswith("image/"):
                    continue
                meta = json.loads(str(art.get("metadata_json") or "{}"))
                lineage = meta.get("lineage") or {}
                if producer.startswith("model:"):
                    # EXPERIMENTAL 媒体：不当受信证据（N0 已挡）——
                    # 这里只标注血缘缺失，不再升级。
                    if not lineage:
                        graph_failures.append(
                            f"LINEAGE_MISSING: 模型自产媒体 "
                            f"{Path(str(art['path'])).name} 无血缘——"
                            "EXPERIMENTAL，不当受信交付证据"
                        )
                    continue
                # 受信声明（kernel:*）媒体必须血缘完整；2D 预演
                # （preview_2d）血缘到 trace 即可——不当场景渲染证据，
                # 诚实降级标注。
                receipt_digest = str(lineage.get("render_receipt_digest", ""))
                trace_id = str(lineage.get("trace_id", ""))
                if lineage.get("kind") == "preview_2d" and trace_id:
                    receipt_digest = "preview"  # 免 receipt 要求
                if not receipt_digest or not trace_id:
                    graph_failures.append(
                        f"LINEAGE_MISSING: 受信声明媒体 "
                        f"{Path(str(art['path'])).name} 缺 render "
                        "receipt/trace 血缘"
                    )
                    continue
                if int(lineage.get("revision", -1)) != int(
                    task["active_revision"]
                ):
                    graph_failures.append(
                        f"REVISION_SPLICE: 血缘 revision "
                        f"{lineage.get('revision')} != 活跃 "
                        f"{task['active_revision']}——跨 revision 拼接"
                    )
                # 证据产生时间必须 ≥ 当前 revision 开始时间（r1 跑的
                # trace 不能服务 r2）。
                rev_row2 = self._conn.execute(
                    "SELECT created_at FROM task_revisions WHERE task_id = ? "
                    "AND revision = ?",
                    (task_id, int(task["active_revision"])),
                ).fetchone()
                trace_json2 = (
                    self._home / "sim" / "traces"
                    / str(lineage.get("trace_id", "")) / "trace.json"
                )
                if rev_row2 and trace_json2.exists():
                    from datetime import datetime as _dt

                    rev_start = _dt.fromisoformat(
                        str(rev_row2["created_at"])
                    ).timestamp()
                    produced_at = trace_json2.stat().st_mtime
                    if produced_at < rev_start - 1.0:  # 1s 时钟宽容
                        graph_failures.append(
                            "REVISION_SPLICE: trace 证据产生时间早于当前 "
                            "revision 开始——旧 revision 证据不得复用"
                        )
                trace_json = (
                    self._home / "sim" / "traces" / trace_id / "trace.json"
                )
                if not trace_json.exists():
                    graph_failures.append(
                        f"LINEAGE_TRACE_MISSING: trace {trace_id} 不存在"
                    )
                    continue
                # renderer 输入 digest 校验只对场景渲染血缘（receipt
                # 类）；preview_2d 是命令回放可视化，无此语义。
                if lineage.get("kind") != "preview_2d":
                    trace_digest = "sha256:" + hashlib.sha256(
                        trace_json.read_bytes()
                    ).hexdigest()
                    if str(lineage.get("input_trace_digest", "")) != trace_digest:
                        graph_failures.append(
                            "LINEAGE_DIGEST_MISMATCH: renderer 输入 digest 与 "
                            "该 trace 实际内容不符——renderer 吃的不是这条 trace"
                        )
        provenance_failures += graph_failures
        provenance_failures += authority_failure
        # R0-2（0826 体验审计 §5.R0-2）：spec 冻结的 required
        # deliverables 按 kind 核验全量产物账本——任务成功 ≠ 用户
        # 请求成功（2D 预览不满足 scene_video——kind 分野是硬边界）。
        spec = self.get_task_spec(task_id)
        spec_deliverables = (spec or {}).get("deliverables") or []
        if spec_deliverables:
            from rosclaw.task_kernel.deliverables import deliverable_verdict

            ledger = [
                dict(r)
                for r in self._conn.execute(
                    "SELECT * FROM artifacts WHERE task_id = ?", (task_id,),
                ).fetchall()
            ]
            dv = deliverable_verdict(spec_deliverables, ledger)
            for kind in dv["missing"]:
                provenance_failures.append(
                    f"DELIVERABLE_MISSING: required 交付物 {kind} 未在产物"
                    "账本（按 kind 匹配——2D 预览不满足场景视频）"
                )
        trusted_present = any(
            str(a.get("producer") or "").startswith("kernel:")
            and not str(a.get("media_type") or "").startswith("image/")
            or (
                str(a.get("producer") or "").startswith("kernel:")
                and json.loads(str(a.get("metadata_json") or "{}")).get("lineage")
            )
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
            # P0-5：误差事实与分级随验收行持久化（checks_json 是
            # 审计面——误差/分级不回填就无从复核"接近阈值"）。
            checks_payload: dict[str, Any] = {"checks": verdict["checks"]}
            if grade:
                checks_payload["grade"] = grade
            if tracking_max_error_m is not None:
                checks_payload["tracking_max_error_m"] = float(
                    tracking_max_error_m
                )
            self._conn.execute(
                "INSERT INTO verifications (verification_id, task_id, "
                "revision, status, checks_json, evidence_json, created_at) "
                "VALUES (?, ?, ?, 'PASS', ?, ?, ?)",
                (verification_id, task_id, int(task["active_revision"]),
                 json.dumps(checks_payload, ensure_ascii=False),
                 json.dumps({"artifact_ids": artifact_ids},
                            ensure_ascii=False),
                 now),
            )
            self._emit(task_id, "verification.completed",
                       {"verification_id": verification_id, "status": "PASS",
                        "checks": verdict["checks"],
                        **({"grade": grade} if grade else {})})
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
        模型临时传入的新规则。PR-N8：同时编译并冻结
        AcceptanceSpecV2（来源归因在案）。"""
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id!r}")
        revision = int(task["active_revision"])
        from rosclaw.task_kernel.acceptance import compile_acceptance

        # 模型经 rosclaw_task_finish 之外不设验收——kernel 设置的
        # acceptance 是任务级输入（task_default 源）。
        spec = compile_acceptance(
            task_id=task_id, revision=revision, task_default=acceptance,
        )
        self._conn.execute(
            "UPDATE task_revisions SET acceptance_json = ?, "
            "acceptance_spec_json = ? "
            "WHERE task_id = ? AND revision = ?",
            (json.dumps(acceptance, ensure_ascii=False),
             json.dumps(spec.to_canonical_dict(), ensure_ascii=False),
             task_id, revision),
        )
        self._emit(task_id, "acceptance.frozen",
                   {"revision": revision, "spec_id": spec.spec_id})

    def get_acceptance_spec(self, task_id: str) -> dict | None:
        """当前活跃 revision 的冻结 AcceptanceSpecV2（dict 视图）。"""
        task = self.get_task(task_id)
        if task is None:
            return None
        row = self._conn.execute(
            "SELECT acceptance_spec_json FROM task_revisions "
            "WHERE task_id = ? AND revision = ?",
            (task_id, int(task["active_revision"])),
        ).fetchone()
        if row is None or not row["acceptance_spec_json"]:
            return None
        return json.loads(row["acceptance_spec_json"])

    def get_task_spec(self, task_id: str) -> dict | None:
        """当前活跃 revision 的冻结 TaskSpecV2（dict 视图）。"""
        task = self.get_task(task_id)
        if task is None:
            return None
        row = self._conn.execute(
            "SELECT task_spec_json FROM task_revisions "
            "WHERE task_id = ? AND revision = ?",
            (task_id, int(task["active_revision"])),
        ).fetchone()
        if row is None or not row["task_spec_json"]:
            return None
        return json.loads(row["task_spec_json"])

    def artifact_refs_for(self, task_id: str) -> list[dict[str, Any]]:
        """R0-4（0826 体验审计 §5.R0-4）：用户可见 ArtifactRef 视图
        ——id/kind/media_type/size/digest/open_command。

        "数据库里有文件"不等于交付成功：交付面（ToolResult/
        TaskOutcome/CLI）只认这份带 open_command 的视图。
        """
        from rosclaw.task_kernel.deliverables import artifact_delivery_kind

        rows = self._conn.execute(
            "SELECT * FROM artifacts WHERE task_id = ? ORDER BY created_at",
            (task_id,),
        ).fetchall()
        refs: list[dict[str, Any]] = []
        for row in rows:
            artifact = dict(row)
            artifact_id = str(artifact["artifact_id"])
            raw_digest = str(artifact["sha256"])
            refs.append({
                "artifact_id": artifact_id,
                "kind": artifact_delivery_kind(artifact),
                "media_type": str(artifact["media_type"]),
                "path": str(artifact["path"]),
                "size_bytes": int(artifact["size_bytes"]),
                "digest": (
                    raw_digest
                    if raw_digest.startswith("sha256:")
                    else f"sha256:{raw_digest}"
                ),
                "producer": str(artifact.get("producer") or ""),
                "open_command": f"rosclaw artifact open {artifact_id}",
            })
        return refs

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
