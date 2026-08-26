"""Task Coordinator（P0-D，0824 总纲 §7.4/§8/§19.P0-D）。

模型不再手动收尾（task_finish/task_blocked/artifact_register 退出
模型工具面）——Coordinator 自动事务：

1. 收集该 revision 的 artifacts（capability 自动登记 + 交付物）；
2. 运行 Verifier（与 finish_task 同一验收事实源——不重写规则）；
3. 生成 TaskOutcomeV2（六维：lifecycle/execution/verification/
   delivery/user_acceptance/evidence）；
4. PASS → 终态 + kernel-owned outcome 落库（deterministic replay——
   重复 consider 返回同一 outcome）；
5. FAIL → RepairDirective（失败 criterion + 错误指纹）；同指纹
   再现 → WAITING_INPUT（不继续烧 token）；
6. 媒体/交付类失败（RENDER_/MEDIA_）= execution SUCCEEDED +
   delivery NEEDS_REPAIR——lifecycle 不关闭（BLOCKED 不再是
   万能终态）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from rosclaw.contracts.common import new_id
from rosclaw.task_kernel.service import TaskKernel

#: 媒体/交付类失败前缀（execution 已成功，只 delivery 待修）。
_DELIVERY_FAILURE_PREFIXES = ("RENDER_", "MEDIA_", "DELIVERABLE_")

#: 目标声明媒体交付物的关键词（通用——不是形状特例：任何要
#: 视频/GIF/MP4/动画的目标都要求媒体交付证据）。
_GOAL_MEDIA_MARKERS = ("视频", "动画", "gif", "mp4", "video")


def _goal_requires_media(goal: str) -> bool:
    lowered = goal.lower()
    return any(marker in lowered for marker in _GOAL_MEDIA_MARKERS)

VerifyRunner = Callable[[dict, list[dict], dict], dict[str, Any]]


def _default_verify_runner(kernel: TaskKernel) -> VerifyRunner:
    """与 finish_task 同一验收事实源（全部产物 + 冻结验收）。"""

    def _run(task: dict, artifacts: list[dict], frozen: dict) -> dict[str, Any]:
        return kernel.finish_task(
            task_id=str(task["task_id"]),
            summary=(
                f"Coordinator 自动验收（{len(artifacts)} 项产物）"
            ),
            artifact_ids=[str(a["artifact_id"]) for a in artifacts],
        )

    return _run


class TaskCoordinator:
    """任务自动收尾权威（agentd 进程内，与 kernel 同连接）。"""

    def __init__(
        self,
        kernel: TaskKernel,
        verify_runner: VerifyRunner | None = None,
    ) -> None:
        self._kernel = kernel
        self._conn: sqlite3.Connection = kernel._conn
        self._verify = verify_runner or _default_verify_runner(kernel)

    def consider(self, task_id: str) -> dict[str, Any] | None:
        """评估 task 是否可收尾；返回 TaskOutcomeV2（幂等）。

        - 已有 outcome（含终态重放）→ 原样返回（deterministic）；
        - 无 artifacts 且无冻结验收 → None（任务还在进行——
          Coordinator 不替模型宣布开始）。"""
        task = self._kernel.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id!r}")
        revision = int(task["active_revision"])
        prior = self._conn.execute(
            "SELECT outcome_json FROM task_outcomes WHERE task_id = ? "
            "AND revision = ?",
            (task_id, revision),
        ).fetchone()
        if prior is not None:
            return json.loads(str(prior["outcome_json"]))
        artifacts = [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM artifacts WHERE task_id = ?", (task_id,)
            ).fetchall()
        ]
        # 完成信号：有产物或有冻结验收——两者皆无说明任务还在
        # 进行（Coordinator 不替模型宣布开始）。
        if not artifacts and not self._kernel.get_acceptance_spec(task_id):
            return None
        frozen = self._kernel.get_acceptance_spec(task_id) or {}
        verdict = self._verify(task, artifacts, frozen)
        failures = list(verdict.get("failures") or [])
        # 目标声明媒体交付物（视频/GIF/MP4/动画）但零媒体交付——
        # 执行成功 ≠ 交付成功（0824 金丝雀实证：模型只 rollout 不
        # 渲染就结束，trace 内部件不是用户要的视频）。
        # R0-2：spec 冻结 deliverables 时以 spec 为准（finish_task
        # 已把 DELIVERABLE_MISSING 放进 failures）——启发式只做
        # 无 spec 任务的回落。
        spec = self._kernel.get_task_spec(task_id) or {}
        spec_deliverables = list(spec.get("deliverables") or [])
        if not spec_deliverables and _goal_requires_media(
            str(task.get("root_goal") or "")
        ):
            has_media = any(
                str(a.get("media_type") or "").startswith(("image/", "video/"))
                for a in artifacts
            )
            if not has_media:
                failures.append(
                    "MEDIA_DELIVERABLE_MISSING: 目标要求视频/GIF/MP4"
                    "交付物但无媒体产物（image/* 或 video/*）——"
                    "渲染步骤未完成"
                )
        passed = (
            verdict.get("status") in ("PASS", "SUCCEEDED")
            or str(verdict.get("status", "")) == "SUCCEEDED"
        ) and not failures
        now = datetime.now(UTC).isoformat()
        if passed:
            outcome = self._build_outcome(
                task, revision, artifacts,
                lifecycle="COMPLETED",
                execution="SUCCEEDED",
                verification="PASS",
                delivery="DELIVERED",
                created_at=now,
            )
            self._store_outcome(task_id, revision, outcome, now)
            return outcome
        # FAIL：区分 delivery 待修（媒体/交付类）与 verification 失败。
        delivery_only = bool(failures) and all(
            f.startswith(_DELIVERY_FAILURE_PREFIXES) for f in failures
        )
        if delivery_only:
            # R0-2：required deliverables 缺失 = 运动执行 PASS 但
            # 用户请求未完整满足——verification PARTIAL（不是整体
            # VERIFIED）；delivery 按已满足 kind 分 MISSING/PARTIAL。
            deliverable_missing = any(
                f.startswith("DELIVERABLE_MISSING") for f in failures
            )
            delivery = "NEEDS_REPAIR"
            verification = "FAIL"
            if deliverable_missing:
                from rosclaw.task_kernel.deliverables import (
                    deliverable_verdict,
                )

                dv = deliverable_verdict(spec_deliverables, artifacts)
                verification = "PARTIAL"
                delivery = "PARTIAL" if dv["satisfied"] else "MISSING"
            outcome = self._build_outcome(
                task, revision, artifacts,
                lifecycle="ACTIVE",
                execution="SUCCEEDED",
                verification=verification,
                delivery=delivery,
                created_at=now,
            )
            directive = self._repair_directive(task_id, revision, failures, now)
            outcome["repair_directive"] = directive
            # 瞬态不落库——修复后下一次 consider 重算。
            return outcome
        # verification FAIL：RepairDirective + 同指纹再现 →
        # WAITING_INPUT（§8.3：不继续烧 token）。
        fingerprint = hashlib.sha256(
            json.dumps(sorted(failures), ensure_ascii=False).encode()
        ).hexdigest()[:24]
        directive = self._repair_directive(
            task_id, revision, failures, now, fingerprint=fingerprint
        )
        seen = self._conn.execute(
            "SELECT COUNT(*) AS n FROM task_repairs WHERE task_id = ? "
            "AND fingerprint = ?",
            (task_id, fingerprint),
        ).fetchone()
        outcome = self._build_outcome(
            task, revision, artifacts,
            lifecycle="ACTIVE",
            execution="SUCCEEDED" if artifacts else "RUNNING",
            verification="FAIL",
            delivery="PARTIAL" if artifacts else "NONE",
            created_at=now,
        )
        outcome["repair_directive"] = directive
        if int(seen["n"]) > 1:
            # 同指纹再现 → WAITING_INPUT（持久化——重放不再重试）。
            self._conn.execute(
                "UPDATE tasks SET state = 'WAITING_INPUT', updated_at = ? "
                "WHERE task_id = ?",
                (now, task_id),
            )
            self._store_outcome(task_id, revision, outcome, now)
        return outcome

    # --------------------------------------------------------------
    def _build_outcome(
        self, task: dict, revision: int, artifacts: list[dict], *,
        lifecycle: str, execution: str, verification: str,
        delivery: str, created_at: str,
    ) -> dict[str, Any]:
        trust = "EXPERIMENTAL"
        if any(
            str(a.get("producer") or "").startswith("kernel:")
            for a in artifacts
        ):
            trust = "TRUSTED"
        accepted = bool(task.get("user_accepted_at"))
        # R0-5（0826 体验审计 §5.R0-5）：证据等级按产物事实拆分
        # （GEOMETRY_PLAN/KINEMATIC_TRACKING/DYNAMIC_ROLLOUT/
        # CONTACT_SIMULATION/SCENE_RENDER/REAL_RECEIPT）——产物
        # 元数据 evidence.levels 是权威（受信管道打戳），scene_3d
        # kind 补 SCENE_RENDER。
        from rosclaw.task_kernel.deliverables import artifact_delivery_kind

        levels: list[str] = []
        seen_levels: set[str] = set()
        for a in artifacts:
            meta = json.loads(str(a.get("metadata_json") or "{}"))
            for level in (meta.get("evidence") or {}).get("levels") or []:
                if level not in seen_levels:
                    seen_levels.add(level)
                    levels.append(str(level))
        kinds = {artifact_delivery_kind(a) for a in artifacts}
        if "scene_3d" in kinds and "SCENE_RENDER" not in seen_levels:
            levels.append("SCENE_RENDER")
        if str(task.get("mode") or "") == "REAL":
            levels.append("REAL_RECEIPT")
        return {
            "schema_version": "rosclaw.task_outcome.v2",
            "task_id": str(task["task_id"]),
            "revision": revision,
            "lifecycle": lifecycle,
            "execution": execution,
            "verification": verification,
            "delivery": delivery,
            "user_acceptance": "ACCEPTED" if accepted else "UNSEEN",
            "evidence": {
                "domain": str(task.get("mode") or "SIMULATION"),
                "trust": trust,
                "levels": levels,
            },
            # R0-4：用户可见交付视图（id/kind/media/size/digest/
            # open_command）——数据库里有文件 ≠ 交付成功。
            "artifact_refs": self._kernel.artifact_refs_for(
                str(task["task_id"])
            ),
            "blocked_on": [],
            "created_at": created_at,
        }

    def _repair_directive(
        self, task_id: str, revision: int, failures: list[str], now: str,
        *, fingerprint: str = "",
    ) -> dict[str, Any]:
        fingerprint = fingerprint or hashlib.sha256(
            json.dumps(sorted(failures), ensure_ascii=False).encode()
        ).hexdigest()[:24]
        directive = {
            "criterion": failures[0] if failures else "",
            "repairable": True,
            "fingerprint": fingerprint,
            "failures": failures,
        }
        self._conn.execute(
            "INSERT INTO task_repairs (repair_id, task_id, revision, "
            "fingerprint, directive_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (new_id("rep"), task_id, revision, fingerprint,
             json.dumps(directive, ensure_ascii=False), now),
        )
        return directive

    def _store_outcome(
        self, task_id: str, revision: int, outcome: dict, now: str
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO task_outcomes (outcome_id, task_id, "
            "revision, outcome_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (new_id("out"), task_id, revision,
             json.dumps(outcome, ensure_ascii=False, sort_keys=True), now),
        )


__all__ = ["TaskCoordinator"]
