"""R0-9 红测试（0826 体验审计 §5.R0-9/§7.4）：重试预算收回
Harness + 金丝雀真实生产断言。

真实事故（0826 体验旅程）：renderer 连续三次裸 JSONDecodeError
——基础设施故障没有重试预算语义（ErrorEnvelope 无 scope/
attempt_budget/recovery_action）；金丝雀只查 DB 文件存在，不查
生产路径/用户可见交付/最终回答一致性。

断言：
1. ErrorEnvelope 语义：基础设施/配置/确定性错误 → details 带
   scope + attempt_budget=0 + recovery_action（模型重试预算为
   0）；transient 且状态变化可重试；
2. 金丝雀断言函数（合成 home 可测）：
   a. 生产 PlanGraph——plan.node_* 事件存在且 5 节点完整；
   b. 用户可见交付——outcome.artifact_refs 含 GIF+MP4 且每条
      带 open_command；scene_video 要求的 mp4 必须是 scene_3d
      kind（2D 预览不能冒充）；
   c. 模型行为预算——具身入口调用 ≤1、bash=0、手工 finish=0；
   d. 证据等级——outcome.evidence.levels 含 GEOMETRY_PLAN/
      KINEMATIC_TRACKING/DYNAMIC_ROLLOUT/SCENE_RENDER；
   e. 屏幕无 raw JSON/无 Unreachable+成功并存；
   f. 最终回答一致性——outcome PARTIAL 时回答不得宣称完整完成。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


class TestErrorEnvelopeBudget:
    async def test_infra_error_zero_budget_with_recovery(
        self, tmp_path: Path
    ) -> None:
        """基础设施错误（RENDER_*）→ details.scope=infrastructure +
        attempt_budget=0 + recovery_action——模型重试预算为 0。"""
        service, mission = await _setup(tmp_path)
        await service._ensure_mcp_discovered()
        lease = await _issue_lease(service, mission)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_compute", mission=mission.mission_id,
                idem="r09_infra", lease=lease,
                arguments={
                    "capability_id": "simulation_render_scene",
                    "arguments": {"trace_id": "trace_nonexistent"},
                },
            )
        )
        assert not result.ok
        details = result.details or {}
        assert details.get("scope") == "infrastructure", details
        assert details.get("attempt_budget") == 0, details
        assert details.get("recovery_action"), details
        await service.close()

    async def test_transient_context_retryable(self, tmp_path: Path) -> None:
        """transient 错误码（CONTEXT_* 类）→ scope=transient +
        attempt_budget=1 + retry_after_condition（可安全重试）。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import (
            _error_envelope_details,
        )

        details = _error_envelope_details("CONTEXT_NOT_FRESH")
        assert details["scope"] == "transient"
        assert details["attempt_budget"] == 1
        assert details["retry_after_condition"]
        # 确定性错误预算 0。
        det = _error_envelope_details("UNKNOWN_CAPABILITY")
        assert det["attempt_budget"] == 0
        assert det["scope"] == "deterministic"


def _fixture_home(tmp_path: Path) -> Path:
    """合成金丝雀 home：missions.db（生产路径证据）+ 屏幕文本。"""
    from rosclaw.storage.migrations import MigrationRunner

    home = tmp_path / "home"
    (home / "agentd").mkdir(parents=True)
    conn = sqlite3.connect(home / "agentd" / "missions.db")
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    now = "2026-08-26T00:00:00+00:00"
    conn.execute(
        "INSERT INTO tasks (task_id, mission_id, root_goal, mode, body_id, "
        "state, active_revision, workspace_path, created_at, updated_at) "
        "VALUES ('task_1', 'm1', '画五角星并做仿真视频', 'SIMULATION', "
        "'sim/ur5e', 'SUCCEEDED', 1, '', ?, ?)",
        (now, now),
    )
    for seq, (etype, payload) in enumerate([
        ("plan.node_started", {"node_id": "resolve_robot"}),
        ("plan.node_completed", {"node_id": "resolve_robot"}),
        ("plan.node_started", {"node_id": "make_path"}),
        ("plan.node_completed", {"node_id": "make_path"}),
        ("plan.node_started", {"node_id": "simulate"}),
        ("plan.node_completed", {"node_id": "simulate"}),
        ("plan.node_started", {"node_id": "render"}),
        ("plan.node_completed", {"node_id": "render"}),
        ("plan.node_started", {"node_id": "render_scene"}),
        ("plan.node_completed", {"node_id": "render_scene"}),
        ("plan.node_started", {"node_id": "verify"}),
        ("plan.node_completed", {"node_id": "verify"}),
    ], start=1):
        conn.execute(
            "INSERT INTO task_events (task_id, seq, event_type, payload_json, "
            "created_at) VALUES ('task_1', ?, ?, ?, ?)",
            (seq, etype, json.dumps(payload), now),
        )
    for artifact_id, path, media, kind in (
        ("art_g", "/h/t1.gif", "image/gif", "preview_2d"),
        ("art_m", "/h/t1-scene.mp4", "video/mp4", "scene_3d"),
        ("art_t", "/h/trace.json", "application/json", ""),
    ):
        meta = json.dumps(
            {"lineage": {"kind": kind, "trace_id": "t1"},
             "evidence": {"levels": ["GEOMETRY_PLAN", "KINEMATIC_TRACKING",
                                     "DYNAMIC_ROLLOUT"]}}
            if kind == "" else {"lineage": {"kind": kind, "trace_id": "t1"}}
        )
        conn.execute(
            "INSERT INTO artifacts (artifact_id, task_id, path, media_type, "
            "sha256, size_bytes, metadata_json, created_at) "
            "VALUES (?, 'task_1', ?, ?, 'abc', 1000, ?, ?)",
            (artifact_id, path, media, meta, now),
        )
    conn.execute(
        "INSERT INTO task_outcomes (task_id, revision, outcome_json, created_at) "
        "VALUES ('task_1', 1, ?, ?)",
        (
            json.dumps({
                "lifecycle": "COMPLETED", "execution": "SUCCEEDED",
                "verification": "PASS", "delivery": "DELIVERED",
                "evidence": {"domain": "SIMULATION", "trust": "TRUSTED",
                             "levels": ["GEOMETRY_PLAN", "KINEMATIC_TRACKING",
                                        "DYNAMIC_ROLLOUT", "SCENE_RENDER"]},
                "artifact_refs": [
                    {"artifact_id": "art_g", "media_type": "image/gif",
                     "kind": "preview_2d",
                     "open_command": "rosclaw artifact open art_g"},
                    {"artifact_id": "art_m", "media_type": "video/mp4",
                     "kind": "scene_3d",
                     "open_command": "rosclaw artifact open art_m"},
                ],
            }),
            now,
        ),
    )
    conn.commit()
    conn.close()
    return home


class TestCanaryAssertions:
    def test_production_plan_graph_events(self, tmp_path: Path) -> None:
        from scripts.star_canary import assert_production_plan_graph

        home = _fixture_home(tmp_path)
        assert_production_plan_graph(home)  # 不抛 = 通过

    def test_plan_graph_missing_fails(self, tmp_path: Path) -> None:
        from scripts.star_canary import assert_production_plan_graph

        home = _fixture_home(tmp_path)
        conn = sqlite3.connect(home / "agentd" / "missions.db")
        conn.execute("DELETE FROM task_events")
        conn.commit()
        conn.close()
        with pytest.raises(AssertionError, match="plan.node"):
            assert_production_plan_graph(home)

    def test_user_visible_delivery(self, tmp_path: Path) -> None:
        from scripts.star_canary import assert_user_visible_delivery

        home = _fixture_home(tmp_path)
        assert_user_visible_delivery(home)

    def test_2d_preview_does_not_satisfy_scene(self, tmp_path: Path) -> None:
        from scripts.star_canary import assert_user_visible_delivery

        home = _fixture_home(tmp_path)
        conn = sqlite3.connect(home / "agentd" / "missions.db")
        conn.execute(
            "UPDATE task_outcomes SET outcome_json = ?",
            (json.dumps({
                "verification": "PASS",
                "artifact_refs": [
                    {"artifact_id": "art_g", "media_type": "image/gif",
                     "kind": "preview_2d",
                     "open_command": "rosclaw artifact open art_g"},
                    {"artifact_id": "art_m", "media_type": "video/mp4",
                     "kind": "preview_2d",
                     "open_command": "rosclaw artifact open art_m"},
                ],
            }),),
        )
        conn.commit()
        conn.close()
        with pytest.raises(AssertionError, match="scene_3d"):
            assert_user_visible_delivery(home, require_scene=True)

    def test_model_behavior_budget(self, tmp_path: Path) -> None:
        from scripts.star_canary import assert_model_behavior_budget

        home = _fixture_home(tmp_path)
        assert_model_behavior_budget(home, max_task_calls=1)

    def test_evidence_levels(self, tmp_path: Path) -> None:
        from scripts.star_canary import assert_evidence_levels

        home = _fixture_home(tmp_path)
        assert_evidence_levels(home)

    def test_screen_clean(self) -> None:
        from scripts.star_canary import assert_screen_clean

        assert_screen_clean(b"\xe4\xbb\xbb\xe5\x8a\xa1\xe5\xae\x8c\xe6\x88\x90 \xe2\x9c\x93 \xe8\xa7\x84\xe5\x88\x92")
        with pytest.raises(AssertionError, match="raw JSON"):
            assert_screen_clean(b'{"artifact_refs": []}')
        with pytest.raises(AssertionError, match="Unreachable|Blocked"):
            assert_screen_clean(b"Kernel Unreachable \xe4\xbb\xbb\xe5\x8a\xa1\xe5\xae\x8c\xe6\x88\x90")

    def test_final_answer_consistency(self) -> None:
        from scripts.star_canary import assert_final_answer_consistent

        # PARTIAL outcome 但回答宣称完整完成 → 拒绝。
        with pytest.raises(AssertionError, match="一致"):
            assert_final_answer_consistent(
                "五角星已绘制完成，视频已交付。",
                {"verification": "PARTIAL", "delivery": "MISSING"},
            )
        # VERIFIED + 完成回答 → 通过。
        assert_final_answer_consistent(
            "五角星已绘制完成，几何验证通过。",
            {"verification": "PASS", "delivery": "DELIVERED"},
        )
        # PARTIAL + 诚实回答 → 通过。
        assert_final_answer_consistent(
            "运动轨迹已验证，但 3D 场景视频未能生成（渲染后端不可用）。",
            {"verification": "PARTIAL", "delivery": "MISSING"},
        )
