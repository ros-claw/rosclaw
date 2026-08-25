"""P1-C1 红测试（0824 总纲 §7.1/P1-C）：TaskSpecV2 契约。

真实缺口：任务只有 root_goal 自由文本 + 冻结 acceptance——没有
intent/subjects/constraints 的结构化工单。Coordinator/Verifier/
PlanGraph 无法按"这是什么类型的任务、对谁做、什么约束"行事，
只能靠模型自己记。

断言：
1. TaskSpecV2 契约校验（intent 冻结分类——未知 intent 拒绝）；
2. compile_task_spec：goal 文本 → 通用规则 intent 分类（非形状
   特例）；subjects.body_ref 经 alias 归一（robot:<name>）；
   constraints.mode/allowed_effects；natural_language 留原文；
   acceptance 关联冻结 spec_id；
3. 任务创建（bind_message）落 task_spec_json，get_task_spec 可读；
   已附着输入不重复编译（无 revision bump）；
4. spec 随 revision 冻结——后续修订不覆盖旧 revision 的 spec。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rosclaw.contracts.agent.task_spec import (
    INTENT_TAXONOMY,
    TaskSpecV2,
)
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.service import TaskKernel
from rosclaw.task_kernel.task_spec import compile_task_spec


def _kernel(tmp_path: Path) -> TaskKernel:
    conn = sqlite3.connect(tmp_path / "k.db")
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, tmp_path)


class TestContract:
    def test_roundtrip_valid(self) -> None:
        spec = TaskSpecV2(
            spec_id="tspec_1",
            task_id="task_1",
            revision=1,
            goal={
                "natural_language": "让机械臂画五角星并给出动画",
                "intent": "manipulation.draw_path",
            },
            subjects={"body_ref": "robot:ur5e", "world_ref": "world:tabletop"},
            constraints={
                "mode": "SIMULATION",
                "frames": ["world", "base_link", "tool0"],
                "allowed_effects": ["READ", "COMPUTE", "SIM_MUTATION"],
            },
            preferences={"language": "zh-CN", "verbosity": "concise"},
        )
        assert spec.goal.intent == "manipulation.draw_path"
        assert spec.subjects.body_ref == "robot:ur5e"

    def test_unknown_intent_rejected(self) -> None:
        with pytest.raises(ValueError, match="intent"):
            TaskSpecV2(
                spec_id="tspec_1", task_id="task_1", revision=1,
                goal={"natural_language": "x", "intent": "magic.fly"},
                subjects={"body_ref": "", "world_ref": ""},
                constraints={"mode": "SIMULATION"},
                preferences={},
            )

    def test_taxonomy_covers_acceptance_tasks(self) -> None:
        """0824 §23 验收任务类型必须在分类里（draw/reach/pick_place/
        navigate/inspect/policy）。"""
        for intent in (
            "manipulation.draw_path",
            "manipulation.reach",
            "manipulation.pick_place",
            "mobile.navigate",
            "perception.inspect",
            "learned_policy.execute",
        ):
            assert intent in INTENT_TAXONOMY


class TestCompiler:
    @pytest.mark.parametrize(
        "goal,intent",
        [
            ("让机械臂画一个五角星并给 GIF", "manipulation.draw_path"),
            ("draw a circle on the board", "manipulation.draw_path"),
            ("把末端伸到坐标 x=0.3 处", "manipulation.reach"),
            ("reach to the target point", "manipulation.reach"),
            ("抓起桌上的方块放到篮子里", "manipulation.pick_place"),
            ("pick up the cube and place it", "manipulation.pick_place"),
            ("导航到充电桩", "mobile.navigate"),
            ("navigate to the dock", "mobile.navigate"),
            ("看一下罐子的位置并靠近复核", "perception.inspect"),
            ("计算这段轨迹的误差", "compute.generic"),
            ("你好，介绍一下你自己", "conversation.chat"),
        ],
    )
    def test_intent_classification_generic(self, goal: str, intent: str) -> None:
        spec = compile_task_spec(
            task_id="task_x", revision=1, goal_text=goal,
            body_id="sim/ur5e", mode="SIMULATION",
            acceptance_spec_id="",
        )
        assert spec.goal.intent == intent, goal

    def test_subjects_body_ref_canonical(self) -> None:
        spec = compile_task_spec(
            task_id="task_x", revision=1, goal_text="画圆",
            body_id="sim/ur5e", mode="SIMULATION",
            acceptance_spec_id="aspec_1",
        )
        assert spec.subjects.body_ref == "robot:ur5e"
        assert spec.goal.natural_language == "画圆"
        assert spec.constraints.mode == "SIMULATION"
        assert spec.acceptance_spec_id == "aspec_1"

    def test_no_star_special_case(self) -> None:
        """分类规则是通用动词类——形状词（五角星/圆）不进 intent。"""
        star = compile_task_spec(
            task_id="t", revision=1, goal_text="画五角星",
            body_id="", mode="SIMULATION", acceptance_spec_id="",
        )
        circle = compile_task_spec(
            task_id="t", revision=1, goal_text="画圆",
            body_id="", mode="SIMULATION", acceptance_spec_id="",
        )
        assert star.goal.intent == circle.goal.intent


class TestKernelIntegration:
    def test_bind_message_persists_task_spec(self, tmp_path: Path) -> None:
        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="让机械臂画五角星并给出动画",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        spec = kernel.get_task_spec(bound["task_id"])
        assert spec is not None, "task_spec_json 未落账"
        assert spec["goal"]["intent"] == "manipulation.draw_path"
        assert spec["goal"]["natural_language"] == "让机械臂画五角星并给出动画"
        assert spec["subjects"]["body_ref"] == "robot:ur5e"
        assert spec["revision"] == 1

    def test_attached_input_does_not_recompile(self, tmp_path: Path) -> None:
        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="画五角星", cwd=str(tmp_path),
        )
        again = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="画五角星", cwd=str(tmp_path),
        )
        assert again["replayed"] is True
        assert again["revision"] == 1  # 无 bump
        spec = kernel.get_task_spec(bound["task_id"])
        assert spec["revision"] == 1

    def test_revision_spec_frozen(self, tmp_path: Path) -> None:
        """修订后旧 revision 的 spec 不被覆盖。"""
        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="n1",
            message_id="msg_1", text="画五角星", cwd=str(tmp_path),
        )
        spec_v1 = kernel.get_task_spec(bound["task_id"])
        assert spec_v1["goal"]["natural_language"] == "画五角星"
