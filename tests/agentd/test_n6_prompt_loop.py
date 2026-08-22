"""PR-N6A/N6B 红测试（调整方案 §五）：删魔法任务路由 + 通用自主闭环。

红测试先行——当前提示词含五角星 recipe/ONLY entry/PREFERRED 宏，
必须红。

N6A：系统提示词只保留身份/权限边界/证据纪律/亲自工作/effect 安全/
通用工作循环；删除五角星 recipe、exact tool 清单、ONLY correct
entry、固定参数、大段错误码与子系统硬背介绍。领域方法移入
Capability Descriptor / Skill / Ecosystem Index / Error Recovery
Registry / Acceptance Template。

N6B：通用闭环 Understand→Ground→Inspect→Plan→Act→Observe→Verify→
Repair→Deliver，且明确"不是每个任务都机械走全部阶段"（问候直答、
单一强类型能力 Fast Path、需要才 inspect、缺能力才造、长任务转
Operation、REAL 才审批）。
"""

from __future__ import annotations

from pathlib import Path

PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src" / "rosclaw" / "agentd" / "context" / "prompts" / "native_agent_v2.md"
)


class TestN6APromptSlimmed:
    def test_no_star_recipe_or_only_entry(self) -> None:
        text = PROMPT.read_text(encoding="utf-8")
        # 五角星具体 recipe / 固定参数 / ONLY 入口全部删除。
        for banned in (
            "ONLY correct entry",
            "PREFERRED entry",
            "star5",
            "0.35",
            "max_tracking_error_m",
            "animation_min_frames",
            "NEVER turn an ordinary simulation request into a development project",
        ):
            assert banned not in text, f"提示词仍含魔法路由内容: {banned!r}"

    def test_no_per_subsystem_tool_briefings(self) -> None:
        """exact tool 清单删除——工具自描述在工具面/Schema/Skill，
        提示词不逐个硬背 rosclaw_* 用法。"""
        text = PROMPT.read_text(encoding="utf-8")
        # 逐个工具的使用说明行（"- rosclaw_xxx:"）不得超过 2 条
        # （保留边界类提及，删用法 briefing）。
        briefings = [
            line for line in text.splitlines()
            if line.startswith("- rosclaw_") and ":" in line
        ]
        assert len(briefings) <= 2, (
            f"仍有 {len(briefings)} 条逐工具 briefing: {briefings}"
        )

    def test_prompt_budget(self) -> None:
        """提示词预算：当前 72 行 → ≤ 50 行（瘦身必须真实发生）。"""
        lines = PROMPT.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 50, f"提示词 {len(lines)} 行，超预算 50"

    def test_keeps_invariants(self) -> None:
        """保留：身份/权限边界/证据纪律/亲自工作/安全 fail closed。"""
        text = PROMPT.read_text(encoding="utf-8")
        assert "rosclawd" in text  # 物理执行权威边界
        assert "fail closed" in text or "拒绝" in text
        # 证据纪律（证据等级措辞）保留
        assert "COMMAND_REPLAY" in text and "SIM_DYN_ROLLOUT" in text
        # 亲自工作
        assert "yourself" in text.lower() or "亲自" in text


class TestN6BAutonomousLoop:
    def test_generic_loop_present(self) -> None:
        """通用工作循环在提示词中，且明确非机械。"""
        text = PROMPT.read_text(encoding="utf-8")
        for stage in ("Understand", "Ground", "Plan", "Act", "Observe",
                      "Verify", "Deliver"):
            assert stage in text, f"通用循环缺 {stage}"
        # 非机械：问候直答 / Fast Path / 需要才 inspect。
        assert "Fast Path" in text or "fast path" in text
        assert "greeting" in text.lower() or "问候" in text

    def test_no_magic_task_routing(self) -> None:
        """rosclaw_task 不再是"首选入口"叙述——它是健康时的快捷宏。"""
        text = PROMPT.read_text(encoding="utf-8")
        assert "PREFERRED entry" not in text
        assert "Never hand-chain" not in text
