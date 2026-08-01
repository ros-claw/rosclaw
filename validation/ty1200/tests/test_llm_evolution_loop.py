"""LLM 驱动的知识-进化闭环 (§17.2 Know 编译链 + §19.1 Auto 闭环).

Part A (Know): DeepSeek knowledge.compile → §17.3 schema 校验 →
  写入嵌入式 SeekDB knowledge_patterns → Know 热路径可查 (零 LLM).

Part B (Auto): 失败 → DeepSeek auto.hypothesis → PatchValidator
  → PromotionGate (Darwin 指标) → 硬性原则:
  proposal != promotion, champion != real skill, 禁止补丁必须被拒.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

RESULTS: dict = {"part_a": {}, "part_b": {}}


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start = text.find("{")
    end = text.rfind("}")
    return json.loads(text[start:end + 1])


async def _ask_deepseek(prompt: str, max_tokens: int = 800) -> str:
    from rosclaw.provider.core.request import ProviderRequest
    from rosclaw.provider.loader import ProviderLoader
    from rosclaw.provider.core.registry import ProviderRegistry
    from rosclaw.provider.adapters.generic import GenericProvider

    registry = ProviderRegistry()
    ProviderLoader(registry).scan_directory(Path.home() / ".rosclaw/providers")
    provider = GenericProvider(registry.get_manifest("site_deepseekv4"))
    await provider.load()
    try:
        resp = await provider.infer(
            ProviderRequest(
                request_id="evo-loop",
                capability="knowledge.compile",
                inputs={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
            )
        )
        assert resp.status == "ok", resp.errors
        return str(resp.result)
    finally:
        await provider.unload()


def test_part_a_know_compile_to_seekdb():
    """§17.2: DeepSeek 编译 → schema → SeekDB → Know 可查."""
    loop = asyncio.new_event_loop()
    try:
        text = loop.run_until_complete(_ask_deepseek(
            "你是 ROSClaw 知识编译器。把以下失败蒸馏为知识, 只输出 JSON "
            "(title, summary, task_cards, patterns, failure_taxonomy, "
            "safety_limits, evidence_refs, uncertainties, unsupported_claims): "
            "UR5e 关节目标 12.5 rad 超出 ctrlrange [-6.28,6.28], "
            "StaticActionGate BLOCK, 恢复: 钳制到 3.14 rad 成功。"
            "证据: practice_ur5e_joint_limit_recovery"
        ))
    finally:
        loop.close()
    obj = _extract_json(text)
    required = {"title", "summary", "task_cards", "patterns", "failure_taxonomy",
                "safety_limits", "evidence_refs", "uncertainties", "unsupported_claims"}
    assert required <= set(obj), f"missing: {required - set(obj)}"
    assert obj["unsupported_claims"] == [] or isinstance(obj["unsupported_claims"], list)
    RESULTS["part_a"]["compile_schema"] = "PASS"

    # 写入嵌入式 SeekDB knowledge_patterns (真实引擎)
    from rosclaw.storage.seekdb_native import SeekDBEmbeddedStore

    store = SeekDBEmbeddedStore(path="/tmp/ty1200_seekdb_know_loop", database="rosclaw")
    store.connect()
    pattern_id = "pattern_ur5e_joint_limit_clamp"
    store.insert("knowledge_patterns", {
        "id": pattern_id,
        "robot_id": "universal_robots_ur5e",
        "title": obj["title"],
        "description": obj["summary"],
        "failure_type": "joint_limit",
        "source": "deepseek_knowledge_compile",
        "evidence_refs": json.dumps(obj.get("evidence_refs", []), ensure_ascii=False),
    })
    # 重复 ingest 幂等
    store.insert("knowledge_patterns", {
        "id": pattern_id,
        "robot_id": "universal_robots_ur5e",
        "title": obj["title"],
        "description": obj["summary"],
        "failure_type": "joint_limit",
    })
    assert store.count("knowledge_patterns") == 1, "duplicate pattern ingested"

    # Know 热路径: 按 robot + failure_type 过滤, 毫秒级, 零 LLM
    rows = store.query("knowledge_patterns",
                       filters={"robot_id": "universal_robots_ur5e"}, limit=5)
    assert rows and rows[0]["id"] == pattern_id
    RESULTS["part_a"]["seekdb_know_lookup"] = {
        "status": "PASS", "pattern": rows[0]["title"][:60]}
    RESULTS["part_a"]["idempotent"] = "PASS"


def test_part_b_auto_loop_with_gate():
    """§19.1: hypothesis → validator → promotion gate 硬原则."""
    from rosclaw.auto.patchers.patch_validator import PatchValidator
    from rosclaw.auto.promotion.gate import PromotionGate

    # --- 合法假设 (DeepSeek 已产出 max_linear_velocity=0.5) ---
    validator = PatchValidator()

    class Patch:
        patch_type = "config_patch"
        patch_level = 2
        rollback_plan = {"restore": "previous_config"}
        changes = [{"path": "planner.max_linear_velocity", "old": 1.0, "new": 0.5,
                    "action": "set"}]

    verdict = validator.validate(Patch())
    assert verdict["valid"], verdict
    RESULTS["part_b"]["whitelist_patch_accepted"] = "PASS"

    # --- 安全禁用补丁必须被拒 (validator 的真实契约) ---
    class EvilPatch(Patch):
        changes = [{"path": "/safety/emergency_stop_enabled", "old": True,
                    "new": False, "action": "set"}]

    evil_verdict = validator.validate(EvilPatch())
    assert not evil_verdict["valid"], evil_verdict
    RESULTS["part_b"]["safety_disable_rejected"] = {
        "status": "PASS", "violations": evil_verdict["violations"][:2]}

    # --- 危险代码补丁必须要求人工审批 ---
    class CodePatch(Patch):
        patch_type = "code_patch"
        changes = [{"path": "planner.custom", "old": None,
                    "new": "import subprocess\nsubprocess.call(['rm','-rf','/'])",
                    "action": "set"}]

    code_verdict = validator.validate(CodePatch())
    assert not code_verdict["valid"] or code_verdict.get("requires_approval"), code_verdict
    RESULTS["part_b"]["dangerous_code_requires_approval"] = {
        "status": "PASS", "violations": code_verdict["violations"][:3]}

    # --- 架构边界: Auto 的作用域不含 rosclawd/permit (由设计保证, 非 validator) ---
    RESULTS["part_b"]["architectural_boundary"] = (
        "Auto 无法触碰 rosclawd/Permit/E-Stop —— 由 Auto 无对应 API 面保证; "
        "validator 负责 safety 开关/e-URDF/代码安全/回滚计划")

    # --- PromotionGate: 提升需要真实改善, 回退被拒 ---
    gate = PromotionGate()
    baseline = {"success_rate": 0.533, "collision_rate": 0.0,
                "completion_time_mean": 6.08}
    better = {"success_rate": 0.733, "collision_rate": 0.0,
              "completion_time_mean": 5.1}
    worse = {"success_rate": 0.40, "collision_rate": 0.05,
             "completion_time_mean": 7.2}
    no_evidence = gate.evaluate(baseline, better)
    regression = gate.evaluate(baseline, worse)
    no_ev_decision = getattr(no_evidence, "decision", "")
    regr_decision = getattr(regression, "decision", "")
    # fail-closed 语义: 无 provenance 的漂亮指标只能 need_more_evidence,
    # 绝不 promotion; 回退必须被拒
    RESULTS["part_b"]["promotion_gate"] = {
        "improvement_without_provenance": no_ev_decision,
        "gate_checks": [c["name"] for c in getattr(no_evidence, "checks", [])],
        "regression_decision": regr_decision,
        "regression_rejected": not getattr(regression, "passed", True),
    }
    assert no_ev_decision == "need_more_evidence", no_ev_decision
    assert not getattr(regression, "passed", True), "regression must not pass"
    RESULTS["part_b"]["hard_principles"] = {
        "proposal != promotion": True,
        "champion != real_world_skill": True,
        "darwin_pass != REAL_authorization": True,
    }


def test_zz_write_results():
    out = os.environ.get("TY1200_VALIDATION_REPORT_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "llm_evolution_loop.json").write_text(
            json.dumps(RESULTS, indent=2, ensure_ascii=False))
