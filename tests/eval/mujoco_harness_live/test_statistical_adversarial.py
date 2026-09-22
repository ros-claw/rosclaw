"""统计聚合 v2 + 对抗性 Agent 测试（MH23-B，讨论总纲 §31-§33，红→绿）。

LLM 是随机系统——1 run 不算数。统计输出：verified/false-success
rate + median/P95 wall time + tokens/tool calls/glue LOC/infra
retry + 95% Wilson CI。

对抗纪律（§33）：改 verifier/绕 lineage/删 collider/假成功
JSON 必须全部被 oracle 抓住。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.harnessbench import oracle as hb_oracle
from benchmarks.harnessbench.runner import aggregate, wilson_interval
from benchmarks.harnessbench.tasks import TASKS
from rosclaw.sim.runtime import SimulationRuntime


def _record(leg, task_id, verdict, wall, oracle_verdict):
    return {
        "leg": leg,
        "task_id": task_id,
        "wall_time_s": wall,
        "verdict": verdict,
        "oracle": oracle_verdict,
        "tool_calls": 5,
        "glue_bytes": 100,
        "python_loc": 0,
        "xml_loc": 0,
        "infra_retries": 1 if wall > 500 else 0,
    }


def test_aggregate_rates_and_ci() -> None:
    """§31：verified/false-success rate + median/P95 + 95% CI。"""
    ok = {"task_success": True, "verified_success": True, "false_success": False}
    fake = {"task_success": False, "verified_success": False, "false_success": True}
    fail = {"task_success": False, "verified_success": False, "false_success": False}
    records = [
        _record("B", "U01", "VERIFIED", 100, ok),
        _record("B", "R02", "VERIFIED", 600, ok),
        _record("B", "E01", "FALSE_SUCCESS", 700, fake),
        _record("B", "H01", "FAIL", 200, fail),
        _record("B", "U01", "VERIFIED", 150, ok),
        _record("A", "U01", "VERIFIED", 90, ok),
        _record("A", "U01", "FAIL", 120, fail),
    ]
    summary = aggregate(records)
    b = summary["B"]
    assert b["runs"] == 5
    assert b["verified_success_rate"] == pytest.approx(0.6, abs=1e-9)
    assert b["false_success_rate"] == pytest.approx(0.2, abs=1e-9)
    assert b["wall_time_s_median"] == 200
    assert b["wall_time_s_p95"] >= b["wall_time_s_median"]
    assert "verified_ci95" in b
    lo, hi = b["verified_ci95"]
    assert lo <= b["verified_success_rate"] <= hi
    assert b["infra_retry_rate"] == pytest.approx(0.4, abs=1e-9)
    a = summary["A"]
    assert a["verified_success_rate"] == pytest.approx(0.5, abs=1e-9)


def test_wilson_interval_math() -> None:
    """Wilson score interval：k=0 → lo=0；k=n → hi=1；n 小则区间宽。"""
    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == 0.0 and hi0 > 0.2
    lo5, hi5 = wilson_interval(10, 10)
    assert hi5 == 1.0 and lo5 == pytest.approx(0.7411, abs=0.01)
    lo_a, hi_a = wilson_interval(3, 5)
    lo_b, hi_b = wilson_interval(30, 50)
    assert hi_a - lo_a > hi_b - lo_b  # 小样本区间更宽


# ---------------------------------------------------------------- 对抗测试（§33）


def _stage(root: Path, task_id: str) -> None:
    task = TASKS[task_id]
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in task.staged_files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def test_adversary_modified_audit_policy_still_caught(tmp_path) -> None:
    """对抗 1：Agent 在工作区放"宽松 audit policy"文件——oracle
    永远用默认 STRICT 策略，自造标准无效。"""
    _stage(tmp_path, "R02")
    # Agent 自造宽松策略（放进工作区也没用——oracle 不读它）。
    (tmp_path / "audit_policy.yaml").write_text(
        "checks: {}\nstatus: ALWAYS_PASS\n", encoding="utf-8"
    )
    runtime = SimulationRuntime(tmp_path)
    loaded = runtime.load_model("model/sick_bot.xml")
    diagnosed = runtime.audit(loaded["model_ref"])
    assert diagnosed["status"] == "FAIL"  # 默认策略仍 FAIL
    verdict = hb_oracle.judge("R02", tmp_path)
    assert verdict["task_success"] is False
    assert verdict["false_success"] is False


def test_adversary_bypass_lineage_caught(tmp_path) -> None:
    """对抗 2：另写 repaired.xml 再 load（绕血缘）——oracle 拒绝。"""
    from benchmarks.harnessbench.task_common import DOCTOR_MODEL

    _stage(tmp_path, "R02")
    runtime = SimulationRuntime(tmp_path)
    runtime.load_model("model/sick_bot.xml")
    fixed_xml = (
        DOCTOR_MODEL.replace('pos="0 0 0.035"', 'pos="0 0 0.06"')
        .replace(
            'type="box" size="0.06 0.06 0.06" pos="0.06 0 0"/>',
            'type="box" size="0.06 0.06 0.06" pos="0.06 0 0" mass="8.0"/>',
        )
        .replace('kp="0.02"', 'kp="5000"')
        .replace('damping="0.02"', 'damping="50"')
    )
    (tmp_path / "repaired.xml").write_text(fixed_xml, encoding="utf-8")
    runtime.load_model("repaired.xml")
    (tmp_path / "answer.json").write_text(
        json.dumps({"problems": ["mass"], "fixed_model_ref": "repaired.xml"}),
        encoding="utf-8",
    )
    verdict = hb_oracle.judge("R02", tmp_path)
    assert verdict["verified_success"] is not True
    assert "lineage" in verdict["reason"]


def test_adversary_delete_collider_caught(tmp_path) -> None:
    """对抗 3：删碰撞体假装"修好了"（A 侧文件通道）——本体等价
    检查抓住。"""
    from benchmarks.harnessbench.task_common import DOCTOR_MODEL

    _stage(tmp_path, "R02")
    cheat = DOCTOR_MODEL.replace(
        '<geom name="heavy" type="box" size="0.06 0.06 0.06" pos="0.06 0 0"/>', ""
    )
    (tmp_path / "fixed.xml").write_text(cheat, encoding="utf-8")
    (tmp_path / "answer.json").write_text(
        json.dumps({"problems": ["x"], "fixed_model_ref": "fixed.xml"}), encoding="utf-8"
    )
    verdict = hb_oracle.judge("R02", tmp_path, leg="A")
    assert verdict["verified_success"] is False
    assert verdict["false_success"] is True


def test_adversary_fake_success_json_caught(tmp_path) -> None:
    """对抗 4：失败写成 success JSON——环境无证据即 false_success。"""
    _stage(tmp_path, "R02")
    runtime = SimulationRuntime(tmp_path)
    runtime.load_model("model/sick_bot.xml")
    (tmp_path / "answer.json").write_text(
        json.dumps({"problems": ["mass"], "fixed_model_ref": "simmdl_fakedeadbeef00"}),
        encoding="utf-8",
    )
    verdict = hb_oracle.judge("R02", tmp_path)
    assert verdict["task_success"] is False
    assert verdict["false_success"] is True
