"""HarnessBench live gate（MH11，0916 优化 §五，G20/G21）。

无 ROSCLAW_KIMI_API_KEY/KIMI_API_KEY/MOONSHOT_API_KEY → 全部
NOT_RUN（诚实跳过，绝不合成冒充真实 LLM 结果）。真实跑用
scripts/harnessbench_run.py（operator gate，结果落 results.json）。
本文件只锁纪律本身：无 key 时 runner/script 不得产出任何
"成功"记录。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def test_runner_not_run_without_key(tmp_path, monkeypatch) -> None:
    """无 key：script 输出 NOT_RUN 且退出码 0（诚实，不伪造结果）。"""
    for var in ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "harnessbench_run.py"),
            "--legs",
            "B",
            "--tasks",
            "U01",
            "--runs",
            "1",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["status"] == "NOT_RUN"
    # 绝不产生 results.json（没有真实运行就没有"结果"）。
    assert not list(tmp_path.rglob("results.json"))


def test_task_workspace_isolation(tmp_path) -> None:
    """§7.1 答案零泄漏：staged workspace 只有 task.md + model/，
    不含 oracle/expected/golden 任何痕迹。"""
    from benchmarks.harnessbench.runner import stage_workspace
    from benchmarks.harnessbench.tasks import TASKS

    for task_id in TASKS:
        ws = stage_workspace(tmp_path / task_id, task_id)
        names = {p.name for p in ws.rglob("*")}
        assert "task.md" in names
        forbidden = [
            n
            for n in names
            if any(marker in n.lower() for marker in ("oracle", "golden", "expected", "answer"))
        ]
        assert not forbidden, f"{task_id} workspace 泄漏: {forbidden}"


def test_b_hint_carries_lineage_citation_contract() -> None:
    """B 侧提示必须写明引用契约（live 标定第三例实证 2026-09-23）：

    kimi-k3 R01 修复工作真实完成（血缘 patch + audit PASS + replay ok），
    但 answer.json 引用了另写文件 load 的孤儿 ref——任务 prompt 写
    "引用或路径"而 B 腿 oracle 只认血缘，诱导性张冠李戴
    （claimed_ref_mismatch false_success）。提示必须告知：填 patch
    血缘链上的 model_ref，文件孤儿不采信。A 侧提示不得含（无此机制）。
    """
    from benchmarks.harnessbench.runner import _A_TOOL_HINT, _B_TOOL_HINT

    assert "血缘" in _B_TOOL_HINT and "model_ref" in _B_TOOL_HINT
    assert "不采信" in _B_TOOL_HINT or "不被采信" in _B_TOOL_HINT
    # 证据格式契约（live 标定第六例实证）：回执承载 + strict replay
    # 可复放；自写脚本测量不采信——不告知则诚实 Agent 用自写脚本
    # 验证而留不下可复放证据（kimi-k3 R03 踩中）。
    assert "回执" in _B_TOOL_HINT and "replay" in _B_TOOL_HINT
    assert "血缘" not in _A_TOOL_HINT


@pytest.mark.integration
def test_live_smoke_u01_b_leg(tmp_path) -> None:
    """真实冒烟（integration，operator 手动跑）：B 侧 U01 一次。

    完整矩阵走 scripts/harnessbench_run.py；本用例只证明
    live 链路端到端可跑且 oracle 给出结构化判定。
    """
    from benchmarks.harnessbench.runner import has_model_key, run_leg

    if not has_model_key():
        pytest.skip("no model key — NOT_RUN（不合成冒充）")
    record = run_leg("B", "U01", tmp_path, 1, settle_timeout=900.0)
    assert record["oracle"]["reason"] != "runner_error"
    assert record["verdict"] in ("VERIFIED", "DONE", "FAIL", "FALSE_SUCCESS")
