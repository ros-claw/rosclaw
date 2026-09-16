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
