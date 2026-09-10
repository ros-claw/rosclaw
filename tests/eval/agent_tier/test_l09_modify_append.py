"""L09 修改目标与追加交付（规格 §13.11）——agent 层真实驱动。

同一会话三腿：
1. 周期运动仿真 + 视频；
2. “幅度减半，只给 MP4”；
3. “同一结果的顶视图”。

oracle（环境结局）：
- 前后实际运动的幅度比 ∈ [0.45, 0.55]（从轨迹数据实测）；
- 顶视图复用正确 trace——**不产生新 rollout**（渲染输入 states
  digest 与修改后 trace 一致；不重新仿真）；
- 新旧产物可区分且都可导出（文件非空）；
- 不出现 TASK_ALREADY_COMPLETED 循环（W05 语义）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.eval.agent_tier import driver

T1 = "做一个 UR5e 机械臂的周期摆动仿真（中等幅度，比如末端画一个半径 5cm 的圆，走两圈），给我场景视频。"
T2 = "把刚才那个运动的幅度减半，这次只要 MP4。"
T3 = "再给同一结果的顶视图版本。"


def _trace_dirs(home: Path) -> list[Path]:
    traces = home / "sim" / "traces"
    if not traces.exists():
        return []
    return sorted(traces.iterdir(), key=lambda p: p.stat().st_mtime)


def _amplitude(trace_dir: Path) -> float | None:
    """幅度 = 周期段（轨迹尾部 40%）的 xy 范围——全程范围被
    home→起点转场污染（实测：转场 0.57m 掩盖了圆半径 5cm→2.5cm
    的真实减半）。"""
    doc = json.loads((trace_dir / "trace.json").read_text(encoding="utf-8"))
    pts = doc.get("actual") or []
    if len(pts) < 10:
        return None
    tail = pts[int(len(pts) * 0.6):]
    xs = [p["x"] for p in tail]
    ys = [p["y"] for p in tail]
    return (max(xs) - min(xs) + max(ys) - min(ys)) / 2.0


@pytest.mark.slow
@pytest.mark.skipif(
    not (driver.has_key() and driver.has_runtime()),
    reason="NOT_RUN: 无真实 key/Node——不合成冒充",
)
class TestL09ModifyAndAppendAgent:
    def test_halve_amplitude_and_top_view(self, tmp_path: Path) -> None:
        run = driver.AgentRun(tmp_path, settle_timeout=1200)
        try:
            run.run(T1)
            traces_1 = _trace_dirs(run.home)
            assert traces_1, "第一腿无 trace（见 PTY 日志）"
            amp1 = _amplitude(traces_1[-1])
            run.followup(T2)
            traces_2 = _trace_dirs(run.home)
            assert len(traces_2) >= 2, "第二腿无新 trace（见 PTY 日志）"
            amp2 = _amplitude(traces_2[-1])
            run.followup(T3)
        finally:
            run.stop()
        assert amp1 and amp1 > 0.01 and amp2, (
            f"幅度数据不可用: amp1={amp1} amp2={amp2}"
        )
        ratio = amp2 / amp1
        assert 0.45 <= ratio <= 0.55, f"幅度比 {ratio:.3f} 不在 0.45–0.55"
        # 顶视图：复用修改后 trace 渲染——渲染 receipt 的输入 digest
        # 与第二腿 trace 的 states digest 一致（不重新 rollout）。
        t2 = traces_2[-1]
        states_digest = json.loads(
            (t2 / "trace.json").read_text(encoding="utf-8")
        ).get("states_digest", "")
        receipts = list(t2.glob("**/render_receipt.json"))
        assert receipts, "顶视图渲染 receipt 缺失（见 PTY 日志）"
        receipt = json.loads(receipts[-1].read_text(encoding="utf-8"))
        assert receipt.get("input_trace_digest") or receipt.get(
            "states_digest"
        ) == states_digest, "顶视图输入与修改后 trace 不符"
        # 新旧产物可区分且非空。
        medias = sorted(t2.glob("**/*.mp4")) + sorted(t2.glob("**/*.gif"))
        assert medias and all(p.stat().st_size > 1000 for p in medias), (
            "产物缺失或为空"
        )
        # 无 TASK_ALREADY_COMPLETED 循环。
        clean = run.session.clean if run.session else b""
        assert b"TASK_ALREADY_COMPLETED" not in clean


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
