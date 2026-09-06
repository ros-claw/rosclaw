"""大道至简 R0-2a 红测试：固定五角星流程显式 demo 化——
`rosclaw demo run ur5e-star`，不拦截用户对话。

方案 R0：「固定五角星流程可以留下，但只能是显式 demo——它不能
再拦截用户对话」。Kernel/Runtime 只报客观执行事实（轨迹执行
成功 + 最大跟踪误差），不宣称用户目标完成。

闭环断言：
1. demo 注册表含 ur5e-star（与 ur5e-reach 同一官方证据面）；
2. run_demo("ur5e-star") 真实跑完 plan→rollout→verify→render：
   COMPLETED + TASK_VERIFIED（PASS）+ 最大误差是客观数值 +
   GIF 交付物在盘 + receipt 持久化；
3. CLI 层 `demo run ur5e-star --json` 退出码 0、载荷带客观指标；
4. 客观事实纪律：receipt 无用户目标语义字段（goal/intent/
   satisfied）——Runtime 只说"轨迹执行成功，误差 X m"。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestStarDemoRegistry:
    def test_ur5e_star_registered(self) -> None:
        from rosclaw.product.demo import DEMOS, list_demos

        assert "ur5e-star" in DEMOS
        ids = [d.id for d in list_demos()]
        assert "ur5e-star" in ids and "ur5e-reach" in ids

    def test_star_demo_objective_facts_only(self, tmp_path: Path) -> None:
        from rosclaw.product.demo import run_demo

        receipt, receipt_path = run_demo("ur5e-star", home=tmp_path)
        assert receipt.final_state.value == "COMPLETED"
        assert receipt.verified, "跟踪验证 PASS 必须是 TASK_VERIFIED"
        # 客观执行事实：轨迹 + 误差数值 + 交付物。
        sim = receipt.simulation_result or {}
        assert sim.get("physics_executed") is True, sim
        metrics = (receipt.verification_result or {}).get("metrics") or {}
        assert isinstance(metrics.get("max_error_m"), float), metrics
        gifs = [a for a in receipt.artifacts if str(a).endswith(".gif")]
        assert gifs and Path(gifs[0]).exists(), receipt.artifacts
        assert receipt_path.exists(), "receipt 未持久化"
        # 客观事实纪律：无用户目标语义（目标完成与否由 Pi 判断，
        # 不是 Runtime）。
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        for forbidden in ("goal_satisfied", "user_goal", "intent_satisfied"):
            assert forbidden not in payload, forbidden

    def test_star_demo_cli(self, tmp_path: Path) -> None:
        import os

        src = str(Path(__file__).resolve().parents[2] / "src")
        env = {**os.environ, "PYTHONPATH": src}
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw.entrypoint", "demo", "run",
             "ur5e-star", "--home", str(tmp_path), "--json"],
            capture_output=True, text=True, timeout=600, env=env,
        )
        assert result.returncode == 0, result.stderr[-500:]
        payload = json.loads(result.stdout)
        assert payload["capability_id"] == "sim.draw_path.star5"
        vr = payload.get("verification_result") or {}
        assert vr.get("verdict") == "PASS", vr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
