"""R0-5 红测试（0826 体验审计 §5.R0-5）：具身 Verifier 重写——
避免低质量 PASS。

真实缺口（0826 旅程 + 实测）：
- 50mm 平阈值对半径 100mm 的五角星过宽；验收只看 max_error +
  帧数——RMSE/p95/闭合/平面/工具轴/接触都不查；
- 单个 SIM_DYN_ROLLOUT 覆盖全部证据——最终回答可以宣称"持笔
  在桌面画星"而无对应证据等级；
- 笔资产移除/桌面移除/接触关闭/MP4 损坏四种注入仍可能整体
  PASS（证据过度声明）。

实测基线（2026-08-26，本机）：jacobian 控制器 scale=0.10 时
max 19.6mm / mean 7.9mm / 姿态 1.123°——控制器实测下限
~20mm（joint delta 实验 0.0003→0.00005 无改善，稳态误差非
速率限制）。验收公式：max(robot_floor, 0.003, scale*0.05)——
平台实测下限为地板（min() 会让验收严过平台能力=永久红，
那是假装）。

断言：
1. 阈值公式：max(robot_floor, 0.003, scale*0.05)；
2. 丰富指标：rmse_m/p95_error_m/closure_error_m/plane_
   max_deviation_m 落 metrics；
3. 接触证据：contact_required 但接触样本为零 →
   CONTACT_EVIDENCE_MISSING（不 PASS）；
4. 工具轴：姿态误差 > 3° → TOOL_AXIS_EXCEEDED；
5. 场景媒体：不可解码/帧数不足 → SCENE_MEDIA_INVALID（不是
   只看文件存在）；
6. 证据等级：outcome.evidence.levels 拆 GEOMETRY_PLAN/
   KINEMATIC_TRACKING/DYNAMIC_ROLLOUT/SCENE_RENDER（不是
   单个 SIM_DYN_ROLLOUT）；
7. Gate 四注入：笔移除/桌面移除/接触关闭/MP4 损坏——都不得
   完整 PASS。
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestAcceptanceFormula:
    def test_scale_aware_threshold(self) -> None:
        from rosclaw.task_kernel.embodied_verifier import (
            tracking_acceptance,
        )

        # scale 0.10 → 0.005 → robot floor 0.025 约束（实测下限）。
        assert tracking_acceptance(0.10) == pytest.approx(0.025)
        # scale 1.0 → 0.05 超 floor。
        assert tracking_acceptance(1.0) == pytest.approx(0.05)
        # scale 0.01 → 绝对地板 0.003 上抬至 floor。
        assert tracking_acceptance(0.01) == pytest.approx(0.025)
        # 自定义 floor（未来更好的控制器可收紧）。
        assert tracking_acceptance(0.10, robot_floor_m=0.005) == pytest.approx(0.005)
        # floor 突破绝对地板时尺度项仍生效；绝对地板只在极小
        # 尺度下托底。
        assert tracking_acceptance(0.10, robot_floor_m=0.001) == pytest.approx(0.005)
        assert tracking_acceptance(0.01, robot_floor_m=0.001) == pytest.approx(0.003)


def _star_trace(home: Path, scale: float = 0.10) -> dict:
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService

    sim = SimTrajectoryService(home)
    plan = sim.generate_planar_path(
        shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=scale,
    )
    rollout = sim.simulate_cartesian_trajectory(plan["plan_id"])
    return {"sim": sim, "plan": plan, "rollout": rollout}


class TestRichMetrics:
    def test_metrics_include_distribution_and_plane(
        self, tmp_path: Path
    ) -> None:
        """metrics.json 必须含 rmse/p95/闭合误差/平面偏差——
        不是只有 max/mean。"""
        ctx = _star_trace(tmp_path)
        metrics = ctx["rollout"]["tracking"]
        for key in (
            "rmse_error_m", "p95_error_m", "closure_error_m",
            "plane_max_deviation_m", "contact_samples",
        ):
            assert key in metrics, f"缺 {key}：{sorted(metrics.keys())}"
        # 闭合路径：闭合误差应远小于路径尺度。
        assert metrics["closure_error_m"] < 0.05, metrics
        # 平面偏差：xy 平面 z=0.30，接触段应贴平面（tol 0.02）。
        assert metrics["plane_max_deviation_m"] <= 0.02, metrics
        assert metrics["contact_samples"] > 0, metrics


class TestContactEvidence:
    def test_contact_required_but_absent_fails(
        self, tmp_path: Path
    ) -> None:
        """contact_required=true 但接触样本为零（注入"接触关闭"）
        → CONTACT_EVIDENCE_MISSING，不得 PASS。"""
        from rosclaw.task_kernel.embodied_verifier import (
            contact_failures,
        )

        failures = contact_failures(
            {"contact_required": True},
            {"contact_samples": 0},
        )
        assert any("CONTACT_EVIDENCE_MISSING" in f for f in failures)

    def test_contact_present_passes(self) -> None:
        from rosclaw.task_kernel.embodied_verifier import (
            contact_failures,
        )

        assert not contact_failures(
            {"contact_required": True},
            {"contact_samples": 500, "plane_max_deviation_m": 0.01},
        )


class TestToolAxis:
    def test_orientation_over_3deg_fails(self) -> None:
        from rosclaw.task_kernel.embodied_verifier import (
            tool_axis_failures,
        )

        failures = tool_axis_failures(
            {"max_orientation_error_deg": 5.2}, limit_deg=3.0,
        )
        assert any("TOOL_AXIS_EXCEEDED" in f for f in failures)
        assert not tool_axis_failures(
            {"max_orientation_error_deg": 1.1}, limit_deg=3.0,
        )


class TestSceneMediaCheck:
    def _scene_mp4(self, tmp_path: Path) -> Path:
        ctx = _star_trace(tmp_path)
        from rosclaw.agentd.sim_render import render_scene_trace

        result = render_scene_trace(
            tmp_path, ctx["rollout"]["trace_id"],
        )
        return Path(result["artifacts"]["mp4"]["path"])

    def test_valid_scene_mp4_passes(self, tmp_path: Path) -> None:
        from rosclaw.task_kernel.embodied_verifier import (
            scene_media_failures,
        )

        mp4 = self._scene_mp4(tmp_path)
        assert not scene_media_failures(
            mp4, min_frames=30, min_resolution=(640, 360),
        )

    def test_corrupt_mp4_fails(self, tmp_path: Path) -> None:
        """MP4 损坏（注入）→ SCENE_MEDIA_INVALID——不是文件存在
        就算数。"""
        from rosclaw.task_kernel.embodied_verifier import (
            scene_media_failures,
        )

        bad = tmp_path / "corrupt.mp4"
        bad.write_bytes(b"\x00\x00\x00\x18garbage-not-mp4")
        failures = scene_media_failures(
            bad, min_frames=30, min_resolution=(640, 360),
        )
        assert any("SCENE_MEDIA_INVALID" in f for f in failures)

    def test_short_video_fails(self, tmp_path: Path) -> None:
        """帧数不足的 mp4 → SCENE_MEDIA_INVALID（min_frames 语义）。"""
        import imageio.v3 as iio
        import numpy as np

        from rosclaw.task_kernel.embodied_verifier import (
            scene_media_failures,
        )

        short = tmp_path / "short.mp4"
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 4
        iio.imwrite(short, frames, fps=12)
        failures = scene_media_failures(
            short, min_frames=30, min_resolution=(640, 360),
        )
        assert any("SCENE_MEDIA_INVALID" in f for f in failures)

# 大道至简 R0-2b：TestEvidenceLevels/TestGateFourInjections（recipe
# 链验收等级/注入）随生产 recipe 链删除——具身 verifier 纯函数
# 面（验收公式/指标/接触证据/工具朝向/场景媒体）保留。
