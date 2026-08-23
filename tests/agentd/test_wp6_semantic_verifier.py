"""WP-6 红测试（0823 审计 P0-1）：语义验收——证据必须真正证明目标。

红测试先行——深度语义检查不存在时必须红。

验收器不能只查"证据存在"（有 GIF/metrics 就 PASS——五角星事故
的根因），必须验证证据**证明了用户目标**：

1. 媒体可解码 + 非空白（损坏/全白 GIF 不得通过）；
2. trajectory_states digest 与 trace 记录一致（篡改即拒）；
3. qpos 必须真实变化（恒等 states = 命令回放伪装动力学）；
4. 场景渲染 GIF 必须有 RenderReceipt 且 input_trace_digest 锚定
   被验 trace（renderer 可信链）；
5. 工具轴保持接触平面法向（WP-5 朝向指标纳入验收）；
6. 接触高度贴合接触平面（画在平面上，不是悬空）；
7. kit 的 ur5e.verify_drawing 诚实标注 NOT_INDEPENDENT /
   COMMAND_REPLAY_ONLY（命令回放校验不是独立验收证据）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _run_pipeline(home: Path, shape: str = "star5") -> dict:
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService

    sim = SimTrajectoryService(home)
    plan = sim.generate_planar_path(
        shape=shape, center_m=[0.35, 0.25, 0.30], scale_m=0.05,
    )
    result = sim.simulate_cartesian_trajectory(plan["plan_id"])
    render = sim.render_trace(result["trace_id"])
    return {"plan": plan, "run": result, "render": render}


def _check(home: Path, ctx: dict, **overrides) -> list[str]:
    from rosclaw.task_kernel.verifier_plugins import TrajectoryVerifier

    args = {
        "trace_json": ctx["run"]["artifacts"]["trace_json"],
        "metrics_json": ctx["run"]["artifacts"]["metrics_json"],
        "gif_path": ctx["render"]["artifact"]["path"],
        "home": home,
        "declared_shape": "star5",
        "max_tracking_error_m": 0.05,
    }
    args.update(overrides)
    return TrajectoryVerifier().check(**args)


class TestMediaSemanticChecks:
    def test_genuine_pipeline_passes_deep(self, tmp_path: Path) -> None:
        """对照组：真实管线必须通过全部深度检查。"""
        ctx = _run_pipeline(tmp_path)
        assert _check(tmp_path, ctx) == []

    def test_blank_gif_rejected(self, tmp_path: Path) -> None:
        """全白 GIF（有文件有帧但没画任何东西）不得通过。"""
        ctx = _run_pipeline(tmp_path)
        from PIL import Image

        gif = Path(ctx["render"]["artifact"]["path"])
        blank = Image.new("RGB", (480, 480), "white")
        blank.save(gif, save_all=True, append_images=[blank.copy()] * 5,
                   duration=80, loop=0)
        failures = _check(tmp_path, ctx)
        assert any("MEDIA_BLANK" in f for f in failures), failures

    def test_corrupt_gif_rejected(self, tmp_path: Path) -> None:
        ctx = _run_pipeline(tmp_path)
        gif = Path(ctx["render"]["artifact"]["path"])
        gif.write_bytes(b"this is not a gif at all")
        failures = _check(tmp_path, ctx)
        assert any("MEDIA_UNDECODABLE" in f for f in failures), failures


class TestStatesIntegrityChecks:
    def test_states_digest_mismatch_rejected(self, tmp_path: Path) -> None:
        """改了 trajectory_states 没同步 trace 记录 → 篡改拒绝。"""
        ctx = _run_pipeline(tmp_path)
        states_path = (
            tmp_path / "sim" / "traces" / ctx["run"]["trace_id"]
            / "trajectory_states.json"
        )
        doc = json.loads(states_path.read_text(encoding="utf-8"))
        doc["states"][10]["qpos"][0] += 0.5
        states_path.write_text(json.dumps(doc), encoding="utf-8")
        failures = _check(tmp_path, ctx)
        assert any("TRACE_STATES_DIGEST_MISMATCH" in f for f in failures), failures

    def test_static_qpos_rejected(self, tmp_path: Path) -> None:
        """qpos 恒等（命令回放伪装动力学）——即使伪造者同步更新
        digest，也必须被物理一致性检查抓住。"""
        ctx = _run_pipeline(tmp_path)
        trace_dir = tmp_path / "sim" / "traces" / ctx["run"]["trace_id"]
        states_path = trace_dir / "trajectory_states.json"
        doc = json.loads(states_path.read_text(encoding="utf-8"))
        first_qpos = doc["states"][0]["qpos"]
        for state in doc["states"]:
            state["qpos"] = list(first_qpos)
        states_path.write_text(json.dumps(doc), encoding="utf-8")
        # 伪造者同步 digest——digest 检查放过，物理检查必须抓住。
        trace_path = trace_dir / "trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace["states_digest"] = "sha256:" + hashlib.sha256(
            states_path.read_bytes()
        ).hexdigest()
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        failures = _check(tmp_path, ctx)
        assert any("STATES_STATIC" in f for f in failures), failures


class TestRenderTrustChain:
    def _scene_gif(self, home: Path, trace_id: str) -> Path:
        from rosclaw.agentd.sim_render import render_scene_trace

        result = render_scene_trace(home, trace_id)
        return Path(result["artifact"]["path"])

    def test_scene_render_genuine_passes(self, tmp_path: Path) -> None:
        ctx = _run_pipeline(tmp_path)
        gif = self._scene_gif(tmp_path, ctx["run"]["trace_id"])
        assert _check(tmp_path, ctx, gif_path=str(gif)) == []

    def test_scene_render_missing_receipt_rejected(self, tmp_path: Path) -> None:
        ctx = _run_pipeline(tmp_path)
        gif = self._scene_gif(tmp_path, ctx["run"]["trace_id"])
        (gif.parent / "render_receipt.json").unlink()
        failures = _check(tmp_path, ctx, gif_path=str(gif))
        assert any("RENDER_RECEIPT_MISSING" in f for f in failures), failures

    def test_scene_render_receipt_mismatch_rejected(self, tmp_path: Path) -> None:
        """receipt 锚定的是别的 trace——拼凑渲染证据。"""
        ctx = _run_pipeline(tmp_path)
        gif = self._scene_gif(tmp_path, ctx["run"]["trace_id"])
        receipt_path = gif.parent / "render_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["input_trace_digest"] = "sha256:" + "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        failures = _check(tmp_path, ctx, gif_path=str(gif))
        assert any("RENDER_INPUT_MISMATCH" in f for f in failures), failures


class TestSe3AcceptanceChecks:
    def test_tool_axis_deviation_rejected(self, tmp_path: Path) -> None:
        ctx = _run_pipeline(tmp_path)
        metrics_path = Path(ctx["run"]["artifacts"]["metrics_json"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["max_orientation_error_deg"] = 90.0
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        failures = _check(tmp_path, ctx)
        assert any("TOOL_AXIS_DEVIATION" in f for f in failures), failures

    def test_orientation_metrics_required_for_spec_trace(
        self, tmp_path: Path
    ) -> None:
        """trace 锚定 SE(3) 规格（spec_digest）但 metrics 无朝向
        指标——验收依据缺失，诚实拒绝。"""
        ctx = _run_pipeline(tmp_path)
        metrics_path = Path(ctx["run"]["artifacts"]["metrics_json"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.pop("max_orientation_error_deg", None)
        metrics.pop("mean_orientation_error_deg", None)
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        failures = _check(tmp_path, ctx)
        assert any("ORIENTATION_METRICS_MISSING" in f for f in failures), failures

    def test_contact_height_deviation_rejected(self, tmp_path: Path) -> None:
        """实际轨迹悬空（接触段 z 偏离接触平面）——没画在平面上。"""
        ctx = _run_pipeline(tmp_path)
        trace_path = Path(ctx["run"]["artifacts"]["trace_json"])
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        for pt in trace["actual"]:
            pt["z"] += 0.10
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        failures = _check(tmp_path, ctx)
        assert any("CONTACT_HEIGHT_DEVIATION" in f for f in failures), failures


class TestVerifyDrawingHonesty:
    def test_verify_drawing_marked_not_independent(self) -> None:
        """kit 配置必须为 ur5e.verify_drawing 声明诚实标注——
        命令回放校验不是独立验收证据。"""
        from rosclaw.sim.robot_kit import kit_for_body, kit_server_spec

        kit = kit_for_body("sim/ur5e")
        assert kit is not None
        spec = kit_server_spec(kit)
        notes = spec.get("verifier_notes") or {}
        note = str(notes.get("ur5e.verify_drawing", ""))
        assert "NOT_INDEPENDENT" in note, "verify_drawing 缺 NOT_INDEPENDENT 标注"
        assert "COMMAND_REPLAY" in note, "verify_drawing 缺 COMMAND_REPLAY 标注"

    def test_verify_drawing_description_honest(self) -> None:
        """模型可见的工具描述必须声明非独立性（模型按描述决策
        证据强度）。"""
        import inspect

        import rosclaw.sim.ur5e_mcp as ur5e_mcp

        src = inspect.getsource(ur5e_mcp.verify_drawing)
        assert "NOT_INDEPENDENT" in src or "非独立" in src, (
            "verify_drawing 描述未声明非独立证据属性"
        )
