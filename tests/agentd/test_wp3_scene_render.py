"""WP-3 红测试（0823 审计 §四.WP-3）：MuJoCo 渲染做成 ROSClaw 正式能力。

红测试先行——sim_render 能力不存在时必须红。

1. simulation.render 正式能力：canonical MJCF + trajectory state
   replay + 相机预设 + GIF 输出 + RenderReceipt（renderer build
   digest + input trace digest + backend）；
2. 后端探测子进程隔离（EGL→OSMesa→Xvfb；进程内探测会把宿主搞崩
   ——本机实证 glfw 递归初始化崩 libc++abi）；
3. 画面真实非空白（像素方差 > 0——不是全黑帧）；
4. 输入篡改/缺失诚实失败（RENDER_INPUT_MISSING / digest 不符）；
5. 能力进 catalog + snapshot（accepts trace ref）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_trace(home: Path) -> dict:
    """真实 rollout 出一份轨迹（受信管线的既有能力）。"""
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService

    sim = SimTrajectoryService(home)
    plan = sim.generate_planar_path(
        shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
    )
    return sim.simulate_cartesian_trajectory(plan["plan_id"])


class TestSceneRender:
    def test_render_produces_real_gif_with_receipt(self, tmp_path: Path) -> None:
        from rosclaw.agentd.sim_render import render_scene_trace

        run = _make_trace(tmp_path)
        result = render_scene_trace(tmp_path, run["trace_id"])
        gif = Path(result["artifact"]["path"])
        assert gif.exists() and gif.stat().st_size > 10_000
        assert result["artifact"]["frames"] >= 30
        receipt = result["receipt"]
        assert receipt["backend"] in ("egl", "osmesa", "xvfb")
        assert receipt["renderer_build_digest"].startswith("sha256:")
        assert receipt["input_trace_digest"].startswith("sha256:")
        assert receipt["camera"] in ("follow", "free", "top")
        # 画面真实非空白：帧像素方差 > 0。
        from PIL import Image

        img = Image.open(gif)
        frame = img.convert("L")
        import numpy as np

        assert float(np.asarray(frame).std()) > 1.0, "渲染帧全黑/全白——不是真实场景"

    def test_backend_probe_subprocess_isolated(self, tmp_path: Path) -> None:
        """探测不得崩宿主进程（本机实证 glfw 进程内递归初始化崩）。"""
        from rosclaw.agentd.sim_render import probe_render_backend

        backend, detail = probe_render_backend()
        assert backend in ("egl", "osmesa", "xvfb"), detail
        assert isinstance(detail, dict) and detail, "探测明细缺失"

    def test_missing_input_honest_failure(self, tmp_path: Path) -> None:
        from rosclaw.agentd.sim_render import render_scene_trace

        with pytest.raises((ValueError, FileNotFoundError), match="RENDER_INPUT"):
            render_scene_trace(tmp_path, "trace_nonexistent")

    def test_tampered_states_rejected(self, tmp_path: Path) -> None:
        """trajectory_states 被换（digest 不符 trace）→ 诚实失败。"""
        from rosclaw.agentd.sim_render import render_scene_trace

        run = _make_trace(tmp_path)
        states_path = (
            tmp_path / "sim" / "traces" / run["trace_id"]
            / "trajectory_states.json"
        )
        states = json.loads(states_path.read_text(encoding="utf-8"))
        states["states"][10]["qpos"][0] += 5.0  # 篡改一帧
        states_path.write_text(json.dumps(states), encoding="utf-8")
        with pytest.raises(ValueError, match="RENDER_INPUT|digest"):
            render_scene_trace(tmp_path, run["trace_id"])


class TestSceneRenderCapability:
    def test_registered_with_refs_and_schema(self, tmp_path: Path) -> None:
        """能力入 catalog：output_schema 非空 + accepts trace ref +
        produces render ref；snapshot 可见。"""
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.agentd.tooling.native_tools import register_native_tools
        from rosclaw.agentd.tooling.snapshot import build_capability_snapshot
        from rosclaw.agentd.tools import BuiltinToolRegistry

        catalog = ToolCatalog()
        register_native_tools(
            catalog,
            BuiltinToolRegistry(body_id="sim/ur5e", body_summary="UR5e"),
            simulation=True,
        )
        cap = catalog.capability("simulation_render_scene")
        assert cap is not None, "simulation_render_scene 未注册"
        assert cap.output_schema, "output_schema 空"
        kinds_in = {r.get("kind") for r in cap.accepts_refs}
        kinds_out = {r.get("kind") for r in cap.produces_refs}
        assert "trace" in kinds_in
        assert "render" in kinds_out
        snap = build_capability_snapshot(
            catalog, body_id="sim/ur5e", mode="SIMULATION"
        )
        active = {a.capability_id for a in snap.active}
        assert "simulation_render_scene" in active
