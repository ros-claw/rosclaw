"""0914 PR-2：轨迹能力三层接通——schema/dispatch/receipt。

实证断点（main=e4e420a 代码核验）：
1. schema 层：simulation_render_scene 的 input_schema 只有
   {trace_id, camera} 且 additionalProperties: False——模型无从
   表达 overlays（sim_render 的 actual_eef_trace 能力对模型不可见，
   "内置工具不支持轨迹"从模型视角是事实）；
2. dispatch 层：_execute_scene_render 永不传 render_spec_path——
   overlay 到不了 renderer；
3. 结果层：receipt 只有 overlays_applied——无 requested/unfulfilled，
   模型无从知道"要的轨迹没画上"（平面投影冒充三维轨迹的土壤）。

闭环断言：
- schema 暴露 overlays 且 kind 枚举与 sim_render 支持集单一来源；
- dispatch 把 overlays/outputs/fps/playback 构造进 RenderSpec 并传入；
- 真实渲染画出 actual_eef_trace（帧像素证据）+ receipt 三字段；
- 请求了但画不上的 overlay 必须进 unfulfilled（不 ok=true 静默）；
- animated 揭示按原始 sample index（禁抽稀后 linspace 重造时间）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_trace(home: Path) -> dict:
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService

    sim = SimTrajectoryService(home)
    plan = sim.generate_planar_path(
        shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
    )
    return sim.simulate_cartesian_trajectory(plan["plan_id"])


def _registry(home: Path):
    from rosclaw.agentd.tools import BuiltinToolRegistry

    return BuiltinToolRegistry(body_id="sim/ur5e", body_summary="", home=home)


class TestSchemaExposesOverlays:
    def test_overlay_kinds_single_source(self) -> None:
        """模型面 schema 的 overlay kind 枚举必须等于 sim_render 的
        支持集（单一来源——防两处手工抄漂移）。"""
        from rosclaw.agentd import sim_render
        from rosclaw.agentd.tools import _TOOL_SCHEMAS

        schema = _TOOL_SCHEMAS["simulation_render_scene"]
        props = schema.parameters["properties"]
        assert "overlays" in props, "模型面 schema 无 overlays——能力不可见"
        enum = set(props["overlays"]["items"]["properties"]["kind"]["enum"])
        assert enum == set(sim_render.SUPPORTED_OVERLAY_KINDS)

    def test_native_descriptor_same_source(self) -> None:
        """native_tools 的 ToolDescriptor 与 sim_render 同一来源。"""
        from rosclaw.agentd import sim_render
        from rosclaw.agentd.tooling import native_tools

        fragment = native_tools._scene_render_input_fragment()
        enum = set(fragment["overlays"]["items"]["properties"]["kind"]["enum"])
        assert enum == set(sim_render.SUPPORTED_OVERLAY_KINDS)


class TestDispatchConstructsSpec:
    def test_overlays_reach_renderer(self, tmp_path: Path, monkeypatch) -> None:
        """dispatch 必须把 overlays 构造进 RenderSpec 并传
        render_spec_path（旧码永不传——overlay 断在 dispatch）。"""
        captured: dict = {}

        def _fake_render(home, trace_id, **kw):
            captured.update(kw)
            return {
                "ok": True, "artifact": {"path": "x.gif"},
                "receipt": {
                    "overlays_requested": ["actual_eef_trace"],
                    "overlays_applied": ["actual_eef_trace"],
                    "overlays_unfulfilled": [],
                },
            }

        monkeypatch.setattr(
            "rosclaw.agentd.sim_render.render_scene_trace", _fake_render
        )
        run = _make_trace(tmp_path)
        result = _registry(tmp_path)._execute_scene_render({
            "trace_id": run["trace_id"],
            "camera": "follow",
            "overlays": [{"kind": "actual_eef_trace", "presentation": "full"}],
        })
        spec_path = captured.get("render_spec_path")
        assert spec_path, "dispatch 未传 render_spec_path"
        doc = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        assert doc["overlays"] == [
            {"kind": "actual_eef_trace", "presentation": "full"}
        ]
        assert result["overlays_applied"] == ["actual_eef_trace"]
        assert result["overlays_unfulfilled"] == []


class TestEndToEndOverlay:
    def test_actual_trace_drawn_and_reported(self, tmp_path: Path) -> None:
        """真实渲染：actual_eef_trace 真画进画面（帧像素证据）+
        receipt 三字段一致。"""
        run = _make_trace(tmp_path)
        result = _registry(tmp_path)._execute_scene_render({
            "trace_id": run["trace_id"],
            "camera": "follow",
            "overlays": [{"kind": "actual_eef_trace"}],
            "outputs": ["gif"],
        })
        assert result["ok"] is True
        receipt = result["receipt"]
        assert receipt["overlays_requested"] == ["actual_eef_trace"]
        assert receipt["overlays_applied"] == ["actual_eef_trace"]
        assert receipt["overlays_unfulfilled"] == []
        # 帧像素证据：actual_eef_trace 的 rgba=(1.0,0.2,0.2)——
        # 画面必须出现红系 overlay 像素（宣称≠画面，0902 教训）。
        import imageio.v3 as iio
        import numpy as np

        gif_path = result["artifacts"]["gif"]["path"]
        frames = [np.asarray(f) for f in iio.imiter(gif_path)]
        assert frames, "无帧"
        red_hits = 0
        for frame in frames[len(frames) // 2 :]:
            r = frame[:, :, 0].astype(int)
            g = frame[:, :, 1].astype(int)
            b = frame[:, :, 2].astype(int)
            red_hits += int(((r > 150) & (r > g + 60) & (r > b + 60)).sum())
        assert red_hits > 200, (
            f"overlay 像素几乎不可见（red_hits={red_hits}）——画了但"
            "被裁/太细等于没画（0914 审计：数值之外必须查帧）"
        )

    def test_unfulfilled_is_honest(self, tmp_path: Path) -> None:
        """请求了 planned_trace 但 trace 无 planned 数据——必须进
        overlays_unfulfilled 带原因（不能 ok=true 让模型误认完成）。"""
        run = _make_trace(tmp_path)
        # 抹掉 planned 数据（模拟数据缺失场景）。
        trace_path = (
            tmp_path / "sim" / "traces" / run["trace_id"] / "trace.json"
        )
        doc = json.loads(trace_path.read_text(encoding="utf-8"))
        doc["planned"] = []
        trace_path.write_text(json.dumps(doc), encoding="utf-8")
        result = _registry(tmp_path)._execute_scene_render({
            "trace_id": run["trace_id"],
            "camera": "follow",
            "overlays": [{"kind": "planned_trace"}],
            "outputs": ["gif"],
        })
        assert result["ok"] is True  # 渲染本身成功
        unfulfilled = result["overlays_unfulfilled"]
        assert [u["kind"] for u in unfulfilled] == ["planned_trace"]
        assert unfulfilled[0]["reason"], "unfulfilled 必须带原因"
        assert result["overlays_applied"] == []


class TestAnimatedReveal:
    def test_reveal_cutoff_uses_original_sample_index(self) -> None:
        """animated 揭示按原始 sample index 截断——抽稀保持索引，
        禁止 linspace 重造时间关联（0914 审计 §4 时间条款）。"""
        from rosclaw.agentd.sim_render import _reveal_prefix

        # 100 个采样点（原始 index 0..99），抽稀到 10 个保留点。
        pts = [(float(i), 0.0, 0.0) for i in range(100)]
        indexed = list(enumerate(pts))
        # 第 50 个原始采样时刻——揭示必须恰好包含 index<=50 的保留点。
        shown = _reveal_prefix(indexed, 50)
        assert shown, "空揭示"
        assert max(i for i, _ in shown) <= 50
        # 保留点的 index 集合必须是原始 index 的子集（不是 linspace
        # 重排）——最后一个被揭示点之后还有未揭示保留点。
        all_indices = {i for i, _ in indexed}
        shown_indices = {i for i, _ in shown}
        assert shown_indices < all_indices
        assert min(all_indices - shown_indices) > 50

    def test_animated_frame_growth(self, tmp_path: Path) -> None:
        """animated 模式：后段帧的 overlay 像素必须多于前段帧
        （轨迹随时间揭示——不是每帧全量静态）。"""
        run = _make_trace(tmp_path)
        result = _registry(tmp_path)._execute_scene_render({
            "trace_id": run["trace_id"],
            "camera": "follow",
            "overlays": [{"kind": "actual_eef_trace", "presentation": "animated"}],
            "outputs": ["gif"],
        })
        assert result["overlays_applied"] == ["actual_eef_trace"]
        import imageio.v3 as iio
        import numpy as np

        frames = [np.asarray(f) for f in iio.imiter(result["artifacts"]["gif"]["path"])]

        def _red(frame) -> int:
            r = frame[:, :, 0].astype(int)
            g = frame[:, :, 1].astype(int)
            b = frame[:, :, 2].astype(int)
            return int(((r > 150) & (r > g + 60) & (r > b + 60)).sum())

        early = sum(_red(f) for f in frames[1:3])
        late = sum(_red(f) for f in frames[-3:-1])
        assert late > early, (
            f"animated 未随时间揭示（early={early} late={late}）"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
