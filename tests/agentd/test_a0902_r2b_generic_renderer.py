"""0902 审计 R2-2 红测试：通用渲染器 adapter——RenderSpec 驱动
（§4.3/R2.4），去本体名/形状硬编码。

现状实证（sim_render.py）：Sandbox.create("ur5e", …) 本体名硬编码；
overlay 概念不存在（trace 只用于取景，画面不画轨迹）；outputs 永远
gif+mp4 不理会声明；receipt 无 body/spec 锚点。

闭环断言：
1. render_from_spec：spec.body_ref → RenderProfile 注册表解析本体
   （不再 hardcode ur5e）；未知本体 → RENDER_PROFILE_MISSING 诚实
   失败；
2. actual_eef_trace overlay 真画进画面（有/无 overlay 的帧像素差
   > 阈值——不是宣称画了）；planned_trace/waypoints/contact_points
   同样真实绘制；
3. overlay 证据绑定：actual_eef_trace 的 source_ref 必须等于本次
   trace（trace:<id>），否则 RENDER_EVIDENCE_MISMATCH——旧轨迹
   overlay 不得冒充本次（0902 假成功教训）；
4. plan 源 overlay（planned_trace/waypoints/contact_points）的
   source_ref 必须匹配 trace 的 plan_hash；
5. spec.outputs 被遵守（["mp4"] 不出 gif）；receipt 记录 body_ref/
   spec_digest/overlays_applied；
6. safety_zone/sensor_frustum 未实现 → RENDER_OVERLAY_UNSUPPORTED
   诚实失败（不静默跳过宣称已画）。
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


def _spec(trace_id: str, **over) -> dict:
    base = {
        "body_ref": "robot:sim/ur5e",
        "world_ref": "world:tabletop",
        "outputs": ["gif"],
    }
    base.update(over)
    return base


def _render(home: Path, spec_dict: dict, trace_id: str) -> dict:
    from rosclaw.agentd.sim_render import render_from_spec
    from rosclaw.contracts.agent.render_spec import RenderSpecV1

    return render_from_spec(home, RenderSpecV1(**spec_dict), trace_id)


def _render_fast(home: Path, spec_dict: dict, trace_id: str) -> dict:
    """小帧数小分辨率（CI 软件光栅慢一个量级——像素证据不依赖
    帧数）。"""
    from rosclaw.agentd.sim_render import render_from_spec
    from rosclaw.contracts.agent.render_spec import RenderSpecV1

    return render_from_spec(
        home, RenderSpecV1(**spec_dict), trace_id,
        max_frames=12, width=320, height=180,
    )


class TestRenderProfileResolution:
    def test_unknown_body_honest_failure(self, tmp_path: Path) -> None:
        run = _make_trace(tmp_path)
        with pytest.raises(ValueError, match="RENDER_PROFILE_MISSING"):
            _render(tmp_path, _spec(run["trace_id"], body_ref="robot:sim/atlas"), run["trace_id"])

    def test_profile_registry_has_ur5e(self) -> None:
        from rosclaw.agentd.render_profiles import resolve_render_profile

        profile = resolve_render_profile("sim/ur5e")
        assert profile.eef_frame
        assert profile.qpos_mapping


class TestOverlayRendering:
    def test_actual_trace_overlay_actually_drawn(self, tmp_path: Path) -> None:
        run = _make_trace(tmp_path)
        tid = run["trace_id"]
        plain = _render_fast(tmp_path, _spec(tid), tid)
        # 两次渲染写同一路径（trace 目录）——必须先读 plain 帧再渲染
        # traced，否则比较的是同一份文件（实证：diff=0 假阴性）。
        import numpy as np
        from PIL import Image

        a = np.asarray(
            Image.open(plain["artifacts"]["gif"]["path"]).convert("L"), dtype=float
        ).copy()
        traced = _render_fast(
            tmp_path,
            _spec(tid, overlays=[{
                "kind": "actual_eef_trace", "source_ref": f"trace:{tid}",
            }]),
            tid,
        )
        assert "actual_eef_trace" in traced["receipt"]["overlays_applied"]
        # 画面证据：overlay 帧与无 overlay 帧必须有像素差（真画了）。
        b = np.asarray(
            Image.open(traced["artifacts"]["gif"]["path"]).convert("L"), dtype=float
        )
        assert float(np.abs(a - b).mean()) > 0.5, "overlay 宣称画了但画面无差异"

    def test_overlay_evidence_binding_rejects_foreign_trace(self, tmp_path: Path) -> None:
        run = _make_trace(tmp_path)
        tid = run["trace_id"]
        with pytest.raises(ValueError, match="RENDER_EVIDENCE_MISMATCH"):
            _render(
                tmp_path,
                _spec(tid, overlays=[{
                    "kind": "actual_eef_trace", "source_ref": "trace:other",
                }]),
                tid,
            )

    def test_plan_sourced_overlays(self, tmp_path: Path) -> None:
        run = _make_trace(tmp_path)
        tid = run["trace_id"]
        plan_hash = json.loads(
            (tmp_path / "sim" / "traces" / tid / "trace.json").read_text()
        )["plan_hash"]
        result = _render(
            tmp_path,
            _spec(tid, overlays=[
                {"kind": "planned_trace", "source_ref": f"plan:{plan_hash[:16]}"},
                {"kind": "waypoints", "source_ref": f"plan:{plan_hash[:16]}"},
                {"kind": "contact_points", "source_ref": f"plan:{plan_hash[:16]}"},
            ]),
            tid,
        )
        applied = result["receipt"]["overlays_applied"]
        assert "planned_trace" in applied
        assert "waypoints" in applied
        assert "contact_points" in applied

    def test_plan_overlay_foreign_plan_rejected(self, tmp_path: Path) -> None:
        run = _make_trace(tmp_path)
        tid = run["trace_id"]
        with pytest.raises(ValueError, match="RENDER_EVIDENCE_MISMATCH"):
            _render(
                tmp_path,
                _spec(tid, overlays=[{
                    "kind": "planned_trace", "source_ref": "plan:deadbeefdeadbeef",
                }]),
                tid,
            )

    def test_unsupported_overlay_honest_failure(self, tmp_path: Path) -> None:
        run = _make_trace(tmp_path)
        tid = run["trace_id"]
        with pytest.raises(ValueError, match="RENDER_OVERLAY_UNSUPPORTED"):
            _render(
                tmp_path,
                _spec(tid, overlays=[{"kind": "safety_zone"}]),
                tid,
            )


class TestSpecDrivenOutputs:
    def test_outputs_honored(self, tmp_path: Path) -> None:
        run = _make_trace(tmp_path)
        tid = run["trace_id"]
        result = _render(tmp_path, _spec(tid, outputs=["mp4"]), tid)
        assert "mp4" in result["artifacts"]
        assert "gif" not in result["artifacts"], "声明只要 mp4 却出了 gif"

    def test_receipt_carries_spec_anchor(self, tmp_path: Path) -> None:
        run = _make_trace(tmp_path)
        tid = run["trace_id"]
        result = _render(tmp_path, _spec(tid), tid)
        receipt = result["receipt"]
        assert receipt["body_ref"] == "robot:sim/ur5e"
        assert receipt["spec_digest"].startswith("sha256:")
        assert receipt["overlays_applied"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
