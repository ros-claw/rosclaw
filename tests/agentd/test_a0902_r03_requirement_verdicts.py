"""0902 审计 R0-3 红测试：逐条 Requirement verdict——每个材料性要求
一条判定；未满足/不可证 → 不得 PASS（反假成功故障注入）。

0902 实证（P0 事故）：新 revision 要求"红色圆柱笔 + 3D 实际轨迹 +
不要 2D"，receipt 仍 tool_ref=""/overlays=[]，系统宣布 PASS/DELIVERED。

闭环断言（故障注入矩阵）：
1. spec 声明持笔 + receipt tool_ref="" → REQUIREMENT_UNMET，不 PASS；
2. 要求轨迹叠加 + receipt 无 overlays → 不 PASS；
3. 禁止纯 2D + 只有 2D 产物 → 不 PASS；
4. 颜色要求（无证据通道）→ UNVERIFIABLE → 不 PASS（诚实——
   不可证不冒充）；
5. 全满足（tool_ref 匹配 + scene_3d 在账 + 轨迹在账）→ 正常 PASS
   （回归护栏）。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _resource() -> dict:
    """真实权威 manifest 资源证明（与 n0/wp4 测试同一形态）。"""
    from rosclaw.cognition.resolver import resolve_resource

    repo = Path(__file__).resolve().parents[2]
    manifest = resolve_resource("robot", "ur5e", product_root=repo)
    assert manifest is not None
    return {
        "resource_id": "robot:ur5e",
        "manifest_digest": manifest["digests"].get("profile", ""),
        "model_path": manifest["paths"]["mjcf"],
        "model_digest": manifest["digests"]["mjcf"],
        "quality": "PRODUCTION",
        "canonical": True,
    }


def _kernel(tmp_path: Path):
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, tmp_path)


def _write_receipt(tmp_path: Path, *, tool_ref: str = "",
                   with_overlays: bool = False) -> Path:
    """render receipt 故障注入点：tool_ref/overlays 由测试控制。"""
    receipt = {
        "schema_version": "rosclaw.render_receipt.v1",
        "backend": "osmesa",
        "camera": "follow",
        "world_id": "empty",
        "tool_ref": tool_ref,
        "input_trace_digest": "sha256:" + "0" * 64,
        "outputs": ["gif", "mp4"],
    }
    if with_overlays:
        receipt["overlays"] = ["actual_eef_trace"]
    path = tmp_path / "render_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _chain_artifacts(kernel, task_id: str, tmp_path: Path,
                     receipt: Path) -> list[str]:
    """当前 revision 的最小受信产物集：trace + 场景视频（带血缘）。"""
    # plan 记录（shape/plane 验收的权威源——trace.plan_hash 链接）。
    plan_hash = "0" * 32
    plans = tmp_path / "sim" / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    (plans / f"plan_{plan_hash[:16]}.json").write_text(json.dumps({
        "shape": "star5", "plane": "xy", "hash": plan_hash,
    }), encoding="utf-8")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({
        "schema_version": "rosclaw.trace.v1",
        "plan_hash": plan_hash,
    }), encoding="utf-8")
    t = kernel.register_artifact(
        task_id=task_id, path=str(trace), media_type="application/json",
        producer="kernel:capability:ur5e.simulate",
        metadata={"resource": _resource()},
    )
    mp4 = tmp_path / "scene.mp4"
    mp4.write_bytes(b"MP4" + b"x" * 4096)
    m = kernel.register_artifact(
        task_id=task_id, path=str(mp4), media_type="video/mp4",
        producer="kernel:sim_render",
        metadata={"resource": _resource(),
                  "lineage": {"render_receipt_path": str(receipt),
                              "kind": "scene_3d"}},
    )
    return [str(t["artifact_id"]), str(m["artifact_id"])]


def _finish(kernel, task_id: str, artifact_ids: list[str]):
    return kernel.finish_task(
        task_id=task_id, summary="done", artifact_ids=artifact_ids,
    )


class TestPerRequirementVerdict:
    def test_empty_tool_ref_blocks_pass(self, tmp_path: Path) -> None:
        """0902 事故面：spec 声明持笔（tool:pen）+ receipt
        tool_ref="" → 不得 PASS。"""
        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_1", text="画五角星，机械臂末端持笔",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        task_id = str(bound["task_id"])
        kernel.note_tool_use(task_id, "rosclaw_task")
        receipt = _write_receipt(tmp_path, tool_ref="")
        ids = _chain_artifacts(kernel, task_id, tmp_path, receipt)
        result = _finish(kernel, task_id, ids)
        assert result["status"] != "SUCCEEDED", result
        assert any("REQUIREMENT_UNMET" in str(f) for f in
                   result.get("failures", [])), result

    def test_missing_overlay_blocks_pass(self, tmp_path: Path) -> None:
        """要求 3D 画面叠加实际轨迹 + receipt 无 overlays → 不 PASS。"""
        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_1",
            text="画五角星视频，在 3D 画面里显示本次实际轨迹",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        task_id = str(bound["task_id"])
        kernel.note_tool_use(task_id, "rosclaw_task")
        receipt = _write_receipt(tmp_path)  # 无 overlays 字段
        ids = _chain_artifacts(kernel, task_id, tmp_path, receipt)
        result = _finish(kernel, task_id, ids)
        assert result["status"] != "SUCCEEDED", result
        assert any("REQUIREMENT_UNMET" in str(f) for f in
                   result.get("failures", [])), result

    def test_forbid_2d_only_with_only_2d_blocks(self, tmp_path: Path) -> None:
        """"不要 2D" + 只有 2D 预览 → 禁止项违反，不 PASS。"""
        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_1", text="画五角星，不要 2D",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        task_id = str(bound["task_id"])
        kernel.note_tool_use(task_id, "rosclaw_task")
        # 只产 2D 预览（无场景视频）。
        gif = tmp_path / "preview.gif"
        gif.write_bytes(b"GIF89a" + b"x" * 2048)
        art = kernel.register_artifact(
            task_id=task_id, path=str(gif), media_type="image/gif",
            producer="kernel:sim_render",
            metadata={"resource": _resource(),
                      "lineage": {"kind": "preview_2d",
                                  "trace_id": "t1"}},
        )
        result = _finish(kernel, task_id, [str(art["artifact_id"])])
        assert result["status"] != "SUCCEEDED", result
        assert any("REQUIREMENT_UNMET" in str(f) for f in
                   result.get("failures", [])), result

    def test_color_requirement_unverifiable_blocks(self, tmp_path: Path) -> None:
        """颜色要求今天没有证据通道 → UNVERIFIABLE → 不 PASS
        （不可证不冒充——能力缺口诚实暴露）。"""
        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_1", text="画五角星，笔要红色",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        task_id = str(bound["task_id"])
        kernel.note_tool_use(task_id, "rosclaw_task")
        receipt = _write_receipt(tmp_path, tool_ref="tool:pen")
        ids = _chain_artifacts(kernel, task_id, tmp_path, receipt)
        result = _finish(kernel, task_id, ids)
        assert result["status"] != "SUCCEEDED", result
        assert any("颜色" in str(f) or "tool_color" in str(f)
                   for f in result.get("failures", [])), result

    def test_all_satisfied_passes(self, tmp_path: Path) -> None:
        """回归护栏：全条款满足 → 正常 PASS。"""
        kernel = _kernel(tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_1", text="画五角星视频",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        task_id = str(bound["task_id"])
        kernel.note_tool_use(task_id, "rosclaw_task")
        receipt = _write_receipt(tmp_path)
        ids = _chain_artifacts(kernel, task_id, tmp_path, receipt)
        result = _finish(kernel, task_id, ids)
        assert result["status"] == "SUCCEEDED", result.get("failures")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
