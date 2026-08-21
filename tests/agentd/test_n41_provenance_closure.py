"""PR-N4.1 红测试：执行资源证明闭环（调整方案 §二）。

红测试先行——修复前必须红：
1. 正式 UR5e 内容被换成 fixture 内容（文件名不变）→ 验收必须
   RESOURCE_DIGEST_MISMATCH；
2. fixture 复制到工作区再跑 → 仍不能冒充正式 UR5e
   （RESOURCE_ID_MISMATCH/NON_CANONICAL_RESOURCE）；
3. SimTrajectoryService 执行 receipt/产物元数据携带完整资源证明
   （resource_id/manifest_digest/model_path/model_digest/quality/
   canonical），可反查 manifest；
4. Verifier 只读 receipt/evidence 元数据——模型口头声明的路径不算；
5. 模型手写脚本产物标 EXPERIMENTAL_EVIDENCE（不得当正式能力证据）。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _kernel(tmp_path: Path):
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, tmp_path)


def _task(kernel, tmp_path: Path) -> str:
    bound = kernel.bind_message(
        mission_id="mis_1", session_ref="s1", backend_native_id="s1",
        message_id="m1", text="画五角星", cwd=str(tmp_path),
        mode="SIMULATION", body_id="sim/ur5e",
    )
    task_id = str(bound["task_id"])
    kernel.note_tool_use(task_id, "rosclaw_task")  # 行为任务标记
    return task_id


def _register_with_resource(kernel, task_id: str, tmp_path: Path,
                            resource: dict | None) -> dict:
    task = kernel.get_task(task_id)
    assert task is not None
    gif = Path(task["workspace_path"]) / "star.gif"
    gif.write_bytes(b"GIF89a" + b"x" * 2048)
    return kernel.register_artifact(
        task_id=task_id, path=str(gif), media_type="image/gif",
        producer="kernel:sim_pipeline",
        metadata={"resource": resource} if resource else None,
    )


class TestProvenanceClosure:
    def test_sim_receipt_carries_resource_proof(self, tmp_path: Path) -> None:
        """执行 receipt/trace 携带完整资源证明——可反查 manifest。"""
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        sim = SimTrajectoryService(tmp_path)
        plan = sim.generate_planar_path(shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05)
        result = sim.simulate_cartesian_trajectory(plan["plan_id"])
        trace = json.loads(
            Path(result["artifacts"]["trace_json"]).read_text(encoding="utf-8")
        )
        resource = trace.get("resource") or {}
        for key in ("resource_id", "manifest_digest", "model_path",
                    "model_digest", "quality", "canonical"):
            assert key in resource, f"receipt 缺 {key}"
        assert resource["resource_id"] == "robot:ur5e"
        assert resource["model_digest"].startswith("sha256:")
        assert resource["quality"] == "PRODUCTION"
        assert resource["canonical"] is True
        # 反查：model_digest 必须与当前权威 manifest 一致。
        from rosclaw.cognition.resolver import resolve_resource

        manifest = resolve_resource("robot", "ur5e", product_root=REPO)
        assert manifest is not None
        assert manifest["digests"]["mjcf"] == resource["model_digest"]

    def test_swapped_canonical_content_fails_verification(
        self, tmp_path: Path
    ) -> None:
        """正式 UR5e 文件内容被换成 fixture 内容（文件名不变）→
        RESOURCE_DIGEST_MISMATCH。"""

        kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        # 受信产物带资源证明——但 model_digest 是 fixture 的 digest
        # （模拟内容被换）。
        fixture_xml = REPO / "tests" / "fixtures" / "ur5e_minimal_fixture.xml"
        import hashlib

        fixture_digest = "sha256:" + hashlib.sha256(
            fixture_xml.read_bytes()
        ).hexdigest()
        art = _register_with_resource(kernel, task_id, tmp_path, resource={
            "resource_id": "robot:ur5e",
            "manifest_digest": "sha256:" + "ab" * 32,
            "model_path": str(REPO / "e-urdf-zoo" / "ur5e" / "robot.mjcf.xml"),
            "model_digest": fixture_digest,  # 内容是 fixture 的
            "quality": "PRODUCTION",
            "canonical": True,
        })
        result = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        assert result["status"] != "SUCCEEDED"
        assert any(
            "RESOURCE_DIGEST_MISMATCH" in str(f)
            for f in result.get("failures", [])
        ), result

    def test_fixture_copied_to_workspace_cannot_impersonate(
        self, tmp_path: Path
    ) -> None:
        """fixture 复制到工作区运行——声明 robot:ur5e 但 digest 是
        fixture 的 → 失败；声明 fixture 身份 → NON_CANONICAL_RESOURCE。"""
        kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        art = _register_with_resource(kernel, task_id, tmp_path, resource={
            "resource_id": "robot:ur5e_minimal_fixture",
            "manifest_digest": "sha256:" + "cd" * 32,
            "model_path": str(tmp_path / "ws" / "copied.xml"),
            "model_digest": "sha256:" + "ef" * 32,
            "quality": "TEST_FIXTURE",
            "canonical": False,
        })
        result = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        assert result["status"] != "SUCCEEDED"
        assert any(
            "NON_CANONICAL_RESOURCE" in str(f) or "RESOURCE_ID_MISMATCH" in str(f)
            for f in result.get("failures", [])
        ), result

    def test_missing_provenance_fails(self, tmp_path: Path) -> None:
        """行为任务产物无资源证明 → RESOURCE_PROVENANCE_MISSING
        （producer=kernel 前缀本身不再是充分条件）。"""
        kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        art = _register_with_resource(kernel, task_id, tmp_path, resource=None)
        result = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        assert result["status"] != "SUCCEEDED"
        assert any(
            "RESOURCE_PROVENANCE_MISSING" in str(f)
            for f in result.get("failures", [])
        ), result

    def test_model_authored_script_marked_experimental(
        self, tmp_path: Path
    ) -> None:
        """模型手写脚本产物标 EXPERIMENTAL_EVIDENCE——不得当正式
        能力证据。"""
        kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        task = kernel.get_task(task_id)
        assert task is not None
        gif = Path(task["workspace_path"]) / "star.gif"
        gif.write_bytes(b"GIF89a" + b"x" * 2048)
        art = kernel.register_artifact(
            task_id=task_id, path=str(gif), media_type="image/gif",
            producer="model:rosclaw_artifact_register",
        )
        meta = json.loads(str(art.get("metadata_json", "{}")) or "{}")
        assert meta.get("evidence_tier") == "EXPERIMENTAL", meta
