"""PR-N4 红测试：Resource Resolver 与 fixture 清理（N 总纲 §8/§PR-N4）。

红测试先行——修复前必须红：
1. 三种资产（ur5e/franka_panda/g1）解析出 ResourceManifestV1——
   全路径存在 + digest + quality=PRODUCTION；
2. production resolver 永不返回 test fixture；fixture 有 manifest
   标记 quality=TEST_FIXTURE、canonical=False；
3. Verifier：机器人任务产物引用 fixture 路径 → RESOURCE_PROVENANCE_FAILED；
4. Sandbox 返回 resource ID + model path + digest；
5. `inspect asset ur5e --json` 返回 manifest 形状。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class TestResourceResolver:
    def test_three_robots_resolve_to_production_manifests(
        self, tmp_path: Path
    ) -> None:
        from rosclaw.cognition.resolver import resolve_resource

        for robot_id in ("ur5e", "franka_panda", "g1"):
            manifest = resolve_resource("robot", robot_id, product_root=REPO)
            assert manifest is not None, f"{robot_id} 未解析"
            assert manifest["canonical"] is True
            assert manifest["quality"] == "PRODUCTION"
            assert manifest["trust"] == "ROSCLAW_OFFICIAL"
            assert manifest["source"] == "e-urdf-zoo"
            for key, path in manifest["paths"].items():
                assert Path(path).exists(), f"{robot_id}.{key} 不存在: {path}"
            assert manifest["digests"], f"{robot_id} 缺 digest"

    def test_production_resolver_never_returns_fixture(self) -> None:
        from rosclaw.cognition.resolver import resolve_resource

        # 简化模型在测试夹具里——以 fixture id 解析：production_only
        # 默认开，返回 None。
        manifest = resolve_resource(
            "robot", "ur5e_minimal_fixture", product_root=REPO
        )
        assert manifest is None, "production resolver 竟返回 test fixture"
        # 显式允许 fixture 时返回 TEST_FIXTURE 标记（测试场景可用）。
        fixture = resolve_resource(
            "robot", "ur5e_minimal_fixture", product_root=REPO,
            production_only=False,
        )
        assert fixture is not None
        assert fixture["quality"] == "TEST_FIXTURE"
        assert fixture["canonical"] is False

    def test_fixture_manifest_exists_and_marked(self) -> None:
        path = REPO / "tests" / "fixtures" / "ur5e_minimal_fixture.yaml"
        assert path.exists(), "fixture manifest 未建立"
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["quality"] == "TEST_FIXTURE"
        assert data["canonical"] is False


class TestProvenanceVerifier:
    def test_fixture_artifact_fails_formal_task(self, tmp_path: Path) -> None:
        """机器人行为任务的产物引用 fixture 路径 → RESOURCE_PROVENANCE_FAILED。"""
        import sqlite3

        from rosclaw.storage.migrations import MigrationRunner
        from rosclaw.task_kernel.service import TaskKernel

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        kernel = TaskKernel(conn, tmp_path)
        bound = kernel.bind_message(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            message_id="m1", text="画五角星", cwd=str(tmp_path),
            mode="SIMULATION", body_id="sim/ur5e",
        )
        task_id = str(bound["task_id"])
        # 产物在 fixture 目录（简化模型渲染的 GIF）。
        fixture_dir = tmp_path / "tests" / "fixtures"
        fixture_dir.mkdir(parents=True)
        gif = fixture_dir / "ur5e_minimal_fixture.gif"
        gif.write_bytes(b"GIF89a" + b"x" * 2048)
        art = kernel.register_artifact(
            task_id=task_id, path=str(gif), media_type="image/gif",
            producer="kernel:sim_pipeline",
        )
        result = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art["artifact_id"]],
        )
        assert result["status"] != "SUCCEEDED"
        assert any(
            "RESOURCE_PROVENANCE" in str(f) for f in result.get("failures", [])
        ), result

    def test_sandbox_returns_resource_identity(self) -> None:
        """Sandbox session 返回 resource ID + model path + digest。"""
        from rosclaw.sandbox.sandbox_api import Sandbox

        sandbox = Sandbox.create("ur5e", world_id="tabletop")
        try:
            assert sandbox.has_physics, sandbox.load_error
            manifest = sandbox.resource_manifest()
            assert manifest["resource_id"] == "robot:ur5e"
            assert Path(manifest["model_path"]).exists()
            assert manifest["model_digest"].startswith("sha256:")
            assert "robot.mjcf.xml" in manifest["model_path"]
        finally:
            sandbox.close()


class TestInspectAsset:
    def test_inspect_asset_resolve_json(self, tmp_path: Path) -> None:
        import contextlib
        import io

        from rosclaw.cognition.inspect_cli import dispatch_inspect_argv

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = dispatch_inspect_argv(["inspect", "asset", "ur5e", "--json"])
        assert code == 0
        data = json.loads(buf.getvalue())
        assert data["resource_id"] == "robot:ur5e"
        assert data["quality"] == "PRODUCTION"
        assert data["canonical"] is True
        assert "mjcf" in data["paths"]
        assert data["digests"]["mjcf"].startswith("sha256:")
