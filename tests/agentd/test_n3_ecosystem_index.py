"""PR-N3 红测试：Ecosystem Index + inspect self（N 总纲 §PR-N3）。

红测试先行——修复前必须红：
1. `inspect robot ur5e` 一次调用返回权威链（eurdf/mjcf/urdf/assets/
   safety/capabilities 全路径存在 + digest + source 标注）；
2. `inspect self --json` 返回版本/commit/包根/索引健康度；
3. FTS 搜索能找到实体（"五角星"命中 skill/doc/executor 类）；
4. 版本指纹：zoo 文件变更 → health 报 stale → 重建刷新；
5. 诚实降级：索引 DB 损坏 → 查询诚实报错或重建，不静默返回旧数据；
6. 无源码 checkout：packaged zoo 路径也能建索引。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class TestEcosystemIndex:
    def test_inspect_robot_returns_canonical_chain(self, tmp_path: Path) -> None:
        from rosclaw.cognition.index.builder import build_index
        from rosclaw.cognition.index.query import robot_chain

        index_path = build_index(tmp_path / "idx", REPO)
        chain = robot_chain(index_path, "ur5e")
        assert chain is not None
        assert chain["canonical"] is True
        assert chain["source"] == "e-urdf-zoo"
        for key in ("eurdf_path", "mjcf_path", "urdf_path", "safety_path",
                    "capabilities_path"):
            assert key in chain, f"缺 {key}"
            assert Path(chain[key]).exists(), f"{key} 不存在: {chain[key]}"
        assert chain["digest"].startswith("sha256:")
        assert chain["quality"] == "production"
        # 事故防线：fixture/简化模型绝不出现在权威链。
        assert "fixtures" not in json.dumps(chain)
        assert "specs/ur5e" not in json.dumps(chain)

    def test_robot_chain_accepts_body_id_forms(self, tmp_path: Path) -> None:
        """0903 体验实证：模型按 header 的 body_id "sim/ur5e" 查
        robot_chain → 未命中报 UNKNOWN_ROBOT。body_id/资源 id/别名
        族都必须命中同一条权威链。"""
        from rosclaw.cognition.index.builder import build_index
        from rosclaw.cognition.index.query import robot_chain

        index_path = build_index(tmp_path / "idx", REPO)
        for form in ("ur5e", "sim/ur5e", "sim_ur5e", "robot:ur5e"):
            chain = robot_chain(index_path, form)
            assert chain is not None, f"{form} 未命中权威链"
            assert chain["canonical"] is True

    def test_inspect_self_shape(self, tmp_path: Path) -> None:
        from rosclaw.cognition.index.builder import build_index
        from rosclaw.cognition.inspect_cli import inspect_self

        index_path = build_index(tmp_path / "idx", REPO)
        info = inspect_self(tmp_path / "rh", index_path)
        assert info["schema_version"] == "rosclaw.inspect_self.v1"
        assert info["product_root"]
        assert info["package_root"]
        assert info["index"]["entity_count"] > 0
        assert info["index"]["ok"] is True

    def test_fts_search_finds_entities(self, tmp_path: Path) -> None:
        from rosclaw.cognition.index.builder import build_index
        from rosclaw.cognition.index.query import search

        index_path = build_index(tmp_path / "idx", REPO)
        hits = search(index_path, "ur5e")
        assert hits, "搜索 'ur5e' 无结果"
        assert any(h["kind"] in ("asset", "robot") for h in hits)

    def test_fingerprint_stale_detection_and_rebuild(self, tmp_path: Path) -> None:
        from rosclaw.cognition.index.builder import build_index
        from rosclaw.cognition.index.query import health

        index_path = build_index(tmp_path / "idx", REPO)
        h1 = health(index_path)
        assert h1["ok"] and not h1["stale"]
        # 改动 zoo 资产 → stale。
        probe = REPO / "e-urdf-zoo" / "ur5e" / "robot.eurdf.yaml"
        original = probe.read_bytes()
        try:
            probe.write_bytes(original + b"\n# touch\n")
            h2 = health(index_path)
            assert h2["stale"], "资产变更后索引未报 stale"
        finally:
            probe.write_bytes(original)
        # 重建后恢复。
        build_index(tmp_path / "idx", REPO)
        h3 = health(index_path)
        assert h3["ok"] and not h3["stale"]

    def test_broken_index_honest_degrade(self, tmp_path: Path) -> None:
        from rosclaw.cognition.index.builder import build_index
        from rosclaw.cognition.index.query import robot_chain

        index_path = build_index(tmp_path / "idx", REPO)
        # 写坏 DB。
        index_path.write_bytes(b"not a sqlite database")
        # 查询必须诚实：重建或明确报错——绝不静默返回旧数据。
        chain = robot_chain(index_path, "ur5e", product_root=REPO)
        assert chain is not None, "损坏后未恢复"
        assert Path(chain["mjcf_path"]).exists()

    def test_works_without_source_checkout(self, tmp_path: Path) -> None:
        """无源码 checkout：packaged zoo 也能建索引（退出条件）。"""
        import shutil

        staged = tmp_path / "staged"
        shutil.copytree(REPO / "e-urdf-zoo", staged / "e-urdf-zoo")
        from rosclaw.cognition.index.builder import build_index
        from rosclaw.cognition.index.query import robot_chain

        index_path = build_index(tmp_path / "idx", staged)
        chain = robot_chain(index_path, "ur5e")
        assert chain is not None
        assert Path(chain["mjcf_path"]).exists()
