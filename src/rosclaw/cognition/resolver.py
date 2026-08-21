"""Resource Resolver（PR-N4，N 总纲 §8）——通用资源解析。

权威性排序（§4.2）：当前运行时注册表/索引 → 当前安装版本正式资产 →
workspace → Hub → 上游文档 → 网络 → Agent 推断。test fixture 永不进入
production 解析结果。

当前来源：Ecosystem Index（N3——e-URDF-Zoo robot 链 + 内置 skill +
文档）；其余 kind 无来源时诚实返回 None（来源随批次注册）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rosclaw.cognition.index.builder import build_index
from rosclaw.cognition.index.query import robot_chain

#: 测试夹具目录标记（production 解析永不命中）。
FIXTURE_MARKER = "tests/fixtures"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_manifest(product_root: Path, name: str) -> dict | None:
    """tests/fixtures/ 下的显式夹具 manifest（*.yaml 标记）。"""
    manifest_path = product_root / "tests" / "fixtures" / f"{name}.yaml"
    if not manifest_path.exists():
        return None
    import yaml

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    paths = data.get("paths", {})
    digests = {}
    for key, rel in paths.items():
        full = product_root / "tests" / "fixtures" / rel
        if full.exists():
            paths[key] = str(full)
            digests[key] = _sha256(full)
    return {
        "schema_version": "rosclaw.resource_manifest.v1",
        "resource_id": data.get("resource_id", f"robot:{name}"),
        "kind": data.get("kind", "robot"),
        "version": str(data.get("version", "1.0")),
        "source": data.get("source", "tests/fixtures"),
        "trust": data.get("trust", "TEST_ONLY"),
        "quality": data.get("quality", "TEST_FIXTURE"),
        "canonical": bool(data.get("canonical", False)),
        "compatibility": data.get("compatibility", {}),
        "paths": paths,
        "digests": digests,
    }


def resolve_resource(
    kind: str,
    resource_id: str,
    *,
    product_root: Path,
    production_only: bool = True,
) -> dict | None:
    """解析资源为 ResourceManifestV1 形状（dict）。

    production_only（默认开）：fixture 永不返回——这是事故防线。
    返回 None = 无权威资源（调用方诚实报错，不得降级到猜测）。
    """
    if kind == "robot":
        # fixture 先判：名字即夹具 → production 直接拒绝。
        fixture = _fixture_manifest(product_root, resource_id)
        if fixture is not None:
            return None if production_only else fixture
        # 生产路径：Ecosystem Index 权威链。
        index_dir = product_root / ".rosclaw-index"
        index_path = index_dir / "ecosystem.db"
        if not index_path.exists():
            build_index(index_dir, product_root)
        chain = robot_chain(index_path, resource_id, product_root=product_root)
        if chain is None:
            return None
        paths = {
            "profile": chain.get("eurdf_path", ""),
            "mjcf": chain.get("mjcf_path", ""),
            "urdf": chain.get("urdf_path", ""),
            "safety": chain.get("safety_path", ""),
            "capabilities": chain.get("capabilities_path", ""),
        }
        paths = {k: v for k, v in paths.items() if v}
        assets_dir = Path(chain.get("mjcf_path", "")).parent / "assets"
        if assets_dir.is_dir():
            paths["meshes"] = str(assets_dir)
        digests = {
            key: _sha256(Path(p)) for key, p in paths.items()
            if Path(p).is_file()
        }
        return {
            "schema_version": "rosclaw.resource_manifest.v1",
            "resource_id": f"robot:{resource_id}",
            "kind": "robot",
            "version": "1.0",
            "source": chain.get("source", "e-urdf-zoo"),
            "trust": "ROSCLAW_OFFICIAL",
            "quality": "PRODUCTION",
            "canonical": True,
            "compatibility": {"simulators": ["mujoco"]},
            "paths": paths,
            "digests": digests,
        }
    # 其余 kind：来源未注册——诚实 None（不猜不编）。
    return None
