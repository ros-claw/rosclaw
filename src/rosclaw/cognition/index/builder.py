"""Ecosystem Index builder（PR-N3）。

程序探测，不靠模型猜：
- e-URDF-Zoo：每机器人 → robot 实体 + 资产实体链（eurdf/mjcf/urdf/
  assets/safety/capabilities/benchmark）+ 内容 digest；
- 内置 Skill（包内 skills/）；
- 文档（docs/、prompts）；
- 产品入口（entrypoint/版本）。

指纹 = zoo 全量内容 digest + 包版本——资产变更即 stale。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from rosclaw.cognition.index.schema import INDEX_VERSION, SCHEMA_SQL


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _zoo_root(product_root: Path) -> Path | None:
    """e-URDF-Zoo 根：product_root/e-urdf-zoo（checkout 或打包布局）。"""
    zoo = product_root / "e-urdf-zoo"
    return zoo if zoo.is_dir() else None


def _product_fingerprint(product_root: Path, zoo: Path | None) -> str:
    """产品指纹：zoo 内容摘要 + 包版本——资产变更即 stale。"""
    h = hashlib.sha256()
    version = "unknown"
    try:
        from rosclaw import __version__

        version = str(__version__)
    except Exception:  # noqa: BLE001 - 版本不可得用 unknown
        pass
    h.update(version.encode())
    if zoo is not None:
        for path in sorted(zoo.rglob("*")):
            if path.is_file():
                h.update(str(path.relative_to(zoo)).encode())
                h.update(_sha256(path).encode())
    return "sha256:" + h.hexdigest()


def _add_entity(
    conn: sqlite3.Connection, *, kind: str, canonical_id: str, name: str,
    path: str = "", source: str = "", digest: str = "", quality: str = "",
    payload: dict | None = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO entities (entity_id, kind, canonical_id, "
        "name, path, source, digest, quality, payload_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (f"{kind}:{canonical_id}", kind, canonical_id, name, path, source,
         digest, quality, json.dumps(payload or {}, ensure_ascii=False)),
    )


def build_index(index_dir: Path, product_root: Path) -> Path:
    """构建/重建索引。返回索引 DB 路径。"""
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "ecosystem.db"
    if index_path.exists():
        index_path.unlink()
    conn = sqlite3.connect(index_path)
    try:
        conn.executescript(SCHEMA_SQL)
        zoo = _zoo_root(product_root)
        fingerprint = _product_fingerprint(product_root, zoo)

        # -- e-URDF-Zoo：robot + 资产链 --------------------------------
        if zoo is not None:
            for robot_dir in sorted(p for p in zoo.iterdir() if p.is_dir()):
                robot_id = robot_dir.name
                eurdf = robot_dir / "robot.eurdf.yaml"
                if not eurdf.exists():
                    continue
                assets = {
                    "eurdf_path": eurdf,
                    "mjcf_path": robot_dir / "robot.mjcf.xml",
                    "urdf_path": robot_dir / "robot.urdf",
                    "safety_path": robot_dir / "safety.yaml",
                    "capabilities_path": robot_dir / "capabilities.yaml",
                    "benchmark_path": robot_dir / "benchmark.yaml",
                    "semantic_path": robot_dir / "semantic.yaml",
                }
                assets = {k: v for k, v in assets.items() if v.exists()}
                _add_entity(
                    conn, kind="robot", canonical_id=robot_id,
                    name=f"robot {robot_id}", source="e-urdf-zoo",
                    digest=_sha256(eurdf), quality="production",
                    payload={
                        "text": f"robot {robot_id} e-urdf canonical profile",
                        "assets": {k: str(v) for k, v in assets.items()},
                    },
                )
                for key, path in assets.items():
                    _add_entity(
                        conn, kind="asset",
                        canonical_id=f"{robot_id}:{key}",
                        name=f"{robot_id} {key}", path=str(path),
                        source="e-urdf-zoo", digest=_sha256(path),
                        quality="production",
                        payload={
                            "text": f"{robot_id} {key} asset",
                            "robot_id": robot_id, "asset_kind": key,
                        },
                    )

        # -- 内置 Skill -------------------------------------------------
        skills_dir = product_root / "packages" / "rosclaw-agent" / "skills"
        if skills_dir.is_dir():
            for skill_dir in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
                skill_file = skill_dir / "SKILL.md"
                if not skill_file.exists():
                    continue
                text = skill_file.read_text(encoding="utf-8")
                _add_entity(
                    conn, kind="skill", canonical_id=skill_dir.name,
                    name=f"skill {skill_dir.name}", path=str(skill_file),
                    source="rosclaw-bundled", digest=_sha256(skill_file),
                    quality="production",
                    payload={"text": f"skill {skill_dir.name} {text[:2000]}"},
                )

        # -- 文档 -------------------------------------------------------
        for docs_dir_name in ("docs",):
            docs_dir = product_root / docs_dir_name
            if not docs_dir.is_dir():
                continue
            for md in sorted(docs_dir.rglob("*.md"))[:200]:
                text = md.read_text(encoding="utf-8", errors="replace")
                _add_entity(
                    conn, kind="doc",
                    canonical_id=str(md.relative_to(product_root)),
                    name=md.stem, path=str(md), source="product-docs",
                    digest=_sha256(md), quality="production",
                    payload={"text": f"{md.stem} {text[:2000]}"},
                )

        # -- 产品入口 ----------------------------------------------------
        _add_entity(
            conn, kind="cli", canonical_id="rosclaw",
            name="rosclaw CLI", source="product",
            quality="production",
            payload={"text": "rosclaw chat setup status doctor inspect"},
        )

        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('index_version', ?)",
            (INDEX_VERSION,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('fingerprint', ?)",
            (fingerprint,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('product_root', ?)",
            (str(product_root),),
        )
        conn.commit()
    finally:
        conn.close()
    return index_path
