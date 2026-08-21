"""Ecosystem Index query（PR-N3）。

- search：FTS5 全文；
- robot_chain：一次调用返回权威资产链（事故防线：fixture/简化模型
  绝不出现）；
- health：索引健康 + stale 检测（指纹比对）；
- 损坏诚实降级：DB 不可读 → 触发重建（调用方给 product_root）或
  明确报错——绝不静默返回旧数据。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rosclaw.cognition.index.builder import _product_fingerprint


def _connect(index_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    return conn


def _entities(conn: sqlite3.Connection, where: str, args: tuple) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM entities WHERE {where}", args  # noqa: S608
        ).fetchall()
    ]


def health(index_path: Path) -> dict:
    """索引健康度 + stale 检测（指纹比对当前产品内容）。"""
    if not index_path.exists():
        return {"ok": False, "stale": False, "entity_count": 0,
                "note": "索引不存在"}
    try:
        conn = _connect(index_path)
        try:
            meta = {
                r["key"]: r["value"]
                for r in conn.execute("SELECT key, value FROM meta").fetchall()
            }
            count = conn.execute("SELECT COUNT(*) AS c FROM entities").fetchone()["c"]
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        return {"ok": False, "stale": False, "entity_count": 0,
                "note": f"索引损坏: {exc}"}
    product_root = Path(meta.get("product_root", ""))
    zoo = product_root / "e-urdf-zoo"
    current = _product_fingerprint(
        product_root, zoo if zoo.is_dir() else None
    )
    return {
        "ok": True,
        "stale": current != meta.get("fingerprint", ""),
        "entity_count": count,
        "index_version": meta.get("index_version", ""),
        "product_root": meta.get("product_root", ""),
    }


def search(index_path: Path, text: str, *, limit: int = 10) -> list[dict]:
    """FTS5 全文搜索。"""
    conn = _connect(index_path)
    try:
        rows = conn.execute(
            "SELECT e.* FROM entities e JOIN entities_fts f "
            "ON f.rowid = e.rowid WHERE entities_fts MATCH ? LIMIT ?",
            (text, limit),
        ).fetchall()
        if not rows:
            # FTS5 默认分词器对 CJK 不做子串匹配——诚实回退 LIKE。
            rows = conn.execute(
                "SELECT * FROM entities WHERE name LIKE ? OR "
                "canonical_id LIKE ? OR payload_json LIKE ? LIMIT ?",
                (f"%{text}%", f"%{text}%", f"%{text}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def robot_chain(index_path: Path, robot_id: str, *, product_root: Path | None = None) -> dict | None:
    """一次调用返回权威资产链。索引损坏 → 重建（product_root 可由
    meta 恢复）后重试一次；仍失败 → None（调用方诚实降级）。"""
    for attempt in range(2):
        try:
            conn = _connect(index_path)
            try:
                robot = _entities(conn, "kind = 'robot' AND canonical_id = ?",
                                  (robot_id,))
                if not robot:
                    return None
                payload = json.loads(robot[0]["payload_json"])
                assets = payload.get("assets", {})
                chain = {
                    "robot_id": robot_id,
                    "canonical": True,
                    "source": robot[0]["source"],
                    "digest": robot[0]["digest"],
                    "quality": robot[0]["quality"],
                    **assets,
                }
                return chain
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            if attempt == 0:
                # 诚实降级：重建而不是返回旧数据。
                from rosclaw.cognition.index.builder import build_index

                root = product_root
                if root is None:
                    try:
                        broken = sqlite3.connect(index_path)
                        row = broken.execute(
                            "SELECT value FROM meta WHERE key = 'product_root'"
                        ).fetchone()
                        root = Path(row[0]) if row else None
                    except sqlite3.DatabaseError:
                        root = None
                if root is None:
                    return None
                build_index(index_path.parent, root)
            else:
                return None
    return None
