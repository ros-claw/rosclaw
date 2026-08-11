"""SessionCatalogV1（总纲 WP-P0-2）：产品级会话索引。

Pi JSONL 是对话原始记录；pi_session_bindings 是安全关系。Catalog
是产品检索投影——标题/摘要/Robot/Mode/Task 状态/成本/归档/搜索，
三件事职责不混。

更新来源（总纲 §5.2）：refresh() 一次性 backfill（JSONL 扫描 +
binding 投影），之后增量 upsert（session/task/usage 事件由调用方
驱动）。标题：确定性规则（首条目标 ≤30 字）；用户 rename 永远
优先（title_source=user 不被自动覆盖）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rosclaw.agentd.session_list import list_sessions

_MAX_TITLE = 30


def _auto_title(first_message: str) -> str:
    """确定性标题：首条目标截断 ≤30 字，不调模型。"""
    text = " ".join(first_message.split())
    if len(text) > _MAX_TITLE:
        return text[: _MAX_TITLE - 1] + "…"
    return text or "（未命名会话）"


class SessionCatalog:
    """agent_sessions 表（migration 022）的产品读写面。"""

    def __init__(self, conn) -> None:
        self._conn = conn

    # -- 写入 ---------------------------------------------------------------
    def upsert(self, **fields: Any) -> None:
        """按 session_id 幂等 upsert；用户标题不被自动字段覆盖。"""
        session_id = fields["session_id"]
        existing = self._conn.execute(
            "SELECT title_source, display_name FROM agent_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if existing is not None and existing["title_source"] == "user":
            # 用户标题永远优先。
            fields.pop("display_name", None)
            fields.pop("title_source", None)
        if existing is None:
            columns = ", ".join(fields)
            placeholders = ", ".join("?" for _ in fields)
            self._conn.execute(
                f"INSERT INTO agent_sessions ({columns}) VALUES ({placeholders})",  # noqa: S608
                tuple(fields.values()),
            )
        else:
            fields.pop("session_id", None)
            if not fields:
                return
            assignments = ", ".join(f"{k} = ?" for k in fields)
            self._conn.execute(
                f"UPDATE agent_sessions SET {assignments}, "  # noqa: S608
                "revision = revision + 1 WHERE session_id = ?",
                (*fields.values(), session_id),
            )

    def rename(self, session_id: str, title: str) -> None:
        self._conn.execute(
            "UPDATE agent_sessions SET display_name = ?, title_source = 'user', "
            "revision = revision + 1 WHERE session_id = ?",
            (title, session_id),
        )

    def archive(self, session_id: str) -> None:
        from datetime import UTC, datetime

        self._conn.execute(
            "UPDATE agent_sessions SET archived_at = ?, revision = revision + 1 "
            "WHERE session_id = ?",
            (datetime.now(UTC).isoformat(), session_id),
        )

    # -- 读取 ---------------------------------------------------------------
    def list(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM agent_sessions"
        if not include_archived:
            sql += " WHERE archived_at IS NULL"
        sql += " ORDER BY last_active_at DESC"
        return [dict(r) for r in self._conn.execute(sql).fetchall()]

    def search(self, query: str) -> list[dict[str, Any]]:
        like = f"%{query}%"
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM agent_sessions WHERE archived_at IS NULL AND ("
                "display_name LIKE ? OR search_text LIKE ? OR session_id LIKE ?"
                ") ORDER BY last_active_at DESC LIMIT 50",
                (like, like, like),
            ).fetchall()
        ]

    # -- backfill -----------------------------------------------------------
    def refresh(self, home: Path) -> int:
        """JSONL 扫描 + binding 投影的增量回填。返回新增/更新行数。"""
        bindings = {
            r["pi_session_id"]: r
            for r in self._conn.execute(
                "SELECT pi_session_id, mission_id, body_id, execution_mode, status "
                "FROM pi_session_bindings"
            ).fetchall()
        }
        missions = {
            r[0]: (r[1], r[2])
            for r in self._conn.execute(
                "SELECT mission_id, effective_body_hash, mode FROM missions"
            ).fetchall()
        }
        count = 0
        for info in list_sessions(home):
            session_id = info["session_id"]
            binding = bindings.get(session_id)
            mission_id = binding["mission_id"] if binding else None
            body_id = binding["body_id"] if binding else ""
            mode = binding["execution_mode"] if binding else ""
            if mission_id and mission_id in missions:
                mode = mode or missions[mission_id][1]
            title = info.get("display_name") or _auto_title(
                info.get("first_message", "")
            )
            lifecycle = "ACTIVE" if binding and binding["status"] == "ACTIVE" else "IDLE"
            search_text = " ".join(
                [
                    title,
                    info.get("first_message", ""),
                    body_id,
                    session_id,
                ]
            )[:2000]
            self.upsert(
                session_id=session_id,
                pi_session_path=info["path"],
                display_name=title,
                title_source="auto" if not info.get("display_name") else "user",
                mission_id=mission_id,
                body_id=body_id,
                execution_mode=mode,
                lifecycle_state=lifecycle,
                search_text=search_text,
                message_count=info.get("message_count", 0),
                created_at=info.get("created") or "",
                last_active_at=info.get("modified") or "",
            )
            count += 1
        return count
