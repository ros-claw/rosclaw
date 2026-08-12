"""UserTurnV2 store（九审 §6.1，NINE-2）。

硬约束：
- 每个用户输入先创建 UserTurn（先落账再路由/执行）；
- 只接受 source=interactive——extension 注入/系统事件不得伪装；
- (session, delivery_seq) 唯一——重复 delivery 只处理一次；
- 恢复历史不重新生成 UserTurn（本 store 只在 live 输入时写入）。
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from rosclaw.contracts.common import new_id


class TurnStore:
    """user_turns 表（migration 023）。"""

    def __init__(self, conn) -> None:
        self._conn = conn

    def record(
        self,
        *,
        pi_session_id: str,
        mission_id: str,
        text: str,
        source: str = "interactive",
    ) -> dict[str, Any]:
        if source != "interactive":
            raise ValueError(
                f"source {source!r} cannot be recorded as a user turn (fail closed)"
            )
        now = datetime.now(UTC).isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(MAX(delivery_seq), 0) + 1 AS next_seq FROM user_turns "
            "WHERE pi_session_id = ?",
            (pi_session_id,),
        ).fetchone()
        seq = int(row["next_seq"])
        text_hash = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
        turn_id = new_id("turn")
        self._conn.execute(
            "INSERT INTO user_turns (turn_id, pi_session_id, mission_id, source, "
            "delivery_seq, text_hash, received_at, persisted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (turn_id, pi_session_id, mission_id, source, seq, text_hash, now, now),
        )
        return {
            "turn_id": turn_id,
            "pi_session_id": pi_session_id,
            "mission_id": mission_id,
            "source": source,
            "delivery_seq": seq,
            "text_hash": text_hash,
            "received_at": now,
        }

    def latest_for_session(self, pi_session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM user_turns WHERE pi_session_id = ? "
            "ORDER BY delivery_seq DESC LIMIT 1",
            (pi_session_id,),
        ).fetchone()
        return dict(row) if row else None
