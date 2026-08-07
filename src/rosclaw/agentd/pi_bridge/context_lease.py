"""ValidatedContextLeaseV1（四审 HOTFIX-1，P0-4A）：agentd 签发的
具身上下文准入证。

语义：
- `pi.context` 校验成功后由 agentd 签发（同一权威源，不是 TUI 自报）；
- action propose/execute 必须出示 `context_lease_id`——admission 按
  ID 重新读取并检查未过期、未撤销、session/mission/body/mode/
  revision 全匹配；
- context fetch 失败、TTL 到期、session 切换、body 变化 → 立即失效；
- lease 只对 admission 有效，不是执行权；模型永远看不到它。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from rosclaw.contracts.common import new_id

LEASE_TTL_SEC = 120.0


@dataclass(frozen=True)
class ValidatedContextLeaseV1:
    context_lease_id: str
    pi_session_id: str
    mission_id: str
    context_revision: int
    context_hash: str
    body_hash: str
    mode: str
    issued_at: str
    expires_at: str
    revoked: bool = False


class ContextLeaseStore:
    def __init__(self, connection) -> None:
        self._conn = connection

    def issue(
        self,
        *,
        pi_session_id: str,
        mission_id: str,
        context_revision: int,
        context_hash: str,
        body_hash: str,
        mode: str,
        ttl_sec: float = LEASE_TTL_SEC,
    ) -> ValidatedContextLeaseV1:
        """签发新 lease——同 (session, mission) 的旧 lease 立即撤销。"""
        now = datetime.now(UTC)
        self._conn.execute(
            "UPDATE pi_context_leases SET revoked = 1 "
            "WHERE pi_session_id = ? AND mission_id = ? AND revoked = 0",
            (pi_session_id, mission_id),
        )
        lease = ValidatedContextLeaseV1(
            context_lease_id=new_id("ctxl"),
            pi_session_id=pi_session_id,
            mission_id=mission_id,
            context_revision=context_revision,
            context_hash=context_hash,
            body_hash=body_hash,
            mode=mode,
            issued_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=ttl_sec)).isoformat(),
        )
        self._conn.execute(
            "INSERT INTO pi_context_leases (context_lease_id, pi_session_id, "
            "mission_id, context_revision, context_hash, body_hash, mode, "
            "issued_at, expires_at, revoked) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (
                lease.context_lease_id,
                lease.pi_session_id,
                lease.mission_id,
                lease.context_revision,
                lease.context_hash,
                lease.body_hash,
                lease.mode,
                lease.issued_at,
                lease.expires_at,
            ),
        )
        self._conn.commit()
        return lease

    def get(self, context_lease_id: str) -> ValidatedContextLeaseV1 | None:
        row = self._conn.execute(
            "SELECT * FROM pi_context_leases WHERE context_lease_id = ?",
            (context_lease_id,),
        ).fetchone()
        if row is None:
            return None
        return ValidatedContextLeaseV1(
            context_lease_id=row["context_lease_id"],
            pi_session_id=row["pi_session_id"],
            mission_id=row["mission_id"],
            context_revision=row["context_revision"],
            context_hash=row["context_hash"],
            body_hash=row["body_hash"],
            mode=row["mode"],
            issued_at=row["issued_at"],
            expires_at=row["expires_at"],
            revoked=bool(row["revoked"]),
        )

    def is_valid(self, lease: ValidatedContextLeaseV1) -> bool:
        return not lease.revoked and lease.expires_at > datetime.now(UTC).isoformat()

    def revoke(self, context_lease_id: str) -> None:
        self._conn.execute(
            "UPDATE pi_context_leases SET revoked = 1 WHERE context_lease_id = ?",
            (context_lease_id,),
        )
        self._conn.commit()

    def revoke_for_session(self, pi_session_id: str) -> int:
        """session 切换/关闭——该 session 全部 lease 立即失效。"""
        cursor = self._conn.execute(
            "UPDATE pi_context_leases SET revoked = 1 "
            "WHERE pi_session_id = ? AND revoked = 0",
            (pi_session_id,),
        )
        self._conn.commit()
        return cursor.rowcount


def context_hash_of(envelope: Any) -> str:
    """envelope 内容 hash（RFC 8785 canonical——与跨语言 hash 同源）。"""
    from rosclaw.contracts.pi.canonical import canonical_dumps

    return hashlib.sha256(
        canonical_dumps(envelope.model_dump(mode="json")).encode()
    ).hexdigest()[:32]
