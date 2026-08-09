"""ActionTxn（四审 HOTFIX-2，P0-4C）：动作事务状态机。

request → session → mission → context → approval → grant → action →
receipt 全 ID 链的单一持久化承载。

状态机：

```text
PROPOSED → AWAITING_OPERATOR → APPROVED → DISPATCHING
→ RECEIPT_PENDING → COMPLETED | FAILED | CANCELLED | EXPIRED
                              ↘ DECLINED（operator 拒绝的终态）
```

幂等语义：
- 同 idempotency_key + 同 request_hash → 返回同一事务（不重复建卡）；
- 同 key + 不同 hash → IDEMPOTENCY_CONFLICT；
- 所有状态转移在数据库事务中完成。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from rosclaw.contracts.common import new_id
from rosclaw.contracts.pi.canonical import canonical_dumps

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "EXPIRED", "DECLINED"}
_VALID_TRANSITIONS = {
    "PROPOSED": {"AWAITING_OPERATOR", "CANCELLED", "EXPIRED"},
    "AWAITING_OPERATOR": {"APPROVED", "DECLINED", "EXPIRED", "CANCELLED"},
    "APPROVED": {"DISPATCHING", "EXPIRED", "CANCELLED"},
    "DISPATCHING": {"RECEIPT_PENDING", "FAILED"},
    "RECEIPT_PENDING": {"COMPLETED", "FAILED"},
}


@dataclass(frozen=True)
class ActionTxn:
    txn_id: str
    idempotency_key: str
    request_hash: str
    pi_session_id: str
    mission_id: str
    context_lease_id: str
    context_revision: int
    body_hash: str
    mode: str
    capability_id: str
    arguments_hash: str
    risk_tier: str
    approval_id: str
    display_hash: str
    grant_id: str
    action_id: str
    receipt_id: str
    state: str
    created_at: str
    expires_at: str
    completed_at: str


def request_hash_of(
    *,
    capability_id: str,
    arguments: dict[str, Any],
    mission_id: str,
    mode: str,
    context_revision: int,
    body_hash: str,
) -> str:
    """请求内容 hash（canonical——同 key 不同内容必须 CONFLICT）。"""
    return hashlib.sha256(
        canonical_dumps(
            {
                "arguments": arguments,
                "body_hash": body_hash,
                "capability_id": capability_id,
                "context_revision": context_revision,
                "mission_id": mission_id,
                "mode": mode,
            }
        ).encode()
    ).hexdigest()


class IdempotencyConflictError(Exception):
    def __init__(self, existing_txn_id: str) -> None:
        super().__init__(
            f"idempotency_key reused with a different request hash "
            f"(existing txn {existing_txn_id})"
        )
        self.code = "IDEMPOTENCY_CONFLICT"
        self.existing_txn_id = existing_txn_id


class ActionTxnStore:
    def __init__(self, connection) -> None:
        self._conn = connection

    def _row_to_txn(self, row) -> ActionTxn:
        return ActionTxn(
            txn_id=row["txn_id"],
            idempotency_key=row["idempotency_key"],
            request_hash=row["request_hash"],
            pi_session_id=row["pi_session_id"],
            mission_id=row["mission_id"],
            context_lease_id=row["context_lease_id"],
            context_revision=row["context_revision"],
            body_hash=row["body_hash"],
            mode=row["mode"],
            capability_id=row["capability_id"],
            arguments_hash=row["arguments_hash"],
            risk_tier=row["risk_tier"],
            approval_id=row["approval_id"],
            display_hash=row["display_hash"],
            grant_id=row["grant_id"],
            action_id=row["action_id"],
            receipt_id=row["receipt_id"],
            state=row["state"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            completed_at=row["completed_at"],
        )

    def get(self, txn_id: str) -> ActionTxn | None:
        row = self._conn.execute(
            "SELECT * FROM action_txns WHERE txn_id = ?", (txn_id,)
        ).fetchone()
        return self._row_to_txn(row) if row else None

    def get_by_idempotency_key(self, key: str) -> ActionTxn | None:
        row = self._conn.execute(
            "SELECT * FROM action_txns WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return self._row_to_txn(row) if row else None

    def get_by_approval(self, approval_id: str) -> ActionTxn | None:
        row = self._conn.execute(
            "SELECT * FROM action_txns WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        return self._row_to_txn(row) if row else None

    def create(
        self,
        *,
        idempotency_key: str,
        request_hash: str,
        pi_session_id: str,
        mission_id: str,
        context_lease_id: str,
        context_revision: int,
        body_hash: str,
        mode: str,
        capability_id: str,
        arguments_hash: str,
        risk_tier: str,
        ttl_sec: float = 600.0,
        expires_at: str = "",
    ) -> ActionTxn:
        """创建事务（PROPOSED）。同 key 已有记录：同 hash 返回既有
        事务，不同 hash 抛 IdempotencyConflictError。"""
        existing = self.get_by_idempotency_key(idempotency_key)
        if existing is not None:
            if existing.request_hash == request_hash:
                return existing
            raise IdempotencyConflictError(existing.txn_id)
        now = datetime.now(UTC)
        txn_id = new_id("atxn")
        self._conn.execute(
            "INSERT INTO action_txns (txn_id, idempotency_key, request_hash, "
            "pi_session_id, mission_id, context_lease_id, context_revision, "
            "body_hash, mode, capability_id, arguments_hash, risk_tier, "
            "approval_id, display_hash, grant_id, action_id, receipt_id, "
            "state, created_at, expires_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', '', '', '', "
            "'PROPOSED', ?, ?, '')",
            (
                txn_id,
                idempotency_key,
                request_hash,
                pi_session_id,
                mission_id,
                context_lease_id,
                context_revision,
                body_hash,
                mode,
                capability_id,
                arguments_hash,
                risk_tier,
                now.isoformat(),
                expires_at or (now + timedelta(seconds=ttl_sec)).isoformat(),
            ),
        )
        self._conn.commit()
        return self.get(txn_id)  # type: ignore[return-value]

    def transition(self, txn_id: str, to_state: str, **fields: str) -> ActionTxn:
        """状态转移（校验合法边；终态不可逆）。fields 可更新
        approval_id/display_hash/grant_id/action_id/receipt_id。"""
        txn = self.get(txn_id)
        if txn is None:
            raise KeyError(f"unknown txn {txn_id}")
        if txn.state in TERMINAL_STATES:
            if to_state != txn.state:
                raise ValueError(
                    f"txn {txn_id} is terminal ({txn.state}); cannot move to {to_state}"
                )
        elif to_state not in _VALID_TRANSITIONS.get(txn.state, set()):
            raise ValueError(f"illegal transition {txn.state} → {to_state}")
        allowed = {"approval_id", "display_hash", "grant_id", "action_id", "receipt_id"}
        updates = {k: v for k, v in fields.items() if k in allowed and v}
        completed = (
            datetime.now(UTC).isoformat() if to_state in TERMINAL_STATES else txn.completed_at
        )
        assignments = ", ".join([f"{k} = ?" for k in updates] + ["state = ?", "completed_at = ?"])
        self._conn.execute(
            f"UPDATE action_txns SET {assignments} WHERE txn_id = ?",  # noqa: S608 - keys whitelisted
            (*updates.values(), to_state, completed, txn_id),
        )
        self._conn.commit()
        return self.get(txn_id)  # type: ignore[return-value]
