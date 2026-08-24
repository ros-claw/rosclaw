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

# 五审 P0-5B：policy max 不得超过 envelope TTL（30s）——否则 prompt 已
# stale 时动作 lease 仍可用（前四次审计的实际错位）。
LEASE_TTL_SEC = 30.0


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
    # 五审 P0-5A：caller 身份绑定（migration 019）。
    # 六审 §5.3/§5.5（migration 020）：binding_id 是 session binding
    # ID（019 错写 writer lease ID）；writer_lease_id/caller_pid 独立。
    binding_id: str = ""
    caller_uid: int = -1
    writer_lease_id: str = ""
    caller_pid: int = -1


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
        binding_id: str = "",
        caller_uid: int = -1,
        writer_lease_id: str = "",
        caller_pid: int = -1,
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
            binding_id=binding_id,
            caller_uid=caller_uid,
            writer_lease_id=writer_lease_id,
            caller_pid=caller_pid,
        )
        self._conn.execute(
            "INSERT INTO pi_context_leases (context_lease_id, pi_session_id, "
            "mission_id, context_revision, context_hash, body_hash, mode, "
            "issued_at, expires_at, revoked, binding_id, caller_uid, "
            "writer_lease_id, caller_pid) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
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
                lease.binding_id,
                lease.caller_uid,
                lease.writer_lease_id,
                lease.caller_pid,
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
        keys = set(row.keys())
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
            binding_id=row["binding_id"] if "binding_id" in keys else "",
            caller_uid=int(row["caller_uid"]) if "caller_uid" in keys else -1,
            writer_lease_id=row["writer_lease_id"] if "writer_lease_id" in keys else "",
            caller_pid=int(row["caller_pid"]) if "caller_pid" in keys else -1,
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
    """envelope 内容 hash（RFC 8785 canonical——与跨语言 hash 同源）。

    P0-5B：只覆盖稳定的具身事实（body/mode/revision/capabilities/
    task graph）。排除 generated_at/expires_at/hash（时间易变），
    以及 pending_approvals/receipts/active_actions/workers——它们是
    动作的*结果*（建卡/建任务即在 envelope 里新增条目），不是使
    上下文失效的输入变化（P0-C 实证：首个 effectful call 建 task
    → workers 新增 → 同回合 admission 误判 CONTEXT_HASH_MISMATCH）。
    """
    from rosclaw.contracts.pi.canonical import canonical_dumps

    payload = envelope.model_dump(mode="json")
    for volatile in (
        "generated_at",
        "expires_at",
        "hash",
        "freshness",
        "pending_approvals",
        "receipts",
        "active_actions",
        "workers",
    ):
        payload.pop(volatile, None)
    # 八审总纲验证轮实测：turn_in_flight 是回合运行时标志（回合内
    # 外翻转），不是具身事实——留在 hash 里会让同一有效上下文在
    # 回合边界上误判 CONTEXT_HASH_MISMATCH。
    self_state = payload.get("self_state")
    if isinstance(self_state, dict):
        self_state.pop("turn_in_flight", None)
    return hashlib.sha256(canonical_dumps(payload).encode()).hexdigest()[:32]
