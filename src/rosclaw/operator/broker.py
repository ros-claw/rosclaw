"""Operator Broker (ADR-0006, PR-OP-060/061).

Owns approval cards and MissionGrants. Invariants enforced here:

- the broker never hands private permit material to the agent — the grant
  public object carries no signature field at all; the private HMAC lives
  in ``mission_grants.private_signature`` and never leaves this module;
- EXACT_ACTION grants are single-use (consumed on first successful verify);
- verify() is fail closed: unknown grant, revoked, expired, wrong
  principal, body hash drift, mode mismatch, risk above ceiling → deny
  with a reason code;
- every decision is journaled to worker-style operator events (reuse of
  the operator_requests table + grant rows is the audit trail).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from datetime import UTC, datetime

from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.contracts.operator.approval import (
    ApprovalRequestV2,
    ApprovalStatus,
)
from rosclaw.contracts.operator.grant import (
    GrantBudgets,
    GrantScope,
    MissionGrantV1,
)

_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class GrantDeniedError(ValidationError):
    """Grant verification failed closed."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code


class OperatorBroker:
    def __init__(self, conn: sqlite3.Connection, *, policy_hash: str) -> None:
        self._conn = conn
        self._policy_hash = policy_hash
        # Broker-local signing key: random, persisted in broker_state. It is
        # NOT derivable from the public policy hash — a public input can
        # never be used to recompute grant signatures (audit fix; local
        # integrity control, not a TPM/remote witness).
        row = self._conn.execute(
            "SELECT value FROM broker_state WHERE key = 'signing_key'"
        ).fetchone()
        if row is None:
            import os

            key = os.urandom(32)
            self._conn.execute(
                "INSERT INTO broker_state (key, value) VALUES ('signing_key', ?)",
                (key,),
            )
            self._secret = key
        else:
            self._secret = bytes(row["value"])

    # ------------------------------------------------------------------
    # approval requests
    # ------------------------------------------------------------------
    def create_request(self, request: ApprovalRequestV2) -> ApprovalRequestV2:
        self._conn.execute(
            "INSERT INTO operator_requests (request_id, mission_id, task_id, "
            "request_json, status, created_at) VALUES (?, ?, ?, ?, 'PENDING', ?)",
            (
                request.request_id,
                request.mission_id,
                request.task_id,
                request.model_dump_json(),
                request.created_at,
            ),
        )
        self._event(
            "rosclaw.operator.approval.requested.v1",
            request.principal,
            {
                "request_id": request.request_id,
                "mission_id": request.mission_id,
                "risk_tier": request.action_display.risk_tier,
            },
        )
        return request

    def get_request(self, request_id: str) -> ApprovalRequestV2 | None:
        row = self._conn.execute(
            "SELECT request_json, status FROM operator_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        request = ApprovalRequestV2(**json.loads(row["request_json"]))
        return request.model_copy(update={"status": ApprovalStatus(row["status"])})

    def pending_requests(self, mission_id: str | None = None) -> list[ApprovalRequestV2]:
        if mission_id:
            rows = self._conn.execute(
                "SELECT request_json FROM operator_requests WHERE status = 'PENDING' "
                "AND mission_id = ?",
                (mission_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT request_json FROM operator_requests WHERE status = 'PENDING'"
            ).fetchall()
        return [ApprovalRequestV2(**json.loads(r["request_json"])) for r in rows]

    def decide(
        self,
        request_id: str,
        *,
        principal: str,
        approve: bool,
        decided_by: str | None = None,
    ) -> MissionGrantV1 | None:
        """Human decision. Approval mints a grant; denial is terminal.

        七审 §2.5：decided_by 可与 grant principal 分离——POLICY_AUTO
        的 grant 仍发给 mission owner（principal），decided_by 记录
        政策权威（审计链）。"""
        row = self._conn.execute(
            "SELECT request_json, status FROM operator_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise ValidationError(f"unknown approval request {request_id!r}")
        if row["status"] != "PENDING":
            raise ValidationError(f"request {request_id!r} already {row['status']}")
        request = ApprovalRequestV2(**json.loads(row["request_json"]))
        if request.expires_at < _utcnow():
            self._conn.execute(
                "UPDATE operator_requests SET status = 'EXPIRED' WHERE request_id = ?",
                (request_id,),
            )
            raise GrantDeniedError("request_expired", "approval request expired")
        status = "APPROVED" if approve else "DENIED"
        self._conn.execute(
            "UPDATE operator_requests SET status = ?, decided_by = ?, decided_at = ? "
            "WHERE request_id = ?",
            (status, decided_by or principal, _utcnow(), request_id),
        )
        self._event(
            f"rosclaw.operator.approval.{status.lower()}.v1",
            principal,
            {"request_id": request_id, "approved": approve},
        )
        if not approve:
            return None
        return self._mint_grant(request, principal)

    def cancel_request(self, request_id: str, *, principal: str) -> None:
        """WP-P0-7（总纲 §7.4）：取消传播——撤销未消费的审批请求。
        仅 PENDING 可撤销；取消是终态（不产 grant、不可再决定）。"""
        row = self._conn.execute(
            "SELECT status FROM operator_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise ValidationError(f"unknown approval request {request_id!r}")
        if row["status"] != "PENDING":
            raise ValidationError(f"request {request_id!r} already {row['status']}")
        self._conn.execute(
            "UPDATE operator_requests SET status = 'CANCELLED', decided_by = ?, "
            "decided_at = ? WHERE request_id = ?",
            (f"cancel:{principal}", _utcnow(), request_id),
        )
        self._event(
            "rosclaw.operator.approval.cancelled.v1",
            principal,
            {"request_id": request_id},
        )

    # ------------------------------------------------------------------
    # grants
    # ------------------------------------------------------------------
    def _mint_grant(self, request: ApprovalRequestV2, principal: str) -> MissionGrantV1:
        grant = MissionGrantV1(
            grant_id=new_id("grant"),
            principal=principal,
            body_id=request.body_id,
            effective_body_hash=request.effective_body_hash,
            mode=request.mode,
            scope=GrantScope(
                tier=request.requested_tier,
                exact_action_intent=hashlib.sha256(
                    request.action_display.model_dump_json().encode()
                ).hexdigest()
                if request.requested_tier == "EXACT_ACTION"
                else None,
            ),
            risk_ceiling=request.action_display.risk_tier,
            budgets=GrantBudgets(max_actions=1 if request.requested_tier == "EXACT_ACTION" else 10),
            policy_hash=self._policy_hash,
            issued_at=_utcnow(),
            expires_at=request.expires_at,
        )
        grant.finalize_hash()
        signature = hmac.new(self._secret, grant.public_hash.encode(), hashlib.sha256).hexdigest()
        self._conn.execute(
            "INSERT INTO mission_grants (grant_id, request_id, public_json, "
            "private_signature, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                grant.grant_id,
                request.request_id,
                grant.model_dump_json(),
                signature,
                _utcnow(),
                grant.expires_at,
            ),
        )
        return grant

    def get_grant_public(self, grant_id: str) -> MissionGrantV1 | None:
        row = self._conn.execute(
            "SELECT public_json FROM mission_grants WHERE grant_id = ?", (grant_id,)
        ).fetchone()
        return MissionGrantV1(**json.loads(row["public_json"])) if row else None

    def revoke(self, grant_id: str, *, principal: str) -> None:
        updated = self._conn.execute(
            "UPDATE mission_grants SET revoked = 1 WHERE grant_id = ?", (grant_id,)
        ).rowcount
        if not updated:
            raise ValidationError(f"unknown grant {grant_id!r}")
        self._event("rosclaw.operator.grant.revoked.v1", principal, {"grant_id": grant_id})

    def verify(
        self,
        grant_id: str,
        *,
        principal: str,
        body_hash: str,
        mode: str,
        risk_tier: str,
        action_intent: str | None = None,
        consume: bool = True,
    ) -> MissionGrantV1:
        """Fail-closed grant check (the agent calls this before dispatch;
        rosclawd will repeat it independently with the private signature)."""
        row = self._conn.execute(
            "SELECT public_json, private_signature, consumed, revoked, expires_at "
            "FROM mission_grants WHERE grant_id = ?",
            (grant_id,),
        ).fetchone()
        if row is None:
            raise GrantDeniedError("unknown_grant", grant_id)
        grant = MissionGrantV1(**json.loads(row["public_json"]))
        expected_sig = hmac.new(
            self._secret, grant.public_hash.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(row["private_signature"], expected_sig):
            raise GrantDeniedError("forged_grant", "signature mismatch")
        if row["revoked"]:
            raise GrantDeniedError("grant_revoked", grant_id)
        if row["expires_at"] < _utcnow():
            raise GrantDeniedError("grant_expired", row["expires_at"])
        if grant.principal != principal:
            raise GrantDeniedError("principal_mismatch", f"{grant.principal} != {principal}")
        if grant.effective_body_hash != body_hash:
            raise GrantDeniedError(
                "body_hash_changed",
                f"grant bound to {grant.effective_body_hash[:18]}…, current {body_hash[:18]}…",
            )
        if grant.mode != mode:
            raise GrantDeniedError("mode_mismatch", f"grant mode {grant.mode} != {mode}")
        if _RISK_ORDER[risk_tier] > _RISK_ORDER[grant.risk_ceiling]:
            raise GrantDeniedError("risk_above_ceiling", f"{risk_tier} > {grant.risk_ceiling}")
        if grant.scope.tier == "EXACT_ACTION":
            if row["consumed"]:
                raise GrantDeniedError("grant_consumed", "EXACT_ACTION grants are single-use")
            if grant.scope.exact_action_intent is not None:
                # 精确动作绑定（§11.2）：调用方必须声明动作意图，省略即拒绝。
                if action_intent is None:
                    raise GrantDeniedError(
                        "missing_action_intent",
                        "EXACT_ACTION verify requires the declared action intent",
                    )
                if action_intent != grant.scope.exact_action_intent:
                    raise GrantDeniedError(
                        "action_intent_mismatch", "action differs from the approved card"
                    )
        if consume and grant.scope.tier == "EXACT_ACTION":
            self._conn.execute(
                "UPDATE mission_grants SET consumed = 1 WHERE grant_id = ?", (grant_id,)
            )
        return grant

    # ------------------------------------------------------------------
    def action_intent_for_grant(self, grant_id: str) -> str | None:
        """Recompute the exact-action intent bound at mint time, from the
        approved card. The agent references grant_id; the broker — not the
        model — determines which action was actually approved."""
        row = self._conn.execute(
            "SELECT request_id FROM mission_grants WHERE grant_id = ?", (grant_id,)
        ).fetchone()
        if row is None:
            return None
        request = self.get_request(row["request_id"])
        if request is None:
            return None
        return hashlib.sha256(request.action_display.model_dump_json().encode()).hexdigest()

    # ------------------------------------------------------------------
    def _event(self, event_type: str, actor_id: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT INTO operator_events (event_id, event_type, actor_id, "
            "payload_json, occurred_at) VALUES (?, ?, ?, ?, ?)",
            (
                new_id("oevt"),
                event_type,
                actor_id,
                json.dumps(payload, sort_keys=True, ensure_ascii=False),
                _utcnow(),
            ),
        )
