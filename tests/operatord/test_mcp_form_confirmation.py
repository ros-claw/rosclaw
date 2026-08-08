from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from rosclaw.contracts.operator.decision import (
    DecisionChallengeV1,
    generate_ed25519_keypair,
)
from rosclaw.operatord.server import OperatorDaemon


class _Identity:
    def __init__(self) -> None:
        self.private, self.public_key_pem = generate_ed25519_keypair()
        self.enrollment_id = "operator-mcp-form"

    def sign(self, payload: bytes) -> str:
        import base64

        return base64.b64encode(self.private.sign(payload)).decode()


class _Daemon:
    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []

    def get_operator_proposal(self, request_id: str) -> dict[str, Any]:
        return {
            "proposal": {
                "request_id": request_id,
                "action_intent_hash": "sha256:exact",
                "execution_mode": "REAL",
            }
        }

    def get_operator_challenge(self, request_id: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        challenge = DecisionChallengeV1(
            proposal_id=request_id,
            challenge_nonce="nonce-mcp-form",
            display_hash="display-hash",
            execution_mode="REAL",
            capability_id="limo.navigate_to_pose",
            canonical_args_hash="sha256:exact",
            issued_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=60)).isoformat(),
            daemon_instance_id="daemon-test",
        )
        return {"challenge": challenge.payload()}

    def decide_operator_proposal(self, request_id: str, **kwargs: Any) -> dict[str, Any]:
        self.decisions.append({"request_id": request_id, **kwargs})
        return {
            "decision_receipt": {
                "proposal_id": request_id,
                "signature_b64": "daemon-signature",
            },
            "action": {"action_id": "action-mcp-form", "state": "QUEUED"},
        }


async def test_mcp_form_confirmation_is_signed_by_operatord(tmp_path) -> None:
    daemon = _Daemon()
    broker = OperatorDaemon(
        identity=_Identity(),
        socket_path=tmp_path / "operatord.sock",
        daemon_client=daemon,
    )

    result = await broker.handle(
        "uid:1000",
        "approvals.confirm_mcp_form",
        {
            "request_id": "proposal-mcp-form",
            "action_intent_hash": "sha256:exact",
            "approve": True,
            "requested_principal": "mcp:test-client",
            "reason": "accepted exact MCP form",
        },
    )

    assert result["ok"] is True
    assert result["approved"] is True
    decision = daemon.decisions[0]
    assert decision["decision"] == "ACCEPT"
    assert decision["channel"] == "mcp_form_via_rosclaw_operatord"
    assert decision["principal_id"] == "mcp:test-client"
    assert decision["proof"]["human_confirmation_method"] == "mcp-form-elicitation"


async def test_mcp_form_confirmation_rejects_intent_mismatch(tmp_path) -> None:
    daemon = _Daemon()
    broker = OperatorDaemon(
        identity=_Identity(),
        socket_path=tmp_path / "operatord.sock",
        daemon_client=daemon,
    )

    result = await broker.handle(
        "uid:1000",
        "approvals.confirm_mcp_form",
        {
            "request_id": "proposal-mcp-form",
            "action_intent_hash": "sha256:changed",
            "approve": True,
        },
    )

    assert result == {"ok": False, "error": "action_intent_hash_mismatch"}
    assert daemon.decisions == []
