"""测试公共助手：Operator Decision Protocol v1 的 identity/proof 构造。

E2 级测试允许在进程内生成真实 Ed25519 keypair 并走**真实**签名/
验证路径（不伪造 proof 字符串）；跨 UID/四进程证明在 T1 docker 测试。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.contracts.operator.decision import (
    DecisionChallengeV1,
    OperatorDecisionProofV1,
)
from rosclaw.operatord.enrollment import OperatorIdentity, enroll

TEST_PRINCIPAL = "user:local:test-operator"


def make_identity(home: Path, *, uid: int | None = None) -> OperatorIdentity:
    home.mkdir(parents=True, exist_ok=True)
    return enroll(home, uid=uid)


def build_proof(
    identity: OperatorIdentity,
    challenge_data: dict,
    decision: str,
    *,
    method: str = "test-tty",
    decided_at: str = "",
) -> dict:
    """用真实 Ed25519 签名构造 OperatorDecisionProofV1。"""
    challenge = DecisionChallengeV1.from_dict(challenge_data)
    proof = OperatorDecisionProofV1(
        enrollment_id=identity.enrollment_id,
        challenge=challenge,
        decision=decision,
        decided_at=decided_at or datetime.now(UTC).isoformat(),
        human_confirmation_method=method,
    )
    proof = replace(proof, signature_b64=identity.sign(proof.signing_payload()))
    return proof.to_dict()


def decide_via_proof(
    client,
    request_id: str,
    identity: OperatorIdentity,
    decision: str = "ACCEPT",
    *,
    principal_id: str = TEST_PRINCIPAL,
    channel: str = "rosclaw_operatord",
    reason: str = "test decision",
    method: str = "test-tty",
) -> dict:
    """challenge.get → sign → decide（与 operatord 同一条协议路径）。"""
    challenge = client.get_operator_challenge(request_id)["challenge"]
    proof = build_proof(identity, challenge, decision, method=method)
    return client.decide_operator_proposal(
        request_id,
        decision=decision,
        principal_id=principal_id,
        channel=channel,
        reason=reason,
        proof=proof,
    )


def register_identity(client, identity: OperatorIdentity) -> dict:
    """以 daemon 管理员身份登记（测试进程通常与 daemon 同 UID）。"""
    return client.register_operator_enrollment(
        identity.enrollment_id,
        public_key_pem=identity.public_key_pem,
        operator_uid=identity.uid,
    )
