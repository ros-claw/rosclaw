"""R1 合约/属性测试：DecisionChallengeV1 / OperatorDecisionProofV1 / DecisionReceiptV1。

签名输入的确定性、全字段绑定（任一字段篡改必败）、协议版本拒绝、
display hash 两侧一致性与变更敏感性。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw.contracts.operator.decision import (
    ACCEPT,
    DECLINE,
    DecisionChallengeV1,
    DecisionReceiptV1,
    OperatorDecisionProofV1,
    canonical_json,
    compute_display_hash,
    generate_ed25519_keypair,
    sign_b64,
    verify_b64,
)


def _challenge() -> DecisionChallengeV1:
    return DecisionChallengeV1(
        proposal_id="proposal_1",
        challenge_nonce="nonce-1",
        display_hash="hash-1",
        execution_mode="SHADOW",
        capability_id="limo.speaker.play_tone",
        canonical_args_hash="args-1",
        issued_at="2026-08-04T00:00:00Z",
        expires_at="2026-08-04T00:01:00Z",
        daemon_instance_id="daemon_1",
        agent_request_id="req-1",
        mission_id="mis-1",
    )


def _receipt() -> DecisionReceiptV1:
    return DecisionReceiptV1(
        proposal_id="proposal_1",
        decision=ACCEPT,
        operator_enrollment_id="oen_1",
        operator_principal="user:local:1000",
        human_confirmation_method="tty-yn",
        challenge_nonce="nonce-1",
        decided_at="2026-08-04T00:00:30Z",
        expires_at="2026-08-04T00:01:00Z",
        daemon_instance_id="daemon_1",
        daemon_key_id="key-1",
        agent_request_id="req-1",
        mission_id="mis-1",
        execution_mode="SHADOW",
        capability_id="limo.speaker.play_tone",
        canonical_args_hash="args-1",
        display_hash="hash-1",
    )


class TestCanonicalJson:
    def test_key_order_independent(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_unicode_stable(self) -> None:
        a = canonical_json({"标题": "播放提示音"})
        b = canonical_json({"标题": "播放提示音"})
        assert a == b


class TestChallenge:
    def test_roundtrip(self) -> None:
        challenge = _challenge()
        assert DecisionChallengeV1.from_dict(challenge.payload()) == challenge

    def test_wrong_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="protocol"):
            DecisionChallengeV1.from_dict({**_challenge().payload(), "protocol_version": "x/9"})


class TestProofSignatureBinding:
    """proof 签名覆盖 challenge 全字段 + decision + 时间 + 方法——任一篡改必败。"""

    @pytest.mark.parametrize(
        "field",
        [
            "proposal_id",
            "challenge_nonce",
            "display_hash",
            "execution_mode",
            "capability_id",
            "canonical_args_hash",
            "expires_at",
            "daemon_instance_id",
            "agent_request_id",
            "mission_id",
        ],
    )
    def test_tampered_challenge_field_fails(self, field: str) -> None:
        private, pem = generate_ed25519_keypair()
        proof = OperatorDecisionProofV1(
            enrollment_id="oen_1",
            challenge=_challenge(),
            decision=ACCEPT,
            decided_at="2026-08-04T00:00:30Z",
            human_confirmation_method="tty-yn",
        )
        sig = sign_b64(private, proof.signing_payload())
        tampered = replace(proof, challenge=replace(proof.challenge, **{field: "TAMPERED"}))
        assert not verify_b64(pem, tampered.signing_payload(), sig)

    @pytest.mark.parametrize("field", ["decision", "decided_at", "human_confirmation_method"])
    def test_tampered_proof_field_fails(self, field: str) -> None:
        private, pem = generate_ed25519_keypair()
        proof = OperatorDecisionProofV1(
            enrollment_id="oen_1",
            challenge=_challenge(),
            decision=ACCEPT,
            decided_at="2026-08-04T00:00:30Z",
            human_confirmation_method="tty-yn",
        )
        sig = sign_b64(private, proof.signing_payload())
        tampered = replace(proof, **{field: "TAMPERED"})
        assert not verify_b64(pem, tampered.signing_payload(), sig)

    def test_wrong_key_fails(self) -> None:
        private, _pem = generate_ed25519_keypair()
        _, other_pem = generate_ed25519_keypair()
        proof = OperatorDecisionProofV1(
            enrollment_id="oen_1",
            challenge=_challenge(),
            decision=ACCEPT,
            decided_at="t",
            human_confirmation_method="m",
        )
        sig = sign_b64(private, proof.signing_payload())
        assert not verify_b64(other_pem, proof.signing_payload(), sig)


class TestReceipt:
    def test_sign_verify_roundtrip(self) -> None:
        private, pem = generate_ed25519_keypair()
        receipt = _receipt().sign(private)
        assert receipt.verify_signature(pem)
        assert DecisionReceiptV1.from_dict(receipt.to_dict()).verify_signature(pem)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("decision", DECLINE),  # decline 伪装 approve 必败
            ("agent_request_id", "req-2"),
            ("mission_id", "mis-2"),
            ("display_hash", "hash-2"),
            ("canonical_args_hash", "args-2"),
            ("challenge_nonce", "nonce-2"),
            ("daemon_key_id", "key-2"),
        ],
    )
    def test_tampered_receipt_field_fails(self, field: str, value: str) -> None:
        private, pem = generate_ed25519_keypair()
        receipt = _receipt().sign(private)
        tampered = replace(receipt, **{field: value})
        assert not tampered.verify_signature(pem)

    def test_receipt_id_stable_per_nonce(self) -> None:
        assert _receipt().receipt_id == _receipt().receipt_id
        assert _receipt().receipt_id != replace(_receipt(), challenge_nonce="n2").receipt_id


class TestDisplayHash:
    def test_same_inputs_same_hash(self) -> None:
        kwargs = {
            "request_id": "r1",
            "title": "t",
            "summary": "s",
            "risk_tier": "LOW",
            "parameters": {"a": 1},
            "body_hash": "b",
            "expires_at": "e",
        }
        assert compute_display_hash(**kwargs) == compute_display_hash(**kwargs)

    @pytest.mark.parametrize("field", ["title", "summary", "risk_tier", "body_hash", "expires_at"])
    def test_any_change_changes_hash(self, field: str) -> None:
        kwargs = {
            "request_id": "r1",
            "title": "t",
            "summary": "s",
            "risk_tier": "LOW",
            "parameters": {"a": 1},
            "body_hash": "b",
            "expires_at": "e",
        }
        changed = {**kwargs, field: "CHANGED"}
        assert compute_display_hash(**kwargs) != compute_display_hash(**changed)
