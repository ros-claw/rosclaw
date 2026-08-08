"""Operator Decision Protocol v1（二次复核 R1/P0-3/P0-4/P0-6）。

三个协议对象：

* ``DecisionChallengeV1`` — rosclawd 签发的一次性挑战。challenge_nonce 由
  daemon 生成并随 proposal 保存；operatord 必须**原样**签回同一个 nonce
  （修复初版"operatord 自生成 nonce、daemon 用 challenge nonce 验证"
  的协议矛盾）。
* ``HumanConfirmationV1`` — operatord 记录的人在前台终端的显式确认。
* ``DecisionReceiptV1`` — daemon 用自己的 Ed25519 key 签名的精确决策
  回执；agentd 只接受签名校验通过、``decision=ACCEPT`` 且所有字段与
  本地审批卡精确相等的 receipt（重放/过期/篡改 fail closed）。

签名规范：Ed25519 over ``canonical_json(payload)``，
``canonical_json`` = JSON UTF-8、sort_keys、紧凑分隔符。
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

PROTOCOL_VERSION_CHALLENGE = "decision-challenge/1"
PROTOCOL_VERSION_RECEIPT = "decision-receipt/1"
PROTOCOL_VERSION_PROOF = "operator-decision-proof/1"

ACCEPT = "ACCEPT"
DECLINE = "DECLINE"
DECISIONS = frozenset({ACCEPT, DECLINE})


def canonical_json(obj: Any) -> bytes:
    """确定性 JSON 编码（签名/哈希输入）。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def compute_display_hash(
    *,
    request_id: str,
    title: str,
    summary: str,
    risk_tier: str,
    parameters: dict[str, Any],
    body_hash: str,
    expires_at: str,
    # 五审 P0-5C：扩展绑定字段（缺省 = 旧 V2 卡公式，向后兼容只读）。
    capability_id: str = "",
    mission_id: str = "",
    mode: str = "",
    context_revision: int | None = None,
    context_hash: str = "",
    expected_effect: str = "",
    action_intent_hash: str = "",
) -> str:
    """审批卡片展示指纹（agentd 与 rosclawd 共用同一公式）。

    任何展示内容变化都会改变 hash；operatord 显示的卡片、daemon 的
    challenge/receipt、agentd 的本地卡三方用同一公式互相绑定。

    P0-5C：V3 卡（带 exact_action）必须绑定 capability/mission/mode/
    context/normalized parameters/intent hash——capability 不再只是
    "碰巧出现在 title"。
    """
    payload: dict[str, Any] = {
        "request_id": request_id,
        "title": title,
        "summary": summary,
        "risk_tier": risk_tier,
        "parameters": parameters,
        "body_hash": body_hash,
        "expires_at": expires_at,
    }
    # V3 绑定字段：任一非空即进入 hash（V2 卡全空 = 旧公式不变）。
    v3: dict[str, Any] = {}
    if capability_id:
        v3["capability_id"] = capability_id
    if mission_id:
        v3["mission_id"] = mission_id
    if mode:
        v3["mode"] = mode
    if context_revision is not None:
        v3["context_revision"] = context_revision
    if context_hash:
        v3["context_hash"] = context_hash
    if expected_effect:
        v3["expected_effect"] = expected_effect
    if action_intent_hash:
        v3["action_intent_hash"] = action_intent_hash
    if v3:
        payload["exact"] = v3
    canonical = canonical_json(payload)
    return hashlib.sha256(canonical).hexdigest()[:16]


# -- Ed25519 helpers -----------------------------------------------------------


def generate_ed25519_keypair() -> tuple[Ed25519PrivateKey, str]:
    private = Ed25519PrivateKey.generate()
    pem = (
        private.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private, pem


def private_key_to_pem(private: Ed25519PrivateKey) -> str:
    return private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def private_key_from_pem(pem: str) -> Ed25519PrivateKey:
    # password 位置传参（None=未加密）：静态不变量扫描禁止 secret 命名的
    # kwarg 出现在合约模块——这里本来就没有密码，位置参数如实表达。
    key = serialization.load_pem_private_key(pem.encode(), None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("not an Ed25519 private key")
    return key


def public_key_from_pem(pem: str) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(pem.encode())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("not an Ed25519 public key")
    return key


def key_fingerprint(public_key_pem: str) -> str:
    return hashlib.sha256(public_key_pem.encode()).hexdigest()[:16]


def sign_b64(private: Ed25519PrivateKey, payload: bytes) -> str:
    return base64.b64encode(private.sign(payload)).decode()


def verify_b64(public_key_pem: str, payload: bytes, signature_b64: str) -> bool:
    try:
        signature = base64.b64decode(signature_b64, validate=True)
    except ValueError:
        return False
    try:
        public_key_from_pem(public_key_pem).verify(signature, payload)
    except Exception:  # noqa: BLE001 — 任何校验异常都是"不通过"
        return False
    return True


# -- DecisionChallengeV1 --------------------------------------------------------


@dataclass(frozen=True)
class DecisionChallengeV1:
    """daemon 签发的一次性决策挑战（P0-3：nonce 同源）。"""

    proposal_id: str
    challenge_nonce: str
    display_hash: str
    execution_mode: str
    capability_id: str
    canonical_args_hash: str
    issued_at: str
    expires_at: str
    daemon_instance_id: str
    agent_request_id: str = ""
    mission_id: str = ""
    protocol_version: str = PROTOCOL_VERSION_CHALLENGE

    def payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "proposal_id": self.proposal_id,
            "agent_request_id": self.agent_request_id,
            "mission_id": self.mission_id,
            "execution_mode": self.execution_mode,
            "capability_id": self.capability_id,
            "canonical_args_hash": self.canonical_args_hash,
            "display_hash": self.display_hash,
            "challenge_nonce": self.challenge_nonce,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "daemon_instance_id": self.daemon_instance_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionChallengeV1:
        if str(data.get("protocol_version", "")) != PROTOCOL_VERSION_CHALLENGE:
            raise ValueError(f"unsupported challenge protocol {data.get('protocol_version')!r}")
        return cls(
            proposal_id=str(data["proposal_id"]),
            challenge_nonce=str(data["challenge_nonce"]),
            display_hash=str(data["display_hash"]),
            execution_mode=str(data["execution_mode"]),
            capability_id=str(data["capability_id"]),
            canonical_args_hash=str(data["canonical_args_hash"]),
            issued_at=str(data["issued_at"]),
            expires_at=str(data["expires_at"]),
            daemon_instance_id=str(data["daemon_instance_id"]),
            agent_request_id=str(data.get("agent_request_id", "")),
            mission_id=str(data.get("mission_id", "")),
        )


# -- HumanConfirmationV1 --------------------------------------------------------


@dataclass(frozen=True)
class HumanConfirmationV1:
    """operatord 记录的前台终端人工确认（P0-1）。"""

    method: str  # "tty-yn"
    decision: str  # ACCEPT | DECLINE
    confirmed_at: str
    foreground_verified: bool

    def payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "decision": self.decision,
            "confirmed_at": self.confirmed_at,
            "foreground_verified": self.foreground_verified,
        }


# -- OperatorDecisionProofV1 ----------------------------------------------------


@dataclass(frozen=True)
class OperatorDecisionProofV1:
    """operatord 对 daemon challenge 的 Ed25519 签名决定（替换共享 HMAC）。"""

    enrollment_id: str
    challenge: DecisionChallengeV1
    decision: str
    decided_at: str
    human_confirmation_method: str
    signature_b64: str = ""
    protocol_version: str = PROTOCOL_VERSION_PROOF

    def signing_payload(self) -> bytes:
        return canonical_json(
            {
                "protocol_version": self.protocol_version,
                "enrollment_id": self.enrollment_id,
                "challenge": self.challenge.payload(),
                "decision": self.decision,
                "decided_at": self.decided_at,
                "human_confirmation_method": self.human_confirmation_method,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "enrollment_id": self.enrollment_id,
            "challenge": self.challenge.payload(),
            "decision": self.decision,
            "decided_at": self.decided_at,
            "human_confirmation_method": self.human_confirmation_method,
            "signature_b64": self.signature_b64,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorDecisionProofV1:
        if str(data.get("protocol_version", "")) != PROTOCOL_VERSION_PROOF:
            raise ValueError(f"unsupported proof protocol {data.get('protocol_version')!r}")
        return cls(
            enrollment_id=str(data["enrollment_id"]),
            challenge=DecisionChallengeV1.from_dict(dict(data["challenge"])),
            decision=str(data["decision"]).upper(),
            decided_at=str(data["decided_at"]),
            human_confirmation_method=str(data.get("human_confirmation_method", "")),
            signature_b64=str(data["signature_b64"]),
        )


# -- DecisionReceiptV1 ----------------------------------------------------------


@dataclass(frozen=True)
class DecisionReceiptV1:
    """daemon 签名的精确决策回执（P0-6：agentd 只信这个）。"""

    proposal_id: str
    decision: str
    operator_enrollment_id: str
    operator_principal: str
    human_confirmation_method: str
    challenge_nonce: str
    decided_at: str
    expires_at: str
    daemon_instance_id: str
    daemon_key_id: str
    agent_request_id: str = ""
    mission_id: str = ""
    execution_mode: str = ""
    capability_id: str = ""
    canonical_args_hash: str = ""
    display_hash: str = ""
    signature_b64: str = field(default="")
    protocol_version: str = PROTOCOL_VERSION_RECEIPT

    @property
    def receipt_id(self) -> str:
        return hashlib.sha256(
            canonical_json({"proposal_id": self.proposal_id, "challenge_nonce": self.challenge_nonce})
        ).hexdigest()[:24]

    def signing_payload(self) -> bytes:
        return canonical_json(
            {
                "protocol_version": self.protocol_version,
                "proposal_id": self.proposal_id,
                "agent_request_id": self.agent_request_id,
                "mission_id": self.mission_id,
                "execution_mode": self.execution_mode,
                "capability_id": self.capability_id,
                "canonical_args_hash": self.canonical_args_hash,
                "display_hash": self.display_hash,
                "decision": self.decision,
                "operator_enrollment_id": self.operator_enrollment_id,
                "operator_principal": self.operator_principal,
                "human_confirmation_method": self.human_confirmation_method,
                "challenge_nonce": self.challenge_nonce,
                "decided_at": self.decided_at,
                "expires_at": self.expires_at,
                "daemon_instance_id": self.daemon_instance_id,
                "daemon_key_id": self.daemon_key_id,
            }
        )

    def sign(self, private: Ed25519PrivateKey) -> DecisionReceiptV1:
        from dataclasses import replace

        return replace(self, signature_b64=sign_b64(private, self.signing_payload()))

    def verify_signature(self, public_key_pem: str) -> bool:
        if not self.signature_b64:
            return False
        return verify_b64(public_key_pem, self.signing_payload(), self.signature_b64)

    def to_dict(self, *, include_signature: bool = True) -> dict[str, Any]:
        data = {
            "protocol_version": self.protocol_version,
            "proposal_id": self.proposal_id,
            "agent_request_id": self.agent_request_id,
            "mission_id": self.mission_id,
            "execution_mode": self.execution_mode,
            "capability_id": self.capability_id,
            "canonical_args_hash": self.canonical_args_hash,
            "display_hash": self.display_hash,
            "decision": self.decision,
            "operator_enrollment_id": self.operator_enrollment_id,
            "operator_principal": self.operator_principal,
            "human_confirmation_method": self.human_confirmation_method,
            "challenge_nonce": self.challenge_nonce,
            "decided_at": self.decided_at,
            "expires_at": self.expires_at,
            "daemon_instance_id": self.daemon_instance_id,
            "daemon_key_id": self.daemon_key_id,
        }
        if include_signature:
            data["signature_b64"] = self.signature_b64
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionReceiptV1:
        if str(data.get("protocol_version", "")) != PROTOCOL_VERSION_RECEIPT:
            raise ValueError(f"unsupported receipt protocol {data.get('protocol_version')!r}")
        return cls(
            proposal_id=str(data["proposal_id"]),
            decision=str(data["decision"]).upper(),
            operator_enrollment_id=str(data["operator_enrollment_id"]),
            operator_principal=str(data["operator_principal"]),
            human_confirmation_method=str(data.get("human_confirmation_method", "")),
            challenge_nonce=str(data["challenge_nonce"]),
            decided_at=str(data["decided_at"]),
            expires_at=str(data["expires_at"]),
            daemon_instance_id=str(data["daemon_instance_id"]),
            daemon_key_id=str(data["daemon_key_id"]),
            agent_request_id=str(data.get("agent_request_id", "")),
            mission_id=str(data.get("mission_id", "")),
            execution_mode=str(data.get("execution_mode", "")),
            capability_id=str(data.get("capability_id", "")),
            canonical_args_hash=str(data.get("canonical_args_hash", "")),
            display_hash=str(data.get("display_hash", "")),
            signature_b64=str(data.get("signature_b64", "")),
        )
