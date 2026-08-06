"""VerifierRegistry — VERIFY intent 的确定性验证层（大纲 §5.7）。

验证器族：
- deterministic：schema、pose bounds、battery threshold、hash。
- receipt：action_id/body/hash/evidence domain 匹配。
- observation：占位（接传感器后注册）。
- task-specific：导航到达等（按 mission 注册）。
- human：显式 human_attested（绝不默认可信）。

模型永远不能自己说"看起来成功"——只能引用注册验证器并提交证据。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol


class UnknownVerifierError(Exception):
    pass


@dataclass(frozen=True)
class VerifierResult:
    verifier_id: str
    success: bool
    evidence_refs: tuple[str, ...] = ()
    measured_at: str = ""
    confidence: float = 0.0
    failure_reason: str | None = None
    human_attested: bool = False


class Verifier(Protocol):
    verifier_id: str

    def run(self, context: dict[str, Any]) -> VerifierResult: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SchemaVerifier:
    verifier_id = "deterministic.schema.v1"

    def run(self, context: dict[str, Any]) -> VerifierResult:
        required = context.get("required_fields") or []
        payload = context.get("payload") or {}
        missing = [f for f in required if f not in payload]
        return VerifierResult(
            verifier_id=self.verifier_id,
            success=not missing,
            evidence_refs=tuple(context.get("evidence_refs") or ()),
            measured_at=_now(),
            confidence=1.0 if not missing else 0.0,
            failure_reason=f"missing fields: {missing}" if missing else None,
        )


class PoseBoundsVerifier:
    verifier_id = "localization.pose_bounds.v1"

    def run(self, context: dict[str, Any]) -> VerifierResult:
        pose = context.get("pose") or {}
        target = context.get("target") or {}
        tolerance = float(context.get("tolerance", 0.05))
        dx = abs(float(pose.get("x", 0.0)) - float(target.get("x", 0.0)))
        dy = abs(float(pose.get("y", 0.0)) - float(target.get("y", 0.0)))
        dyaw = abs(float(pose.get("yaw", 0.0)) - float(target.get("yaw", 0.0)))
        ok = dx <= tolerance and dy <= tolerance and dyaw <= max(tolerance, 0.1)
        return VerifierResult(
            verifier_id=self.verifier_id,
            success=ok,
            evidence_refs=tuple(context.get("evidence_refs") or ()),
            measured_at=_now(),
            confidence=0.98 if ok else 0.0,
            failure_reason=None if ok else f"deviation ({dx:.3f},{dy:.3f},{dyaw:.3f})",
        )


class ReceiptMatchVerifier:
    verifier_id = "receipt.action_match.v1"

    def run(self, context: dict[str, Any]) -> VerifierResult:
        receipt = context.get("receipt") or {}
        expected_action_id = context.get("action_id")
        trust = receipt.get("trust_level", "")
        ok = (
            receipt.get("action_id") == expected_action_id
            and trust not in ("SYNTHETIC", "UNKNOWN", "")
            and receipt.get("final_state") == "COMPLETED"
        )
        return VerifierResult(
            verifier_id=self.verifier_id,
            success=ok,
            evidence_refs=tuple(context.get("evidence_refs") or ()),
            measured_at=_now(),
            confidence=0.99 if ok else 0.0,
            failure_reason=(
                None if ok else f"receipt mismatch/trust={trust}/state={receipt.get('final_state')}"
            ),
        )


class HumanAttestVerifier:
    verifier_id = "human.attested.v1"

    def run(self, context: dict[str, Any]) -> VerifierResult:
        attested = bool(context.get("human_attested"))
        return VerifierResult(
            verifier_id=self.verifier_id,
            success=attested,
            evidence_refs=tuple(context.get("evidence_refs") or ()),
            measured_at=_now(),
            confidence=0.5 if attested else 0.0,
            failure_reason=None if attested else "no explicit human attestation",
            human_attested=True,
        )


class VerifierRegistry:
    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}
        for verifier in (
            SchemaVerifier(),
            PoseBoundsVerifier(),
            ReceiptMatchVerifier(),
            HumanAttestVerifier(),
        ):
            self.register(verifier)

    def register(self, verifier: Verifier) -> None:
        self._verifiers[verifier.verifier_id] = verifier

    def run(self, verifier_id: str, context: dict[str, Any]) -> VerifierResult:
        verifier = self._verifiers.get(verifier_id)
        if verifier is None:
            raise UnknownVerifierError(f"unknown verifier {verifier_id!r}")
        return verifier.run(context)

    def run_many(self, verifier_ids: list[str], context: dict[str, Any]) -> VerifierResult:
        """AND 语义：任一失败即整体失败（返回首个失败结果）。"""
        last: VerifierResult | None = None
        for verifier_id in verifier_ids:
            result = self.run(verifier_id, context)
            if not result.success:
                return result
            last = result
        return last or VerifierResult(
            verifier_id="registry.empty", success=False, failure_reason="no verifiers given"
        )
