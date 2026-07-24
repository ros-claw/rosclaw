"""Causal Provider routing experiment for ContactPush.

The harness models the narrow contract between a model Provider and the
planner. Provider output has no actuator authority: it is schema checked,
confidence gated, and converted into a policy only after task constraints pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rosclaw.simforge.contact_push_learning import (
    ContactPushCandidate,
    ContactPushTaskKnowledge,
)
from rosclaw.simforge.tasks.contact_push_v3 import (
    ContactPushPhysics,
    ContactPushPolicy,
    ContactPushResult,
    ContactPushScenario,
)


class ProviderRoute(StrEnum):
    MODEL_PLAN = "MODEL_PLAN"
    SAFE_STOP = "SAFE_STOP"
    REJECT_INVALID = "REJECT_INVALID"
    REJECT_LOW_CONFIDENCE = "REJECT_LOW_CONFIDENCE"


@dataclass(frozen=True)
class ProviderResponse:
    status: str
    confidence: float
    policy: dict[str, Any] | None


@dataclass(frozen=True)
class ProviderDecision:
    route: ProviderRoute
    policy: ContactPushPolicy | None
    reason: str
    executable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.value,
            "policy": self.policy.to_dict() if self.policy is not None else None,
            "reason": self.reason,
            "executable": self.executable,
        }


@dataclass(frozen=True)
class ContactPushProviderCausalResult:
    baseline: ContactPushResult
    treatment: ContactPushResult
    treatment_replay: ContactPushResult
    correct: ProviderDecision
    timeout: ProviderDecision
    illegal: ProviderDecision
    low_confidence: ProviderDecision
    output_hash: str
    schema_version: str = "rosclaw.contact_push_provider_causal.v1"

    @property
    def decision_changed(self) -> bool:
        return (
            self.correct.route is ProviderRoute.MODEL_PLAN
            and self.timeout.route is ProviderRoute.SAFE_STOP
        )

    @property
    def outcome_changed(self) -> bool:
        return not self.baseline.success and self.treatment.success

    @property
    def strict_replay(self) -> bool:
        return (
            self.treatment.summary_dict() == self.treatment_replay.summary_dict()
            and self.treatment.trace == self.treatment_replay.trace
        )

    @property
    def fault_routes_passed(self) -> bool:
        return bool(
            self.timeout.route is ProviderRoute.SAFE_STOP
            and not self.timeout.executable
            and self.illegal.route is ProviderRoute.REJECT_INVALID
            and not self.illegal.executable
            and self.low_confidence.route is ProviderRoute.REJECT_LOW_CONFIDENCE
            and not self.low_confidence.executable
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline": self.baseline.summary_dict(),
            "treatment": self.treatment.summary_dict(),
            "treatment_replay": self.treatment_replay.summary_dict(),
            "routes": {
                "correct": self.correct.to_dict(),
                "timeout": self.timeout.to_dict(),
                "illegal": self.illegal.to_dict(),
                "low_confidence": self.low_confidence.to_dict(),
            },
            "decision_changed": self.decision_changed,
            "outcome_changed": self.outcome_changed,
            "strict_replay": self.strict_replay,
            "fault_routes_passed": self.fault_routes_passed,
            "output_hash": self.output_hash,
        }


class ContactPushProviderRouter:
    def __init__(self, *, minimum_confidence: float = 0.80) -> None:
        if not math.isfinite(minimum_confidence) or not 0 < minimum_confidence <= 1:
            raise ValueError("minimum Provider confidence must be in (0, 1]")
        self.minimum_confidence = minimum_confidence
        self.knowledge = ContactPushTaskKnowledge.default()

    def route(self, response: ProviderResponse) -> ProviderDecision:
        if response.status == "timeout":
            return ProviderDecision(
                route=ProviderRoute.SAFE_STOP,
                policy=None,
                reason="provider_timeout_requires_replan",
                executable=False,
            )
        if response.status != "ok" or not isinstance(response.policy, dict):
            return ProviderDecision(
                route=ProviderRoute.REJECT_INVALID,
                policy=None,
                reason="provider_response_schema_invalid",
                executable=False,
            )
        if not math.isfinite(response.confidence):
            return ProviderDecision(
                route=ProviderRoute.REJECT_INVALID,
                policy=None,
                reason="provider_confidence_non_finite",
                executable=False,
            )
        if response.confidence < self.minimum_confidence:
            return ProviderDecision(
                route=ProviderRoute.REJECT_LOW_CONFIDENCE,
                policy=None,
                reason="provider_confidence_below_gate",
                executable=False,
            )
        try:
            policy = _policy_from_provider(response.policy)
        except (KeyError, TypeError, ValueError):
            return ProviderDecision(
                route=ProviderRoute.REJECT_INVALID,
                policy=None,
                reason="provider_policy_invalid",
                executable=False,
            )
        valid, errors = self.knowledge.validate_policy(policy)
        if not valid:
            return ProviderDecision(
                route=ProviderRoute.REJECT_INVALID,
                policy=None,
                reason="task_card_rejected:" + ",".join(errors),
                executable=False,
            )
        return ProviderDecision(
            route=ProviderRoute.MODEL_PLAN,
            policy=policy,
            reason="schema_confidence_and_task_card_passed",
            executable=True,
        )


def evaluate_contact_push_provider(
    *,
    scenario: ContactPushScenario,
    candidate: ContactPushCandidate,
) -> ContactPushProviderCausalResult:
    physics = ContactPushPhysics(trace_stride=10)
    model_policy = candidate.policy_for(scenario)
    router = ContactPushProviderRouter()
    correct = router.route(
        ProviderResponse(
            status="ok",
            confidence=0.97,
            policy=model_policy.to_dict(),
        )
    )
    timeout = router.route(ProviderResponse(status="timeout", confidence=0.0, policy=None))
    illegal = router.route(
        ProviderResponse(
            status="ok",
            confidence=0.99,
            policy={**model_policy.to_dict(), "push_velocity_mps": 99.0},
        )
    )
    low_confidence = router.route(
        ProviderResponse(
            status="ok",
            confidence=0.30,
            policy=model_policy.to_dict(),
        )
    )
    if correct.policy is None:
        raise RuntimeError("correct Provider response did not produce a policy")
    baseline = physics.run(scenario, ContactPushPolicy.baseline())
    treatment = physics.run(scenario, correct.policy)
    replay = physics.run(scenario, correct.policy)
    identity = {
        "scenario": scenario.scenario_commitment,
        "candidate": candidate.candidate_hash,
        "baseline": baseline.summary_dict(),
        "treatment": treatment.summary_dict(),
        "routes": {
            "correct": correct.to_dict(),
            "timeout": timeout.to_dict(),
            "illegal": illegal.to_dict(),
            "low_confidence": low_confidence.to_dict(),
        },
    }
    result = ContactPushProviderCausalResult(
        baseline=baseline,
        treatment=treatment,
        treatment_replay=replay,
        correct=correct,
        timeout=timeout,
        illegal=illegal,
        low_confidence=low_confidence,
        output_hash=_hash_json(identity),
    )
    if not (
        result.decision_changed
        and result.outcome_changed
        and result.fault_routes_passed
        and result.strict_replay
    ):
        raise RuntimeError("Provider causal experiment did not satisfy its contract")
    return result


def _policy_from_provider(value: dict[str, Any]) -> ContactPushPolicy:
    allowed = {
        "push_velocity_mps",
        "contact_duration_sec",
        "contact_offset_y_m",
        "deceleration_fraction",
        "micro_push",
        "policy_type",
    }
    if set(value) != allowed:
        raise ValueError("Provider policy fields do not match the allowlist")
    if not isinstance(value["micro_push"], bool):
        raise TypeError("Provider micro_push must be boolean")
    return ContactPushPolicy(
        push_velocity_mps=_number(value["push_velocity_mps"]),
        contact_duration_sec=_number(value["contact_duration_sec"]),
        contact_offset_y_m=_number(value["contact_offset_y_m"]),
        deceleration_fraction=_number(value["deceleration_fraction"]),
        micro_push=value["micro_push"],
        policy_type=str(value["policy_type"]),
    )


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Provider numeric policy fields must be numbers")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("Provider numeric policy fields must be finite")
    return normalized


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "ContactPushProviderCausalResult",
    "ContactPushProviderRouter",
    "ProviderDecision",
    "ProviderResponse",
    "ProviderRoute",
    "evaluate_contact_push_provider",
]
