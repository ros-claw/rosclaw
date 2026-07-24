"""EVO-3 arm policies (数据库优化v4 §11).

The decision logic per arm, isolated from any transport:

* A (No Memory): failures get no intervention, ever.
* B (Fixed Cooldown): every failure gets the same static cooldown patch —
  the run2 control arm (2.0% invalid, 0 Memory Hurt).
* C (Regime-aware): the selective pipeline decides; in a healthy regime
  the thermal memory yields ABSTAIN (the Memory Hurt ≈ 0 hypothesis).

``ArmC`` is wired exactly like production: facade retrieval (HOW path) →
regime builder → matcher → selective decision → choreography gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.how.selective import InterventionAction

FIXED_COOLDOWN_PATCH = {"inter_round_cooldown_sec": 5}


@dataclass
class ArmOutcome:
    """One failure's arm outcome (what the arm would do)."""

    arm: str
    acted: bool
    patch: dict[str, Any] | None
    decision_action: str | None = None
    reason_codes: list[str] | None = None
    selected_memory_id: str | None = None


class ArmA:
    name = "A_no_memory"

    def respond(self, failure: dict[str, Any], regime: Any) -> ArmOutcome:
        return ArmOutcome(arm=self.name, acted=False, patch=None)


class ArmB:
    name = "B_fixed_cooldown"

    def __init__(self, patch: dict[str, Any] | None = None) -> None:
        self._patch = dict(patch or FIXED_COOLDOWN_PATCH)

    def respond(self, failure: dict[str, Any], regime: Any) -> ArmOutcome:
        return ArmOutcome(arm=self.name, acted=True, patch=dict(self._patch))


class ArmC:
    name = "C_regime_aware"

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def respond(self, failure: dict[str, Any], regime: Any) -> ArmOutcome:
        decision = self._pipeline.decide(
            str(failure.get("signature") or failure.get("failure_type") or ""),
            regime,
            body_id=failure.get("body_id") or regime.body_id,
            joint_name=failure.get("joint_name"),
        )
        acted = decision.action is InterventionAction.APPLY
        return ArmOutcome(
            arm=self.name,
            acted=acted,
            patch=decision.suggested_patch if acted else None,
            decision_action=decision.action.value,
            reason_codes=decision.reason_codes,
            selected_memory_id=decision.selected_memory_id,
        )


def build_arm(
    arm: str,
    *,
    pipeline: Any | None = None,
) -> Any:
    if arm == ArmA.name:
        return ArmA()
    if arm == ArmB.name:
        return ArmB()
    if arm == ArmC.name:
        if pipeline is None:
            raise ValueError("ArmC requires the selective pipeline")
        return ArmC(pipeline)
    raise ValueError(f"unknown arm {arm!r}")
