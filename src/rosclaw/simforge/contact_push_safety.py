"""Physics-backed sandbox screening for ContactPush candidates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from rosclaw.simforge.tasks.contact_push_v3 import (
    ContactPushPhysics,
    ContactPushPolicy,
    ContactPushScenario,
    ContactPushStatus,
)


@dataclass(frozen=True)
class ContactPushSandboxDecision:
    allowed: bool
    reason: str
    status: ContactPushStatus
    peak_force_n: float
    result_hash: str
    replay_verified: bool
    schema_version: str = "rosclaw.contact_push_sandbox_decision.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allowed": self.allowed,
            "reason": self.reason,
            "status": self.status.value,
            "peak_force_n": self.peak_force_n,
            "result_hash": self.result_hash,
            "replay_verified": self.replay_verified,
        }


class ContactPushSandboxVerifier:
    """Use an isolated MuJoCo rollout as a fail-closed pre-activation shield."""

    def __init__(self, physics: ContactPushPhysics | None = None) -> None:
        self.physics = physics or ContactPushPhysics(trace_stride=100)

    def screen(
        self,
        *,
        scenario: ContactPushScenario,
        policy: ContactPushPolicy,
    ) -> ContactPushSandboxDecision:
        result = self.physics.run(scenario, policy)
        replay = self.physics.run(scenario, policy)
        result_hash = _hash_json(result.summary_dict())
        replay_verified = result_hash == _hash_json(replay.summary_dict())
        if not replay_verified:
            return ContactPushSandboxDecision(
                allowed=False,
                reason="NON_DETERMINISTIC_PHYSICS",
                status=result.status,
                peak_force_n=result.peak_contact_force_n,
                result_hash=result_hash,
                replay_verified=False,
            )
        if result.status is ContactPushStatus.NON_FINITE:
            reason = "NON_FINITE_STATE"
            allowed = False
        elif result.status is ContactPushStatus.FORCE_LIMIT:
            reason = "FORCE_LIMIT_EXCEEDED"
            allowed = False
        else:
            reason = "SAFETY_VERIFIED"
            allowed = True
        return ContactPushSandboxDecision(
            allowed=allowed,
            reason=reason,
            status=result.status,
            peak_force_n=result.peak_contact_force_n,
            result_hash=result_hash,
            replay_verified=True,
        )


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "ContactPushSandboxDecision",
    "ContactPushSandboxVerifier",
]
