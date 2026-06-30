"""Generic body-cognition and promotion-gate types.

BodyCognition stores learned traits about a physical body: force-model
configuration, sim2real deltas, promoted policies, and known safe poses.
PromotionGateResult records whether a candidate policy passed repeatability
and safety gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromotionGateResult:
    """Result of a promotion gate evaluation for one policy."""

    policy_id: str = ""
    passed: bool = False
    rounds: int = 0
    contact_detected_rate: float = 0.0
    force_window_pass_rate: float = 0.0
    shape_score_mean: float = 0.0
    max_force_net: Optional[float] = None
    max_temp_c: Optional[float] = None
    errors: int = 0
    status_protection_events: int = 0
    over_contact_events: int = 0
    failures: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    evidence_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromotionGateResult":
        return cls(**data)


@dataclass
class BodyCognition:
    """Generic learned-body-knowledge container.

    Mirrors the structure used by RH56's ~/.rosclaw-rh56/body/body_cognition.yaml
    but without RH56-specific field names, so other bodies can reuse it.
    """

    body_id: str = ""
    schema_version: str = "rosclaw.body.cognition.v1"
    updated_at: str = ""

    force_model: Dict[str, Any] = field(default_factory=dict)
    known_traits: List[str] = field(default_factory=list)
    promoted_policies: List[Dict[str, Any]] = field(default_factory=list)
    countdown_validations: List[Dict[str, Any]] = field(default_factory=list)
    sim2real_deltas: Dict[str, Any] = field(default_factory=dict)
    search_history: List[Dict[str, Any]] = field(default_factory=list)

    # Free-form notes for humans and agents.
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BodyCognition":
        return cls(**data)

    def get_promoted_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Return the latest promoted policy with the given id."""
        for policy in reversed(self.promoted_policies):
            if policy.get("policy_id") == policy_id:
                return policy
        return None
