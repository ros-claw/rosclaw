"""Two-stage worker scheduling (总纲 §9.6).

Stage A: deterministic hard filters — safety and permission are never
scored, they are gates. Stage B: explainable weighted scoring with the
feature vector, score, policy version and reason recorded for audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rosclaw.contracts.worker.card import WorkerCardV1
from rosclaw.contracts.worker.order import WorkOrderV1

SCHEDULER_POLICY_VERSION = "scheduler.v1"

#: (capability, reliability, availability, latency, cost, privacy, diversity)
_WEIGHTS = (0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05)


class SchedulingError(Exception):
    """No worker can take this order."""


@dataclass(frozen=True)
class CandidateView:
    """Runtime evidence the scheduler may use (cards alone are declarations)."""

    card: WorkerCardV1
    registry_status: str = "ENABLED"
    ready: bool = True
    running_orders: int = 0
    reliability: float = 0.5  # task-family success rate, 0..1
    avg_latency_ms: float = 10_000.0
    cost_rank: float = 0.5  # 0 cheapest .. 1 most expensive
    circuit_open: bool = False
    trust_evidence: int = 0


@dataclass(frozen=True)
class ScoredCandidate:
    worker_id: str
    score: float
    features: dict[str, float]
    reasons: tuple[str, ...] = field(default_factory=tuple)


def hard_filter(order: WorkOrderV1, view: CandidateView) -> str | None:
    """Return a rejection reason, or None if the candidate passes."""
    card = view.card
    if view.registry_status != "ENABLED":
        return f"worker registry status is {view.registry_status}"
    if view.circuit_open:
        return "circuit breaker open for this capability family"
    if not view.ready:
        return "worker health is not READY"
    cap = next((c for c in card.capabilities if c.name == order.capability), None)
    if cap is None:
        return f"capability {order.capability!r} not declared"
    if cap.side_effect_class != order.side_effect_policy.class_:
        return (
            f"side-effect class mismatch: capability {cap.side_effect_class!r} "
            f"vs order {order.side_effect_policy.class_!r}"
        )
    if order.side_effect_policy.class_ == "physical":
        return "physical side effects are forbidden for cognitive workers"
    if view.running_orders >= card.constraints.max_concurrency:
        return (
            f"concurrency budget exhausted "
            f"({view.running_orders}/{card.constraints.max_concurrency})"
        )
    if order.delegation_depth > 0 and order.budgets.max_children < 1:
        return "delegation depth exceeds max_children budget"
    import sys

    if card.constraints.supported_platforms and not any(
        sys.platform.startswith(p) for p in card.constraints.supported_platforms
    ):
        return f"platform {sys.platform} not supported"
    return None


def score_candidate(order: WorkOrderV1, view: CandidateView) -> ScoredCandidate:
    card = view.card
    capability_fit = 1.0  # hard filter already ensured declaration
    reliability = min(max(view.reliability, 0.0), 1.0)
    availability = 1.0 - min(view.running_orders / max(card.constraints.max_concurrency, 1), 1.0)
    # Latency fit: deadline-relative; no deadline → neutral 0.5.
    latency_fit = 0.5
    if order.inputs.get("deadline_ms"):
        budget_ms = float(order.inputs["deadline_ms"])
        latency_fit = min(max(1.0 - view.avg_latency_ms / max(budget_ms, 1.0), 0.0), 1.0)
    cost_fit = 1.0 - min(max(view.cost_rank, 0.0), 1.0)
    privacy_fit = 1.0 if card.security.isolation in ("process", "container") else 0.5
    diversity = 0.5  # caller may boost for independent verification picks
    features = {
        "capability": capability_fit,
        "reliability": reliability,
        "availability": availability,
        "latency": latency_fit,
        "cost": cost_fit,
        "privacy": privacy_fit,
        "diversity": diversity,
    }
    score = sum(w * f for w, f in zip(_WEIGHTS, features.values(), strict=True))
    reasons = tuple(f"{k}={v:.2f}" for k, v in features.items())
    return ScoredCandidate(
        worker_id=card.worker_id, score=round(score, 4), features=features, reasons=reasons
    )


class Scheduler:
    """Stateless two-stage scheduler. Selections are auditable."""

    policy_version = SCHEDULER_POLICY_VERSION

    def select(
        self, order: WorkOrderV1, candidates: list[CandidateView]
    ) -> tuple[CandidateView, ScoredCandidate]:
        scored: list[tuple[CandidateView, ScoredCandidate]] = []
        rejections: dict[str, str] = {}
        for view in candidates:
            reason = hard_filter(order, view)
            if reason is not None:
                rejections[view.card.worker_id] = reason
                continue
            scored.append((view, score_candidate(order, view)))
        if not scored:
            detail = "; ".join(f"{w}: {r}" for w, r in sorted(rejections.items()))
            raise SchedulingError(f"no eligible worker for {order.capability!r}: {detail}")
        scored.sort(key=lambda item: item[1].score, reverse=True)
        return scored[0]
