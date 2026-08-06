"""Contract Net allocator (总纲 §10.6): explainable, deterministic.

announce → local-feasibility bids → deterministic feature scoring →
award (epoch + lease) → accept/reject/counter → heartbeat → evidence →
release dependencies. No model voting anywhere in the path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rosclaw.contracts.common import ValidationError

ALLOCATOR_POLICY_VERSION = "contract_net.v1"


@dataclass(frozen=True)
class TaskAnnouncement:
    task_id: str
    team_id: str
    team_epoch: int
    required_capabilities: tuple[str, ...]
    region: str | None = None
    deadline_ms: int | None = None
    risk_tier: str = "LOW"
    success_criteria: str = ""
    idempotency_key: str | None = None
    #: none | sandbox_process | workspace_write | network_write
    side_effect_class: str = "none"


@dataclass(frozen=True)
class Bid:
    member_id: str
    eta_ms: float
    energy_cost: float
    capability_fit: float  # 0..1 (fraction of required capabilities held)
    reliability: float  # 0..1 task-family success rate
    current_load: float  # 0..1
    comms_quality: float  # 0..1
    risk_fit: float = 1.0


@dataclass(frozen=True)
class ScoredBid:
    bid: Bid
    score: float
    features: dict[str, float]
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AllocationResult:
    task_id: str
    winner: str
    score: float
    scored_bids: tuple[ScoredBid, ...]
    policy_version: str = ALLOCATOR_POLICY_VERSION


#: eta, energy, capability, risk, reliability, load, comms (总纲 §10.6.4).
_WEIGHTS = {
    "eta": 0.20,
    "energy": 0.10,
    "capability": 0.25,
    "risk": 0.10,
    "reliability": 0.15,
    "load": 0.10,
    "comms": 0.10,
}

_MAX_ETA_MS = 60_000.0
_MAX_ENERGY = 1_000.0


def score_bid(announcement: TaskAnnouncement, bid: Bid) -> ScoredBid:
    features = {
        "eta": max(0.0, 1.0 - bid.eta_ms / _MAX_ETA_MS),
        "energy": max(0.0, 1.0 - bid.energy_cost / _MAX_ENERGY),
        "capability": bid.capability_fit,
        "risk": bid.risk_fit,
        "reliability": bid.reliability,
        "load": 1.0 - bid.current_load,
        "comms": bid.comms_quality,
    }
    score = sum(_WEIGHTS[k] * v for k, v in features.items())
    reasons = tuple(f"{k}={v:.2f}" for k, v in features.items())
    return ScoredBid(bid=bid, score=round(score, 4), features=features, reasons=reasons)


class ContractNetAllocator:
    policy_version = ALLOCATOR_POLICY_VERSION

    def allocate(self, announcement: TaskAnnouncement, bids: list[Bid]) -> AllocationResult:
        eligible: list[ScoredBid] = []
        rejections: list[str] = []
        for bid in bids:
            if bid.capability_fit < 1.0:
                rejections.append(f"{bid.member_id}: capability_fit {bid.capability_fit}")
                continue
            if announcement.deadline_ms and bid.eta_ms > announcement.deadline_ms:
                rejections.append(f"{bid.member_id}: eta {bid.eta_ms} > deadline")
                continue
            if bid.risk_fit <= 0.0:
                rejections.append(f"{bid.member_id}: risk not feasible")
                continue
            eligible.append(score_bid(announcement, bid))
        if not eligible:
            raise ValidationError(
                f"no feasible bidder for task {announcement.task_id!r}: {rejections}"
            )
        eligible.sort(key=lambda s: s.score, reverse=True)
        winner = eligible[0]
        return AllocationResult(
            task_id=announcement.task_id,
            winner=winner.bid.member_id,
            score=winner.score,
            scored_bids=tuple(eligible),
        )
