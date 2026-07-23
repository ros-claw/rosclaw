"""Paired outcome aggregation and exact/bootstrapped statistics."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.simforge.models import HumanInvolvement, Partition
from rosclaw.simforge.monitors import RobustnessAggregator


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    collision: bool
    unsafe_allow: bool
    false_block: bool
    robustness: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.robustness):
            raise ValueError("episode robustness must be finite")


@dataclass(frozen=True)
class PairedEpisode:
    pair_id: str
    scenario_commitment: str
    seed_commitment: str
    baseline: EpisodeOutcome
    candidate: EpisodeOutcome
    physics_executed: bool
    independently_verified: bool
    strict_replay: bool
    artifact_hash_valid: bool
    data_quality_valid: bool

    def __post_init__(self) -> None:
        if not self.pair_id or not self.scenario_commitment or not self.seed_commitment:
            raise ValueError("paired episodes require identity commitments")


@dataclass(frozen=True)
class AggregateMetrics:
    baseline_success_rate: float
    candidate_success_rate: float
    baseline_collision_rate: float
    candidate_collision_rate: float
    baseline_unsafe_allow_rate: float
    candidate_unsafe_allow_rate: float
    baseline_false_block_rate: float
    candidate_false_block_rate: float
    baseline_p05_robustness: float
    candidate_p05_robustness: float
    baseline_cvar05_robustness: float
    candidate_cvar05_robustness: float
    success_mcnemar_pvalue: float
    success_delta_ci95: tuple[float, float]


@dataclass(frozen=True)
class EvidenceAttestation:
    physics_complete: bool
    independently_verified: bool
    scenario_seed_paired: bool
    strict_replay: bool
    artifact_hashes_valid: bool
    data_quality_valid: bool
    shards_complete: bool = True


@dataclass(frozen=True)
class EvaluationBundle:
    task_id: str
    candidate_hash: str
    partition: Partition
    paired_episodes: int
    metrics: AggregateMetrics
    attestation: EvidenceAttestation
    pair_set_commitment: str
    human_involvement: HumanInvolvement
    evidence_refs: tuple[str, ...] = ()
    schema_version: str = "rosclaw.simforge.evaluation_bundle.v1"

    @classmethod
    def from_pairs(
        cls,
        *,
        task_id: str,
        candidate_hash: str,
        partition: Partition,
        pairs: list[PairedEpisode] | tuple[PairedEpisode, ...],
        human_involvement: HumanInvolvement | None = None,
        bootstrap_seed: int = 0,
        evidence_refs: tuple[str, ...] = (),
    ) -> EvaluationBundle:
        if not pairs:
            raise ValueError("evaluation bundles require at least one pair")
        identities = [
            (pair.pair_id, pair.scenario_commitment, pair.seed_commitment) for pair in pairs
        ]
        unique = len({pair.pair_id for pair in pairs}) == len(pairs) and len(
            {(pair.scenario_commitment, pair.seed_commitment) for pair in pairs}
        ) == len(pairs)
        pair_hash = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(sorted(identities), separators=(",", ":")).encode()
            ).hexdigest()
        )
        baseline = [pair.baseline for pair in pairs]
        candidate = [pair.candidate for pair in pairs]
        deltas = [
            float(c.success) - float(b.success) for b, c in zip(baseline, candidate, strict=True)
        ]
        metrics = AggregateMetrics(
            baseline_success_rate=_rate(item.success for item in baseline),
            candidate_success_rate=_rate(item.success for item in candidate),
            baseline_collision_rate=_rate(item.collision for item in baseline),
            candidate_collision_rate=_rate(item.collision for item in candidate),
            baseline_unsafe_allow_rate=_rate(item.unsafe_allow for item in baseline),
            candidate_unsafe_allow_rate=_rate(item.unsafe_allow for item in candidate),
            baseline_false_block_rate=_rate(item.false_block for item in baseline),
            candidate_false_block_rate=_rate(item.false_block for item in candidate),
            baseline_p05_robustness=RobustnessAggregator.quantile(
                tuple(item.robustness for item in baseline), 0.05
            ),
            candidate_p05_robustness=RobustnessAggregator.quantile(
                tuple(item.robustness for item in candidate), 0.05
            ),
            baseline_cvar05_robustness=RobustnessAggregator.lower_tail_cvar(
                tuple(item.robustness for item in baseline), 0.05
            ),
            candidate_cvar05_robustness=RobustnessAggregator.lower_tail_cvar(
                tuple(item.robustness for item in candidate), 0.05
            ),
            success_mcnemar_pvalue=exact_mcnemar(baseline, candidate),
            success_delta_ci95=paired_bootstrap_ci(deltas, seed=bootstrap_seed),
        )
        return cls(
            task_id=task_id,
            candidate_hash=candidate_hash,
            partition=partition,
            paired_episodes=len(pairs),
            metrics=metrics,
            attestation=EvidenceAttestation(
                physics_complete=all(pair.physics_executed for pair in pairs),
                independently_verified=all(pair.independently_verified for pair in pairs),
                scenario_seed_paired=unique,
                strict_replay=all(pair.strict_replay for pair in pairs),
                artifact_hashes_valid=all(pair.artifact_hash_valid for pair in pairs),
                data_quality_valid=all(pair.data_quality_valid for pair in pairs),
            ),
            pair_set_commitment=pair_hash,
            human_involvement=human_involvement or HumanInvolvement(),
            evidence_refs=evidence_refs,
        )

    def aggregate_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "candidate_hash": self.candidate_hash,
            "partition": self.partition.value,
            "paired_episodes": self.paired_episodes,
            "aggregate_metrics": asdict(self.metrics),
            "attestation": asdict(self.attestation),
            "pair_set_commitment": self.pair_set_commitment,
            "human_involvement": asdict(self.human_involvement),
            "evidence_refs": list(self.evidence_refs),
            "complete": all(asdict(self.attestation).values()),
        }


def exact_mcnemar(baseline: list[EpisodeOutcome], candidate: list[EpisodeOutcome]) -> float:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("McNemar inputs must be non-empty and paired")
    baseline_only = sum(
        b.success and not c.success for b, c in zip(baseline, candidate, strict=True)
    )
    candidate_only = sum(
        c.success and not b.success for b, c in zip(baseline, candidate, strict=True)
    )
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(baseline_only, candidate_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_bootstrap_ci(
    deltas: list[float], *, seed: int, samples: int = 2000
) -> tuple[float, float]:
    if not deltas or samples < 100:
        raise ValueError("paired bootstrap requires data and at least 100 samples")
    rng = random.Random(seed)
    count = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples)
    )
    return (
        means[math.floor(0.025 * (samples - 1))],
        means[math.ceil(0.975 * (samples - 1))],
    )


def _rate(values: Any) -> float:
    normalized = list(values)
    return sum(map(bool, normalized)) / len(normalized)


__all__ = [
    "AggregateMetrics",
    "EpisodeOutcome",
    "EvaluationBundle",
    "EvidenceAttestation",
    "PairedEpisode",
    "exact_mcnemar",
    "paired_bootstrap_ci",
]
