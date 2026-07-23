"""Paired outcome aggregation and exact/bootstrapped statistics."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import secrets
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

from rosclaw.simforge.models import HumanInvolvement, Partition
from rosclaw.simforge.monitors import RobustnessAggregator

_MAX_EVALUATION_PAIRS = 10_000
_MAX_BOOTSTRAP_SAMPLES = 100_000
_PROCESS_EVIDENCE_KEY = secrets.token_bytes(32)


class EvidenceVerificationSource(StrEnum):
    PHYSICS_RECEIPTS = "physics_receipts"
    SIGNED_HOLDOUT = "signed_holdout"
    SIGNED_SCALE_CURVE = "signed_scale_curve"


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    collision: bool
    unsafe_allow: bool
    false_block: bool
    robustness: float

    def __post_init__(self) -> None:
        for name in ("success", "collision", "unsafe_allow", "false_block"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"episode {name} must be boolean")
        if isinstance(self.robustness, bool) or not isinstance(self.robustness, (int, float)):
            raise ValueError("episode robustness must be numeric")
        if not math.isfinite(float(self.robustness)):
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
        if not isinstance(self.pair_id, str) or not 1 <= len(self.pair_id) <= 256:
            raise ValueError("paired episodes require identity commitments")
        for name in ("scenario_commitment", "seed_commitment"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"paired episode {name} must be a sha256 commitment")
        if not isinstance(self.baseline, EpisodeOutcome) or not isinstance(
            self.candidate, EpisodeOutcome
        ):
            raise ValueError("paired episodes require typed baseline and candidate outcomes")
        for name in (
            "physics_executed",
            "independently_verified",
            "strict_replay",
            "artifact_hash_valid",
            "data_quality_valid",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"paired episode {name} must be boolean")


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

    def __post_init__(self) -> None:
        rate_fields = (
            "baseline_success_rate",
            "candidate_success_rate",
            "baseline_collision_rate",
            "candidate_collision_rate",
            "baseline_unsafe_allow_rate",
            "candidate_unsafe_allow_rate",
            "baseline_false_block_rate",
            "candidate_false_block_rate",
            "success_mcnemar_pvalue",
        )
        for name in rate_fields:
            value = _finite_number(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "baseline_p05_robustness",
            "candidate_p05_robustness",
            "baseline_cvar05_robustness",
            "candidate_cvar05_robustness",
        ):
            _finite_number(getattr(self, name), name)
        if not isinstance(self.success_delta_ci95, tuple) or len(self.success_delta_ci95) != 2:
            raise ValueError("success_delta_ci95 must contain exactly two values")
        lower = _finite_number(self.success_delta_ci95[0], "success_delta_ci95 lower")
        upper = _finite_number(self.success_delta_ci95[1], "success_delta_ci95 upper")
        if not -1 <= lower <= upper <= 1:
            raise ValueError("success_delta_ci95 must be ordered within [-1, 1]")


@dataclass(frozen=True)
class EvidenceAttestation:
    physics_complete: bool
    independently_verified: bool
    scenario_seed_paired: bool
    strict_replay: bool
    artifact_hashes_valid: bool
    data_quality_valid: bool
    shards_complete: bool = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, bool):
                raise ValueError(f"evidence attestation {name} must be boolean")


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
    _verification_source: EvidenceVerificationSource | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _verification_seal: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not 1 <= len(self.task_id) <= 128:
            raise ValueError("evaluation task_id must contain 1..128 characters")
        if not _is_sha256(self.candidate_hash):
            raise ValueError("evaluation candidate_hash must be a sha256 digest")
        if not isinstance(self.partition, Partition):
            raise ValueError("evaluation partition must be a Partition")
        if (
            isinstance(self.paired_episodes, bool)
            or not isinstance(self.paired_episodes, int)
            or not 1 <= self.paired_episodes <= _MAX_EVALUATION_PAIRS
        ):
            raise ValueError(f"paired_episodes must be in [1, {_MAX_EVALUATION_PAIRS}]")
        if not isinstance(self.metrics, AggregateMetrics):
            raise ValueError("evaluation metrics must be AggregateMetrics")
        if not isinstance(self.attestation, EvidenceAttestation):
            raise ValueError("evaluation attestation must be EvidenceAttestation")
        if not _is_sha256(self.pair_set_commitment):
            raise ValueError("pair_set_commitment must be a sha256 digest")
        if not isinstance(self.human_involvement, HumanInvolvement):
            raise ValueError("human_involvement must be HumanInvolvement")
        if (
            not isinstance(self.evidence_refs, tuple)
            or len(self.evidence_refs) > _MAX_EVALUATION_PAIRS
            or any(not _is_sha256(item) for item in self.evidence_refs)
        ):
            raise ValueError("evidence_refs must contain bounded sha256 digests")
        if self.schema_version != "rosclaw.simforge.evaluation_bundle.v1":
            raise ValueError("unsupported evaluation bundle schema")
        if self._verification_source is not None and not isinstance(
            self._verification_source, EvidenceVerificationSource
        ):
            raise ValueError("invalid evaluation verification source")
        if self._verification_seal is not None and not _is_sha256(self._verification_seal):
            raise ValueError("invalid evaluation verification seal")

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
        if not pairs or len(pairs) > _MAX_EVALUATION_PAIRS:
            raise ValueError(f"evaluation bundles require 1..{_MAX_EVALUATION_PAIRS} pairs")
        if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
            raise ValueError("bootstrap_seed must be an integer")
        if not isinstance(partition, Partition):
            raise ValueError("evaluation partition must be a Partition")
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
        metrics = asdict(self.metrics)
        metrics["success_delta_ci95"] = list(self.metrics.success_delta_ci95)
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "candidate_hash": self.candidate_hash,
            "partition": self.partition.value,
            "paired_episodes": self.paired_episodes,
            "aggregate_metrics": metrics,
            "attestation": asdict(self.attestation),
            "pair_set_commitment": self.pair_set_commitment,
            "human_involvement": asdict(self.human_involvement),
            "evidence_refs": list(self.evidence_refs),
            "complete": all(asdict(self.attestation).values()),
        }


@dataclass(frozen=True)
class StressEvidence:
    task_id: str
    candidate_hash: str
    worlds: int
    complete: bool
    critical_backend_disagreements: int
    scale_curve_commitment: str
    _verification_seal: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not 1 <= len(self.task_id) <= 128:
            raise ValueError("stress task_id must contain 1..128 characters")
        if not _is_sha256(self.candidate_hash):
            raise ValueError("stress candidate_hash must be a sha256 digest")
        if (
            isinstance(self.worlds, bool)
            or not isinstance(self.worlds, int)
            or not 1 <= self.worlds <= 100_000
        ):
            raise ValueError("stress worlds must be in [1, 100000]")
        if not isinstance(self.complete, bool):
            raise ValueError("stress complete must be boolean")
        if (
            isinstance(self.critical_backend_disagreements, bool)
            or not isinstance(self.critical_backend_disagreements, int)
            or not 0 <= self.critical_backend_disagreements <= self.worlds
        ):
            raise ValueError("stress disagreement count is invalid")
        if not _is_sha256(self.scale_curve_commitment):
            raise ValueError("scale curve commitment must be a sha256 digest")
        if self._verification_seal is not None and not _is_sha256(self._verification_seal):
            raise ValueError("invalid stress verification seal")


def _attest_evaluation_bundle(
    bundle: EvaluationBundle,
    source: EvidenceVerificationSource,
) -> EvaluationBundle:
    """Bind verified evidence to this process after its source-specific checks pass."""

    unsigned = replace(
        bundle,
        _verification_source=source,
        _verification_seal=None,
    )
    return replace(
        unsigned,
        _verification_seal=_seal(
            f"evaluation:{source.value}",
            unsigned.aggregate_dict(),
        ),
    )


def _evaluation_bundle_verified(
    bundle: EvaluationBundle,
    source: EvidenceVerificationSource,
) -> bool:
    if bundle._verification_source is not source or bundle._verification_seal is None:
        return False
    return hmac.compare_digest(
        bundle._verification_seal,
        _seal(f"evaluation:{source.value}", bundle.aggregate_dict()),
    )


def _attest_stress_evidence(evidence: StressEvidence) -> StressEvidence:
    unsigned = replace(evidence, _verification_seal=None)
    return replace(
        unsigned,
        _verification_seal=_seal(
            f"stress:{EvidenceVerificationSource.SIGNED_SCALE_CURVE.value}",
            _stress_payload(unsigned),
        ),
    )


def _stress_evidence_verified(evidence: StressEvidence) -> bool:
    if evidence._verification_seal is None:
        return False
    return hmac.compare_digest(
        evidence._verification_seal,
        _seal(
            f"stress:{EvidenceVerificationSource.SIGNED_SCALE_CURVE.value}",
            _stress_payload(evidence),
        ),
    )


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
    lower = min(baseline_only, candidate_only)
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2)
        for index in range(lower + 1)
    ]
    maximum = max(log_probabilities)
    log_tail = maximum + math.log(sum(math.exp(value - maximum) for value in log_probabilities))
    return min(1.0, 2.0 * math.exp(log_tail))


def paired_bootstrap_ci(
    deltas: list[float], *, seed: int, samples: int = 2000
) -> tuple[float, float]:
    if (
        not deltas
        or len(deltas) > _MAX_EVALUATION_PAIRS
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or not 100 <= samples <= _MAX_BOOTSTRAP_SAMPLES
        or len(deltas) * samples > 20_000_000
    ):
        raise ValueError(
            "paired bootstrap requires bounded data, an integer seed, and at most "
            "20000000 resampled values"
        )
    normalized = [_finite_number(value, "paired bootstrap delta") for value in deltas]
    rng = random.Random(seed)
    count = len(normalized)
    means = sorted(
        sum(normalized[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples)
    )
    return (
        means[math.floor(0.025 * (samples - 1))],
        means[math.ceil(0.975 * (samples - 1))],
    )


def _rate(values: Any) -> float:
    normalized = list(values)
    if not normalized or any(not isinstance(value, bool) for value in normalized):
        raise ValueError("rate inputs must be non-empty booleans")
    return sum(normalized) / len(normalized)


def _stress_payload(evidence: StressEvidence) -> dict[str, Any]:
    return {
        "task_id": evidence.task_id,
        "candidate_hash": evidence.candidate_hash,
        "worlds": evidence.worlds,
        "complete": evidence.complete,
        "critical_backend_disagreements": evidence.critical_backend_disagreements,
        "scale_curve_commitment": evidence.scale_curve_commitment,
    }


def _seal(domain: str, value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    digest = hmac.new(
        _PROCESS_EVIDENCE_KEY,
        domain.encode() + b"\0" + payload,
        hashlib.sha256,
    ).hexdigest()
    return "sha256:" + digest


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


__all__ = [
    "AggregateMetrics",
    "EvidenceVerificationSource",
    "EpisodeOutcome",
    "EvaluationBundle",
    "EvidenceAttestation",
    "PairedEpisode",
    "StressEvidence",
    "exact_mcnemar",
    "paired_bootstrap_ci",
]
