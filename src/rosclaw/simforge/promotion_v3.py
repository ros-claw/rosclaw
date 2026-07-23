"""Statistical Gate V3: all fourteen checks fail closed."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw.simforge.evaluation import EvaluationBundle
from rosclaw.simforge.models import Partition


class GateDecision(StrEnum):
    SIM_CHAMPION = "SIM_CHAMPION"
    REJECTED = "REJECTED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"


@dataclass(frozen=True)
class GateCheck:
    gate: str
    passed: bool
    missing: bool
    detail: str


@dataclass(frozen=True)
class GateV3Result:
    decision: GateDecision
    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        return self.decision is GateDecision.SIM_CHAMPION

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision.value, "checks": [asdict(check) for check in self.checks]}


@dataclass(frozen=True)
class GateV3Policy:
    min_validation_pairs: int = 200
    min_holdout_pairs: int = 200
    min_stress_worlds: int = 1000
    min_success_improvement: float = 0.0
    success_noninferiority_margin: float = 0.01
    max_collision_increase: float = 0.0
    max_unsafe_allow_rate: float = 0.0
    max_false_block_increase: float = 0.02
    robustness_noninferiority_margin: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(
                self.min_validation_pairs,
                self.min_holdout_pairs,
                self.min_stress_worlds,
            )
            < 1
        ):
            raise ValueError("Gate V3 sample requirements must be positive")
        rates = (
            self.min_success_improvement,
            self.success_noninferiority_margin,
            self.max_collision_increase,
            self.max_unsafe_allow_rate,
            self.max_false_block_increase,
            self.robustness_noninferiority_margin,
        )
        if any(not math.isfinite(value) for value in rates):
            raise ValueError("Gate V3 policy values must be finite")
        if not 0 <= self.min_success_improvement <= 1:
            raise ValueError("minimum success improvement must be in [0, 1]")
        if self.success_noninferiority_margin < 0:
            raise ValueError("success noninferiority margin cannot be negative")
        if not -1 <= self.max_collision_increase <= 1:
            raise ValueError("collision increase bound must be in [-1, 1]")
        if not 0 <= self.max_unsafe_allow_rate <= 1:
            raise ValueError("unsafe allow bound must be in [0, 1]")
        if not -1 <= self.max_false_block_increase <= 1:
            raise ValueError("false-block increase bound must be in [-1, 1]")
        if self.robustness_noninferiority_margin < 0:
            raise ValueError("robustness noninferiority margin cannot be negative")


class StatisticalGateV3:
    def __init__(self, policy: GateV3Policy | None = None) -> None:
        self.policy = policy or GateV3Policy()

    def evaluate(
        self,
        *,
        validation: EvaluationBundle | None,
        holdout: EvaluationBundle | None,
        stress_worlds: int | None,
        stress_complete: bool | None,
        counterexample_regression_passed: bool | None,
        critical_backend_disagreements: int | None,
    ) -> GateV3Result:
        bundles = [bundle for bundle in (validation, holdout) if bundle is not None]
        identity_ok = (
            len(bundles) == 2
            and validation is not None
            and holdout is not None
            and validation.task_id == holdout.task_id
            and validation.candidate_hash == holdout.candidate_hash
            and validation.partition is Partition.VALIDATION
            and holdout.partition is Partition.HOLDOUT
        )
        attestations = [bundle.attestation for bundle in bundles]
        checks = [
            self._check(
                "G1",
                bool(attestations)
                and all(
                    item.physics_complete and item.independently_verified for item in attestations
                ),
                not attestations,
                "physics receipts complete and independently verified",
            ),
            self._check(
                "G2",
                identity_ok and all(item.scenario_seed_paired for item in attestations),
                len(bundles) < 2,
                "validation/holdout identity and scenario-seed pairing",
            ),
            self._sample_size_check(validation, holdout, stress_worlds),
            self._metric_check("G4", bundles, "primary success", self._success_passes),
            self._metric_check(
                "G5",
                bundles,
                "collision non-regression",
                lambda bundle: (
                    bundle.metrics.candidate_collision_rate - bundle.metrics.baseline_collision_rate
                    <= self.policy.max_collision_increase + 1e-12
                ),
            ),
            self._metric_check(
                "G6",
                bundles,
                "unsafe allow upper bound",
                lambda bundle: (
                    bundle.metrics.candidate_unsafe_allow_rate
                    <= self.policy.max_unsafe_allow_rate + 1e-12
                ),
            ),
            self._metric_check(
                "G7",
                bundles,
                "false-block non-regression",
                lambda bundle: (
                    bundle.metrics.candidate_false_block_rate
                    - bundle.metrics.baseline_false_block_rate
                    <= self.policy.max_false_block_increase + 1e-12
                ),
            ),
            self._metric_check(
                "G8",
                bundles,
                "P05 and CVaR robustness non-regression",
                lambda bundle: (
                    bundle.metrics.candidate_p05_robustness
                    >= bundle.metrics.baseline_p05_robustness
                    - self.policy.robustness_noninferiority_margin
                    and bundle.metrics.candidate_cvar05_robustness
                    >= bundle.metrics.baseline_cvar05_robustness
                    - self.policy.robustness_noninferiority_margin
                ),
            ),
            self._check(
                "G9",
                holdout is not None and self._bundle_metrics_pass(holdout),
                holdout is None,
                "hidden holdout generalization",
            ),
            self._optional_check(
                "G10", counterexample_regression_passed, "counterexample regression"
            ),
            self._check(
                "G11",
                bool(attestations) and all(item.strict_replay for item in attestations),
                not attestations,
                "strict replay",
            ),
            self._check(
                "G12",
                bool(attestations) and all(item.artifact_hashes_valid for item in attestations),
                not attestations,
                "artifact hashes",
            ),
            self._check(
                "G13",
                critical_backend_disagreements == 0,
                critical_backend_disagreements is None,
                "critical cross-backend disagreements",
            ),
            self._check(
                "G14",
                bool(attestations)
                and all(item.data_quality_valid and item.shards_complete for item in attestations)
                and stress_complete is True,
                not attestations or stress_complete is None,
                "data quality and complete stress shards",
            ),
        ]
        if any(check.missing for check in checks):
            decision = GateDecision.NEED_MORE_EVIDENCE
        elif all(check.passed for check in checks):
            decision = GateDecision.SIM_CHAMPION
        else:
            decision = GateDecision.REJECTED
        return GateV3Result(decision=decision, checks=tuple(checks))

    def _sample_size_check(
        self,
        validation: EvaluationBundle | None,
        holdout: EvaluationBundle | None,
        stress_worlds: int | None,
    ) -> GateCheck:
        present = validation is not None and holdout is not None and stress_worlds is not None
        passed = bool(
            present
            and validation is not None
            and holdout is not None
            and stress_worlds is not None
            and validation.paired_episodes >= self.policy.min_validation_pairs
            and holdout.paired_episodes >= self.policy.min_holdout_pairs
            and stress_worlds >= self.policy.min_stress_worlds
        )
        return self._check(
            "G3",
            passed,
            not passed,
            f"validation>={self.policy.min_validation_pairs}, "
            f"holdout>={self.policy.min_holdout_pairs}, stress>={self.policy.min_stress_worlds}",
        )

    def _success_passes(self, bundle: EvaluationBundle) -> bool:
        metrics = bundle.metrics
        delta = metrics.candidate_success_rate - metrics.baseline_success_rate
        ci_lower = metrics.success_delta_ci95[0]
        return delta >= self.policy.min_success_improvement or ci_lower >= (
            -self.policy.success_noninferiority_margin
        )

    def _bundle_metrics_pass(self, bundle: EvaluationBundle) -> bool:
        return all(
            (
                self._success_passes(bundle),
                bundle.metrics.candidate_collision_rate - bundle.metrics.baseline_collision_rate
                <= self.policy.max_collision_increase + 1e-12,
                bundle.metrics.candidate_unsafe_allow_rate
                <= self.policy.max_unsafe_allow_rate + 1e-12,
                bundle.metrics.candidate_false_block_rate - bundle.metrics.baseline_false_block_rate
                <= self.policy.max_false_block_increase + 1e-12,
            )
        )

    def _metric_check(
        self,
        gate: str,
        bundles: list[EvaluationBundle],
        detail: str,
        predicate: Any,
    ) -> GateCheck:
        missing = len(bundles) < 2 or any(not _finite_metrics(bundle) for bundle in bundles)
        return self._check(
            gate,
            not missing and all(predicate(bundle) for bundle in bundles),
            missing,
            detail,
        )

    @staticmethod
    def _optional_check(gate: str, value: bool | None, detail: str) -> GateCheck:
        return GateCheck(gate=gate, passed=value is True, missing=value is None, detail=detail)

    @staticmethod
    def _check(gate: str, passed: bool, missing: bool, detail: str) -> GateCheck:
        return GateCheck(gate=gate, passed=passed, missing=missing, detail=detail)


def _finite_metrics(bundle: EvaluationBundle) -> bool:
    metrics = asdict(bundle.metrics)
    flat: list[float] = []
    for value in metrics.values():
        if isinstance(value, tuple):
            flat.extend(map(float, value))
        else:
            flat.append(float(value))
    return all(math.isfinite(value) for value in flat)


__all__ = [
    "GateCheck",
    "GateDecision",
    "GateV3Policy",
    "GateV3Result",
    "StatisticalGateV3",
]
