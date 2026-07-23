"""Safety and evidence gates for simulation-only candidate promotion."""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rosclaw.sandbox.evidence import (
    SimulationEvidenceVerification,
    verify_promotion_receipt,
)

logger = logging.getLogger("rosclaw.auto.promotion.gate")


@dataclass
class GateResult:
    """Result of running a promotion gate."""

    passed: bool = False
    decision: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    next_level: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "checks": self.checks,
            "next_level": self.next_level,
            "reason": self.reason,
        }


class PromotionGate:
    """Promote a paired physics experiment no further than simulation."""

    LEVEL_ORDER = ["baseline", "sim", "sandbox", "real_candidate", "real"]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        receipt_verifier: Callable[[dict[str, Any]], SimulationEvidenceVerification] | None = None,
    ) -> None:
        self.config = config or {}
        self.min_success_delta = float(self.config.get("min_success_improvement", 0.05))
        self.max_collision_increase = float(self.config.get("max_collision_increase", 0.0))
        self.max_success_std = float(self.config.get("max_success_std", 0.08))
        self.min_seeds = int(self.config.get("min_seeds", 2))
        self.max_risk_score = float(self.config.get("max_risk_score", 0.2))
        if self.min_seeds < 2:
            raise ValueError("PromotionGate.min_seeds must be at least 2")
        for name, value in (
            ("min_success_improvement", self.min_success_delta),
            ("max_collision_increase", self.max_collision_increase),
            ("max_success_std", self.max_success_std),
            ("max_risk_score", self.max_risk_score),
        ):
            if not math.isfinite(value):
                raise ValueError(f"PromotionGate.{name} must be finite")
        self._receipt_verifier = receipt_verifier or verify_promotion_receipt

    @staticmethod
    def _number(metrics: dict[str, Any], name: str) -> float | None:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        normalized = float(value)
        return normalized if math.isfinite(normalized) else None

    @staticmethod
    def _outcome(receipt: dict[str, Any]) -> dict[str, float]:
        return {
            "success_rate": 1.0 if receipt.get("is_safe") is True else 0.0,
            "collision_rate": 1.0 if receipt.get("collision_pairs") else 0.0,
        }

    def evaluate(
        self,
        baseline_metrics: dict[str, Any],
        candidate_metrics: dict[str, Any],
        current_level: str = "baseline",
        per_seed: dict[Any, Any] | None = None,
        sandbox_risk_score: float | None = None,
        simulation_receipts: list[dict[str, Any]] | None = None,
        regression_results: dict[str, Any] | None = None,
    ) -> GateResult:
        checks: list[dict[str, Any]] = []
        passed = True

        level_known = current_level in self.LEVEL_ORDER
        next_level = self._next_level(current_level) if level_known else current_level
        simulation_ceiling = level_known and current_level == "baseline" and next_level == "sim"

        baseline_success = self._number(baseline_metrics, "success_rate")
        candidate_success = self._number(candidate_metrics, "success_rate")
        metrics_present = baseline_success is not None and candidate_success is not None
        success_delta = candidate_success - baseline_success if metrics_present else -math.inf
        metric_improvement = metrics_present and success_delta >= self.min_success_delta
        checks.append(
            {
                "name": "metric_improvement",
                "passed": metric_improvement,
                "missing_evidence": not metrics_present,
                "detail": (
                    f"success_rate_delta={success_delta:.3f} (need >= {self.min_success_delta})"
                    if metrics_present
                    else "finite baseline and candidate success_rate values are required"
                ),
            }
        )
        passed = passed and metric_improvement

        baseline_collision = self._number(baseline_metrics, "collision_rate")
        candidate_collision = self._number(candidate_metrics, "collision_rate")
        collision_present = baseline_collision is not None and candidate_collision is not None
        collision_delta = (
            candidate_collision - baseline_collision if collision_present else math.inf
        )
        safety_ok = collision_present and collision_delta <= self.max_collision_increase + 1e-12
        checks.append(
            {
                "name": "safety_non_regression",
                "passed": safety_ok,
                "missing_evidence": not collision_present,
                "detail": (
                    f"collision_rate_delta={collision_delta:.3f} "
                    f"(need <= {self.max_collision_increase})"
                    if collision_present
                    else "finite baseline and candidate collision_rate values are required"
                ),
            }
        )
        passed = passed and safety_ok

        supplied_receipts = simulation_receipts or []
        verified_receipts: list[dict[str, Any]] = []
        receipt_errors: list[str] = []
        for index, receipt in enumerate(supplied_receipts):
            if not isinstance(receipt, dict):
                receipt_errors.append(f"receipt[{index}]:not_mapping")
                continue
            try:
                verification = self._receipt_verifier(receipt)
            except Exception as exc:  # noqa: BLE001 - promotion must fail closed
                receipt_errors.append(f"receipt[{index}]:verifier_error:{type(exc).__name__}")
                continue
            if verification.verified:
                verified_receipts.append(receipt)
            else:
                detail = ",".join(verification.errors or verification.replay.mismatches)
                receipt_errors.append(f"receipt[{index}]:{detail or verification.replay.reason}")

        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        duplicate_variants = False
        for receipt in verified_receipts:
            seed_key = str(receipt.get("seed"))
            variant = str(receipt.get("evaluation_variant"))
            pair = grouped.setdefault(seed_key, {})
            if variant in pair:
                duplicate_variants = True
            pair[variant] = receipt
        physics_ok = (
            bool(supplied_receipts) and not receipt_errors and len(grouped) >= self.min_seeds
        )
        checks.append(
            {
                "name": "physics_evidence",
                "passed": physics_ok,
                "missing_evidence": not supplied_receipts or len(grouped) < self.min_seeds,
                "detail": (
                    f"strictly_replayed_receipts={len(verified_receipts)}, "
                    f"distinct_seeds={len(grouped)}, errors={receipt_errors}"
                ),
            }
        )
        passed = passed and physics_ok

        supplied_per_seed = {str(key): value for key, value in (per_seed or {}).items()}
        paired = (
            physics_ok
            and not duplicate_variants
            and set(supplied_per_seed) == set(grouped)
            and all(set(pair) == {"baseline", "candidate"} for pair in grouped.values())
        )
        initial_states: set[str] = set()
        pairing_consistent = paired
        derived_per_seed: dict[str, dict[str, dict[str, float]]] = {}
        if paired:
            for seed, pair in grouped.items():
                baseline_receipt = pair["baseline"]
                candidate_receipt = pair["candidate"]
                baseline_randomization = baseline_receipt.get("randomization") or {}
                candidate_randomization = candidate_receipt.get("randomization") or {}
                baseline_request = baseline_receipt.get("request")
                candidate_request = candidate_receipt.get("request")
                initial_state = str(baseline_randomization.get("initial_state_hash") or "")
                paired_request_contract = False
                if isinstance(baseline_request, dict) and isinstance(candidate_request, dict):
                    baseline_contract = dict(baseline_request)
                    candidate_contract = dict(candidate_request)
                    baseline_contract.pop("trajectory", None)
                    candidate_contract.pop("trajectory", None)
                    paired_request_contract = baseline_contract == candidate_contract
                if not (
                    baseline_receipt.get("pair_id") == candidate_receipt.get("pair_id")
                    and initial_state
                    and initial_state == candidate_randomization.get("initial_state_hash")
                    and baseline_randomization.get("parameter_hash")
                    == candidate_randomization.get("parameter_hash")
                    and baseline_receipt.get("model_hash") == candidate_receipt.get("model_hash")
                    and baseline_receipt.get("world_asset_hash")
                    == candidate_receipt.get("world_asset_hash")
                    and baseline_receipt.get("backend") == candidate_receipt.get("backend")
                    and paired_request_contract
                ):
                    pairing_consistent = False
                initial_states.add(initial_state)
                derived_per_seed[seed] = {
                    "baseline": self._outcome(baseline_receipt),
                    "candidate": self._outcome(candidate_receipt),
                }
                supplied_pair = supplied_per_seed.get(seed)
                if not isinstance(supplied_pair, dict):
                    pairing_consistent = False
                    continue
                for variant in ("baseline", "candidate"):
                    supplied_outcome = supplied_pair.get(variant)
                    if not isinstance(supplied_outcome, dict):
                        pairing_consistent = False
                        continue
                    for metric_name, derived_value in derived_per_seed[seed][variant].items():
                        supplied_value = self._number(supplied_outcome, metric_name)
                        if supplied_value is None or not math.isclose(
                            supplied_value, derived_value, abs_tol=1e-12
                        ):
                            pairing_consistent = False
        seed_effective = pairing_consistent and len(initial_states) == len(grouped)
        paired_ok = paired and pairing_consistent and seed_effective
        checks.append(
            {
                "name": "paired_seed_evidence",
                "passed": paired_ok,
                "missing_evidence": not paired,
                "detail": (
                    f"paired_seeds={len(grouped)}, unique_initial_states={len(initial_states)}"
                    if paired
                    else "exact baseline/candidate receipt pairs are required for every metric seed"
                ),
            }
        )
        passed = passed and paired_ok

        derived_baseline: dict[str, float] = {}
        derived_candidate: dict[str, float] = {}
        metrics_consistent = paired_ok
        if paired_ok:
            for metric_name in ("success_rate", "collision_rate"):
                derived_baseline[metric_name] = statistics.mean(
                    item["baseline"][metric_name] for item in derived_per_seed.values()
                )
                derived_candidate[metric_name] = statistics.mean(
                    item["candidate"][metric_name] for item in derived_per_seed.values()
                )
                supplied_baseline = self._number(baseline_metrics, metric_name)
                supplied_candidate = self._number(candidate_metrics, metric_name)
                metrics_consistent = bool(
                    metrics_consistent
                    and supplied_baseline is not None
                    and supplied_candidate is not None
                    and math.isclose(
                        supplied_baseline,
                        derived_baseline[metric_name],
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        supplied_candidate,
                        derived_candidate[metric_name],
                        abs_tol=1e-12,
                    )
                )
        checks.append(
            {
                "name": "metric_provenance",
                "passed": metrics_consistent,
                "missing_evidence": not paired_ok,
                "detail": "aggregate metrics match replayed per-seed receipts",
            }
        )
        passed = passed and metrics_consistent

        candidate_seed_success = [
            item["candidate"]["success_rate"] for item in derived_per_seed.values()
        ]
        success_std = (
            statistics.stdev(candidate_seed_success)
            if len(candidate_seed_success) > 1
            else math.inf
        )
        robustness_ok = paired_ok and success_std <= self.max_success_std
        checks.append(
            {
                "name": "robustness",
                "passed": robustness_ok,
                "missing_evidence": len(candidate_seed_success) < 2,
                "detail": f"success_rate_std={success_std:.3f} (need <= {self.max_success_std})",
            }
        )
        passed = passed and robustness_ok

        seed_confirmation = paired_ok and len(grouped) >= self.min_seeds
        checks.append(
            {
                "name": "second_seed_confirmation",
                "passed": seed_confirmation,
                "missing_evidence": not seed_confirmation,
                "detail": f"effective_seeds={len(initial_states)} (need >= {self.min_seeds})",
            }
        )
        passed = passed and seed_confirmation

        risk_valid = (
            not isinstance(sandbox_risk_score, bool)
            and isinstance(sandbox_risk_score, (int, float))
            and math.isfinite(float(sandbox_risk_score))
        )
        risk_ok = risk_valid and 0.0 <= float(sandbox_risk_score) < self.max_risk_score
        checks.append(
            {
                "name": "sandbox_clearance",
                "passed": risk_ok,
                "missing_evidence": not risk_valid,
                "detail": f"risk_score={sandbox_risk_score!r} (need >= 0 and < {self.max_risk_score})",
            }
        )
        passed = passed and risk_ok

        expected_critical: list[str] = []
        if paired_ok and derived_candidate["collision_rate"] > derived_baseline["collision_rate"]:
            expected_critical.append("collision_rate_regression")
        expected_episodes = len(verified_receipts)
        regression_ok = bool(
            isinstance(regression_results, dict)
            and regression_results.get("suite") == "physics_counterexample_v1"
            and regression_results.get("episodes") == expected_episodes
            and regression_results.get("critical_regressions") == expected_critical
            and regression_results.get("passed") is (not expected_critical)
        )
        checks.append(
            {
                "name": "regression_suite",
                "passed": regression_ok,
                "missing_evidence": not isinstance(regression_results, dict),
                "detail": f"episodes={expected_episodes}, critical={expected_critical}",
            }
        )
        passed = passed and regression_ok

        checks.append(
            {
                "name": "simulation_promotion_ceiling",
                "passed": simulation_ceiling,
                "missing_evidence": not level_known,
                "detail": "simulation evidence may only promote baseline to sim",
            }
        )
        passed = passed and simulation_ceiling

        if passed:
            decision = "promote_to_sim"
            reason = "All replayed physics gates passed; promote baseline to simulation"
        elif not simulation_ceiling and level_known:
            decision = "need_hardware_evidence"
            reason = "Simulation evidence cannot promote beyond the simulation level"
        elif (metrics_present and success_delta < 0.0) or (collision_present and not safety_ok):
            decision = "reject"
            reason = "Candidate task or safety metrics regressed"
        elif any(check.get("missing_evidence") for check in checks):
            decision = "need_more_evidence"
            reason = "Required replayed physics, paired-seed, or regression evidence is missing"
        else:
            decision = "reject"
            reason = "Candidate failed one or more completed promotion gates"

        logger.info("PromotionGate: %s -> %s", decision, reason)
        return GateResult(
            passed=passed,
            decision=decision,
            checks=checks,
            next_level="sim" if passed else current_level,
            reason=reason,
        )

    def _next_level(self, current: str) -> str:
        if current not in self.LEVEL_ORDER:
            raise ValueError(f"Unknown promotion level: {current}")
        index = self.LEVEL_ORDER.index(current)
        return self.LEVEL_ORDER[min(index + 1, len(self.LEVEL_ORDER) - 1)]


__all__ = ["GateResult", "PromotionGate"]
