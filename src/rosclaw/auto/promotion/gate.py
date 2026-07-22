"""PromotionGate — safety-gated skill champion promotion."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("rosclaw.auto.promotion.gate")


@dataclass
class GateResult:
    """Result of running a promotion gate."""

    passed: bool = False
    decision: str = ""
    checks: list[dict] = field(default_factory=list)
    next_level: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "checks": self.checks,
            "next_level": self.next_level,
            "reason": self.reason,
        }


class PromotionGate:
    """Evaluate whether a candidate skill can be promoted.

    Gates (in order):
    1. metric_improvement  — success_rate_delta >= threshold
    2. safety_non_regression — collision_rate_delta <= 0
    3. physics_evidence — verified SimulationReceipts exist
    4. paired_seeds — baseline/candidate data exist on every seed
    5. robustness — success_rate_std <= threshold
    6. second_seed_confirmation — confirmed on >= 2 seeds
    7. sandbox_clearance — risk_score < threshold
    8. regression_suite — caller supplies a completed regression result
    """

    LEVEL_ORDER = ["baseline", "sim", "sandbox", "real_candidate", "real"]

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.min_success_delta = self.config.get("min_success_improvement", 0.05)
        self.max_collision_increase = self.config.get("max_collision_increase", 0.00)
        self.max_success_std = self.config.get("max_success_std", 0.08)
        self.min_seeds = self.config.get("min_seeds", 2)
        self.max_risk_score = self.config.get("max_risk_score", 0.2)

    def evaluate(
        self,
        baseline_metrics: dict,
        candidate_metrics: dict,
        current_level: str = "baseline",
        per_seed: dict | None = None,
        sandbox_risk_score: float = 0.0,
        simulation_receipts: list[dict] | None = None,
        regression_results: dict | None = None,
    ) -> GateResult:
        """Run all promotion gates and return decision."""
        checks = []
        passed = True
        next_level = self._next_level(current_level)

        # Gate 1: metric_improvement
        sr_delta = candidate_metrics.get("success_rate", 0) - baseline_metrics.get(
            "success_rate", 0
        )
        g1 = sr_delta >= self.min_success_delta
        checks.append(
            {
                "name": "metric_improvement",
                "passed": g1,
                "detail": f"success_rate_delta={sr_delta:.3f} (need >= {self.min_success_delta})",
            }
        )
        if not g1:
            passed = False

        # Gate 2: safety_non_regression
        col_delta = candidate_metrics.get("collision_rate", 0) - baseline_metrics.get(
            "collision_rate", 0
        )
        g2 = col_delta <= self.max_collision_increase + 0.001
        checks.append(
            {
                "name": "safety_non_regression",
                "passed": g2,
                "detail": f"collision_rate_delta={col_delta:.3f} (need <= {self.max_collision_increase})",
            }
        )
        if not g2:
            passed = False

        # Gate 3: promotion-grade physics receipts must exist. Fixture/mock
        # output can never satisfy this gate.
        valid_receipts = [
            receipt
            for receipt in (simulation_receipts or [])
            if self._valid_simulation_receipt(receipt)
        ]
        receipt_seeds = {
            (receipt.get("simulation_result") or receipt).get("seed") for receipt in valid_receipts
        }
        receipt_seeds.discard(None)
        g3 = len(receipt_seeds) >= self.min_seeds
        checks.append(
            {
                "name": "physics_evidence",
                "passed": g3,
                "missing_evidence": not g3,
                "detail": (
                    f"promotion-grade receipts={len(valid_receipts)}, "
                    f"distinct_seeds={len(receipt_seeds)} (need >= {self.min_seeds})"
                ),
            }
        )
        if not g3:
            passed = False

        # Gate 4: every seed must contain a paired baseline and candidate.
        paired = bool(per_seed) and all(
            isinstance(result, dict)
            and isinstance(result.get("baseline"), dict)
            and isinstance(result.get("candidate"), dict)
            for result in (per_seed or {}).values()
        )
        checks.append(
            {
                "name": "paired_seed_evidence",
                "passed": paired,
                "missing_evidence": not paired,
                "detail": (
                    f"paired_seeds={len(per_seed or {})}"
                    if paired
                    else "baseline/candidate pairs missing"
                ),
            }
        )
        if not paired:
            passed = False

        # Gate 5: robustness (std across paired seeds).
        if paired and len(per_seed or {}) > 1:
            srs = [
                s["candidate"]["success_rate"]
                for s in (per_seed or {}).values()
                if "candidate" in s and "success_rate" in s.get("candidate", {})
            ]
            if srs:
                import statistics

                sr_std = statistics.stdev(srs) if len(srs) > 1 else 0.0
                g3 = sr_std <= self.max_success_std
                checks.append(
                    {
                        "name": "robustness",
                        "passed": g3,
                        "detail": f"success_rate_std={sr_std:.3f} (need <= {self.max_success_std})",
                    }
                )
                if not g3:
                    passed = False
            else:
                checks.append(
                    {
                        "name": "robustness",
                        "passed": False,
                        "missing_evidence": True,
                        "detail": "candidate success_rate missing from per-seed data",
                    }
                )
                passed = False
        else:
            checks.append(
                {
                    "name": "robustness",
                    "passed": False,
                    "missing_evidence": True,
                    "detail": "at least two paired seeds are required",
                }
            )
            passed = False

        # Gate 6: second-seed confirmation is mandatory.
        seed_count = len(per_seed or {})
        g6 = seed_count >= self.min_seeds
        checks.append(
            {
                "name": "second_seed_confirmation",
                "passed": g6,
                "missing_evidence": not g6,
                "detail": f"seeds={seed_count} (need >= {self.min_seeds})",
            }
        )
        if not g6:
            passed = False

        # Gate 7: sandbox_clearance
        g7 = sandbox_risk_score < self.max_risk_score
        checks.append(
            {
                "name": "sandbox_clearance",
                "passed": g7,
                "detail": f"risk_score={sandbox_risk_score:.3f} (need < {self.max_risk_score})",
            }
        )
        if not g7:
            passed = False

        # Gate 8: regression_suite must be supplied by a real runner.
        regression_present = isinstance(regression_results, dict)
        regression_passed = bool(
            regression_present
            and regression_results.get("passed") is True
            and not regression_results.get("critical_regressions")
        )
        checks.append(
            {
                "name": "regression_suite",
                "passed": regression_passed,
                "missing_evidence": not regression_present,
                "detail": (
                    "completed without critical regression"
                    if regression_passed
                    else "verified regression result is missing or failed"
                ),
            }
        )
        if not regression_passed:
            passed = False

        if passed:
            decision = f"promote_to_{next_level}"
            reason = f"All gates passed; promote from {current_level} to {next_level}"
        else:
            missing_evidence = any(check.get("missing_evidence") for check in checks)
            if sr_delta < 0 or not g2:
                decision = "reject"
                reason = "Candidate task or safety metrics regressed; reject"
            elif missing_evidence:
                decision = "need_more_evidence"
                reason = "Required physics, paired-seed, or regression evidence is missing"
            else:
                decision = "reject"
                reason = "Candidate failed one or more completed promotion gates"

        logger.info("PromotionGate: %s -> %s", decision, reason)
        return GateResult(
            passed=passed,
            decision=decision,
            checks=checks,
            next_level=next_level if passed else current_level,
            reason=reason,
        )

    @staticmethod
    def _valid_simulation_receipt(receipt: dict) -> bool:
        if receipt.get("schema_version") == "rosclaw.simulation_receipt.v1":
            quality = receipt.get("data_quality") or {}
            replay = receipt.get("replay_report") or {}
            return bool(
                receipt.get("evidence_domain") == "SIMULATION"
                and receipt.get("physics_executed") is True
                and receipt.get("body_snapshot_hash")
                and receipt.get("model_hash")
                and receipt.get("action_hash")
                and receipt.get("artifact_hashes")
                and quality.get("artifact_hash_valid") is True
                and quality.get("body_snapshot_match") is True
                and replay.get("verified") is True
            )
        simulation = receipt.get("simulation_result") or {}
        dispatch = receipt.get("dispatch_result") or {}
        quality = (receipt.get("verification_result") or {}).get("data_quality") or {}
        hashes = simulation.get("artifact_hashes")
        return bool(
            receipt.get("execution_mode", receipt.get("mode")) == "SIMULATION"
            and receipt.get("evidence_domain") == "SIMULATION"
            and simulation.get("has_physics") is True
            and simulation.get("physics_executed") is True
            and dispatch.get("physics_executed") is True
            and receipt.get("body_snapshot_hash")
            and simulation.get("model_hash")
            and simulation.get("action_hash")
            and isinstance(hashes, dict)
            and hashes
            and quality.get("artifact_hash_valid") is True
            and quality.get("body_snapshot_match") is True
        )

    def _next_level(self, current: str) -> str:
        idx = self.LEVEL_ORDER.index(current) if current in self.LEVEL_ORDER else 0
        return self.LEVEL_ORDER[min(idx + 1, len(self.LEVEL_ORDER) - 1)]
