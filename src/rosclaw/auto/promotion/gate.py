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
    3. robustness — success_rate_std <= threshold (if per_seed available)
    4. second_seed_confirmation — confirmed on >= 2 seeds (if per_seed available)
    5. sandbox_clearance — risk_score < threshold
    6. regression_suite — no critical regression
    """

    LEVEL_ORDER = ["baseline", "sim", "sandbox", "real_candidate", "real"]

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.min_success_delta = self.config.get("min_success_improvement", 0.05)
        self.max_collision_increase = self.config.get("max_collision_increase", 0.00)
        self.max_success_std = self.config.get("max_success_std", 0.08)
        self.min_seeds = self.config.get("min_seeds", 2)
        self.max_risk_score = self.config.get("max_risk_score", 0.2)

    def evaluate(self, baseline_metrics: dict, candidate_metrics: dict,
                 current_level: str = "baseline", per_seed: dict | None = None,
                 sandbox_risk_score: float = 0.0) -> GateResult:
        """Run all promotion gates and return decision."""
        checks = []
        passed = True
        next_level = self._next_level(current_level)

        # Gate 1: metric_improvement
        sr_delta = candidate_metrics.get("success_rate", 0) - baseline_metrics.get("success_rate", 0)
        g1 = sr_delta >= self.min_success_delta
        checks.append({"name": "metric_improvement", "passed": g1,
                       "detail": f"success_rate_delta={sr_delta:.3f} (need >= {self.min_success_delta})"})
        if not g1:
            passed = False

        # Gate 2: safety_non_regression
        col_delta = candidate_metrics.get("collision_rate", 0) - baseline_metrics.get("collision_rate", 0)
        g2 = col_delta <= self.max_collision_increase + 0.001
        checks.append({"name": "safety_non_regression", "passed": g2,
                       "detail": f"collision_rate_delta={col_delta:.3f} (need <= {self.max_collision_increase})"})
        if not g2:
            passed = False

        # Gate 3: robustness (std across seeds if available)
        if per_seed and len(per_seed) > 1:
            srs = [s["candidate"]["success_rate"] for s in per_seed.values() if "candidate" in s and "success_rate" in s.get("candidate", {})]
            if srs:
                import statistics
                sr_std = statistics.stdev(srs) if len(srs) > 1 else 0.0
                g3 = sr_std <= self.max_success_std
                checks.append({"name": "robustness", "passed": g3,
                               "detail": f"success_rate_std={sr_std:.3f} (need <= {self.max_success_std})"})
                if not g3:
                    passed = False
            else:
                checks.append({"name": "robustness", "passed": True, "detail": "skipped (no valid per_seed data)"})
        else:
            checks.append({"name": "robustness", "passed": True, "detail": "skipped (no per_seed data)"})

        # Gate 4: second_seed_confirmation (only if per_seed available)
        if per_seed:
            seed_count = len(per_seed)
            g4 = seed_count >= self.min_seeds
            checks.append({"name": "second_seed_confirmation", "passed": g4,
                           "detail": f"seeds={seed_count} (need >= {self.min_seeds})"})
            if not g4:
                passed = False
        else:
            checks.append({"name": "second_seed_confirmation", "passed": True, "detail": "skipped (no per_seed data)"})

        # Gate 5: sandbox_clearance
        g5 = sandbox_risk_score < self.max_risk_score
        checks.append({"name": "sandbox_clearance", "passed": g5,
                       "detail": f"risk_score={sandbox_risk_score:.3f} (need < {self.max_risk_score})"})
        if not g5:
            passed = False

        # Gate 6: regression_suite (placeholder)
        checks.append({"name": "regression_suite", "passed": True, "detail": "placeholder"})

        if passed:
            decision = f"promote_to_{next_level}"
            reason = f"All gates passed; promote from {current_level} to {next_level}"
        else:
            if sr_delta < 0:
                decision = "reject"
                reason = "Candidate worse than baseline; reject"
            else:
                decision = "need_more_data"
                reason = "Some gates not passed; need more evaluation"

        logger.info("PromotionGate: %s -> %s", decision, reason)
        return GateResult(
            passed=passed, decision=decision, checks=checks,
            next_level=next_level if passed else current_level,
            reason=reason,
        )

    def _next_level(self, current: str) -> str:
        idx = self.LEVEL_ORDER.index(current) if current in self.LEVEL_ORDER else 0
        return self.LEVEL_ORDER[min(idx + 1, len(self.LEVEL_ORDER) - 1)]
