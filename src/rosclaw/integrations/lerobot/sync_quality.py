"""Synchronization quality gates for Gate B.1.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It evaluates per-feature synchronization statistics against
configurable thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from rosclaw.integrations.lerobot.clock_mapping import ClockMappingResult
from rosclaw.integrations.lerobot.practice_normalizer import NormalizationError
from rosclaw.integrations.lerobot.sync_stats import SyncFeatureStats

QualityAction = Literal["fail", "warn", "ignore"]


@dataclass
class FeatureQualityRule:
    """Thresholds for one feature."""

    minimum_coverage: float | None = None
    max_p95_skew_ms: float | None = None
    max_mean_skew_ms: float | None = None
    max_mean_hold_age_ms: float | None = None
    min_samples_per_window: int | None = None
    action: QualityAction = "warn"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"action": self.action}
        if self.minimum_coverage is not None:
            out["minimum_coverage"] = self.minimum_coverage
        if self.max_p95_skew_ms is not None:
            out["max_p95_skew_ms"] = self.max_p95_skew_ms
        if self.max_mean_skew_ms is not None:
            out["max_mean_skew_ms"] = self.max_mean_skew_ms
        if self.max_mean_hold_age_ms is not None:
            out["max_mean_hold_age_ms"] = self.max_mean_hold_age_ms
        if self.min_samples_per_window is not None:
            out["min_samples_per_window"] = self.min_samples_per_window
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureQualityRule:
        return cls(
            minimum_coverage=data.get("minimum_coverage"),
            max_p95_skew_ms=data.get("max_p95_skew_ms"),
            max_mean_skew_ms=data.get("max_mean_skew_ms"),
            max_mean_hold_age_ms=data.get("max_mean_hold_age_ms"),
            min_samples_per_window=data.get("min_samples_per_window"),
            action=data.get("action", "warn"),
        )


@dataclass
class ClockQualityRule:
    """Thresholds for clock-level checks."""

    unmapped_mixed_domains_action: QualityAction = "fail"

    def to_dict(self) -> dict[str, Any]:
        return {"unmapped_mixed_domains_action": self.unmapped_mixed_domains_action}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClockQualityRule:
        return cls(
            unmapped_mixed_domains_action=data.get(
                "unmapped_mixed_domains_action", "fail"
            ),
        )


@dataclass
class QualityConfig:
    """Complete quality gate configuration."""

    profile: str = "balanced"
    rules: dict[str, FeatureQualityRule] = field(default_factory=dict)
    clocks: ClockQualityRule = field(default_factory=ClockQualityRule)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "rules": {k: v.to_dict() for k, v in self.rules.items()},
            "clocks": self.clocks.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityConfig:
        return cls(
            profile=data.get("profile", "balanced"),
            rules={
                k: FeatureQualityRule.from_dict(v)
                for k, v in data.get("rules", {}).items()
            },
            clocks=ClockQualityRule.from_dict(data.get("clocks", {})),
        )

    def rule_for(self, feature_key: str) -> FeatureQualityRule | None:
        return self.rules.get(feature_key)


QUALITY_PROFILES: dict[str, dict[str, FeatureQualityRule]] = {
    "strict": {
        "observation.state": FeatureQualityRule(
            minimum_coverage=1.0, action="fail"
        ),
        "action": FeatureQualityRule(minimum_coverage=1.0, action="fail"),
        "observation.images.front": FeatureQualityRule(
            minimum_coverage=0.95, max_p95_skew_ms=33.0, action="fail"
        ),
        "observation.motor_current": FeatureQualityRule(
            minimum_coverage=0.95, action="warn"
        ),
        "observation.force_torque": FeatureQualityRule(
            minimum_coverage=0.95, action="warn"
        ),
        "observation.contact": FeatureQualityRule(
            minimum_coverage=0.95, action="warn"
        ),
    },
    "balanced": {
        "observation.state": FeatureQualityRule(
            minimum_coverage=1.0, action="fail"
        ),
        "action": FeatureQualityRule(minimum_coverage=1.0, action="fail"),
        "observation.images.front": FeatureQualityRule(
            minimum_coverage=0.95, max_p95_skew_ms=50.0, action="fail"
        ),
        "observation.motor_current": FeatureQualityRule(
            minimum_coverage=0.80, action="warn"
        ),
        "observation.force_torque": FeatureQualityRule(
            minimum_coverage=0.80, action="warn"
        ),
        "observation.contact": FeatureQualityRule(
            minimum_coverage=0.80, action="warn"
        ),
    },
    "permissive": {
        "observation.state": FeatureQualityRule(
            minimum_coverage=1.0, action="fail"
        ),
        "action": FeatureQualityRule(minimum_coverage=1.0, action="fail"),
        "observation.images.front": FeatureQualityRule(
            minimum_coverage=0.80, max_p95_skew_ms=100.0, action="warn"
        ),
        "observation.motor_current": FeatureQualityRule(
            minimum_coverage=0.50, action="warn"
        ),
        "observation.force_torque": FeatureQualityRule(
            minimum_coverage=0.50, action="warn"
        ),
        "observation.contact": FeatureQualityRule(
            minimum_coverage=0.50, action="warn"
        ),
    },
}


def default_quality_config(profile: str = "balanced") -> QualityConfig:
    """Return the built-in quality gate configuration for ``profile``."""
    if profile not in QUALITY_PROFILES:
        raise NormalizationError(
            "invalid_quality_profile",
            f"Unknown quality profile '{profile}'. Expected one of {set(QUALITY_PROFILES)}.",
        )
    return QualityConfig(
        profile=profile,
        rules=dict(QUALITY_PROFILES[profile]),
        clocks=ClockQualityRule(unmapped_mixed_domains_action="fail"),
    )


@dataclass
class FeatureQualityResult:
    """Evaluation result for one feature."""

    feature_key: str
    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_key": self.feature_key,
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class QualityResult:
    """Overall quality gate evaluation result."""

    profile: str
    passed: bool
    action: QualityAction
    feature_results: list[FeatureQualityResult]
    clock_result: dict[str, Any]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "passed": self.passed,
            "action": self.action,
            "feature_results": [r.to_dict() for r in self.feature_results],
            "clock_result": self.clock_result,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _evaluate_feature(
    stats: SyncFeatureStats,
    rule: FeatureQualityRule | None,
) -> FeatureQualityResult:
    result = FeatureQualityResult(feature_key=stats.feature_key, passed=True)
    if rule is None:
        return result

    if rule.minimum_coverage is not None:
        coverage = stats.coverage_ratio if stats.coverage_ratio is not None else 0.0
        if coverage + 1e-9 < rule.minimum_coverage:
            msg = (
                f"{stats.feature_key} coverage {coverage:.3f} below "
                f"minimum {rule.minimum_coverage:.3f}"
            )
            if rule.action == "fail":
                result.errors.append(msg)
                result.passed = False
            elif rule.action == "warn":
                result.warnings.append(msg)

    if (
        rule.max_p95_skew_ms is not None
        and stats.p95_skew_ms is not None
        and stats.p95_skew_ms > rule.max_p95_skew_ms + 1e-9
    ):
        msg = (
            f"{stats.feature_key} p95 skew {stats.p95_skew_ms:.3f} ms exceeds "
            f"{rule.max_p95_skew_ms:.3f} ms"
        )
        if rule.action == "fail":
            result.errors.append(msg)
            result.passed = False
        elif rule.action == "warn":
            result.warnings.append(msg)

    if (
        rule.max_mean_skew_ms is not None
        and stats.mean_skew_ms is not None
        and stats.mean_skew_ms > rule.max_mean_skew_ms + 1e-9
    ):
        msg = (
            f"{stats.feature_key} mean skew {stats.mean_skew_ms:.3f} ms exceeds "
            f"{rule.max_mean_skew_ms:.3f} ms"
        )
        if rule.action == "fail":
            result.errors.append(msg)
            result.passed = False
        elif rule.action == "warn":
            result.warnings.append(msg)

    if (
        rule.max_mean_hold_age_ms is not None
        and stats.mean_hold_age_ms is not None
        and stats.mean_hold_age_ms > rule.max_mean_hold_age_ms + 1e-9
    ):
        msg = (
            f"{stats.feature_key} mean hold age {stats.mean_hold_age_ms:.3f} ms "
            f"exceeds {rule.max_mean_hold_age_ms:.3f} ms"
        )
        if rule.action == "fail":
            result.errors.append(msg)
            result.passed = False
        elif rule.action == "warn":
            result.warnings.append(msg)

    if (
        rule.min_samples_per_window is not None
        and stats.min_samples_per_window is not None
        and stats.min_samples_per_window < rule.min_samples_per_window
    ):
        msg = (
            f"{stats.feature_key} min samples per window "
            f"{stats.min_samples_per_window} below {rule.min_samples_per_window}"
        )
        if rule.action == "fail":
            result.errors.append(msg)
            result.passed = False
        elif rule.action == "warn":
            result.warnings.append(msg)

    return result


def evaluate_quality(
    stats: list[SyncFeatureStats],
    mapping_result: ClockMappingResult,
    config: QualityConfig,
) -> QualityResult:
    """Evaluate synchronization statistics against quality gates."""
    feature_results: list[FeatureQualityResult] = []
    errors: list[str] = []
    warnings: list[str] = []
    any_fail = False
    any_warn = False

    for feature_stats in stats:
        rule = config.rule_for(feature_stats.feature_key)
        result = _evaluate_feature(feature_stats, rule)
        feature_results.append(result)
        if not result.passed:
            any_fail = True
        if result.warnings:
            any_warn = True
        errors.extend(result.errors)
        warnings.extend(result.warnings)

    clock_passed = True
    clock_action: QualityAction = "ignore"
    if mapping_result.unmapped_clocks:
        clock_action = config.clocks.unmapped_mixed_domains_action
        msg = f"Unmapped clock domains: {sorted(mapping_result.unmapped_clocks)}"
        if clock_action == "fail":
            errors.append(msg)
            clock_passed = False
            any_fail = True
        elif clock_action == "warn":
            warnings.append(msg)
            any_warn = True

    clock_result = {
        "passed": clock_passed,
        "action": clock_action,
        "unmapped_clocks": sorted(mapping_result.unmapped_clocks),
    }

    if any_fail:
        overall_action: QualityAction = "fail"
    elif any_warn:
        overall_action = "warn"
    else:
        overall_action = "ignore"

    return QualityResult(
        profile=config.profile,
        passed=not any_fail,
        action=overall_action,
        feature_results=feature_results,
        clock_result=clock_result,
        errors=errors,
        warnings=warnings,
    )


__all__ = [
    "FeatureQualityRule",
    "QualityAction",
    "QualityConfig",
    "QualityResult",
    "default_quality_config",
    "evaluate_quality",
]
