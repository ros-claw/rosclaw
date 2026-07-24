"""ChoreographyValidator (数据库优化v4 §8.4).

    读取当前 Contract
    → 应用 Candidate Patch 到 Timing Model
    → 重算 Phase Timeline
    → 检查 Reveal Window
    → 检查 Round Budget
    → 检查 Patch Stacking
    → Allow / Block

run1's death-spiral patch (servo_speed_scale + per_phase_delay) must be
blocked 100% of the time — those parameters are in
``forbidden_parameters`` and can never pass (v4 §11.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract import ControlChoreographyContract
from .timing import TimingModel, apply_patch

# Violation codes (stable, machine-readable).
V_FORBIDDEN_PARAMETER = "forbidden_parameter"
V_UNKNOWN_PARAMETER = "unknown_parameter"
V_OUT_OF_RANGE = "parameter_out_of_range"
V_NOT_PATCHABLE_PHASE = "parameter_not_between_round_allowed"
V_NON_STACKABLE = "non_stackable_parameter_conflict"
V_REVEAL_WINDOW = "reveal_window_violated"
V_ROUND_BUDGET = "round_budget_exceeded"
V_PHASE_DURATION = "phase_duration_violated"


@dataclass
class ChoreographyValidation:
    """Validator output (v4 §8.4)."""

    allowed: bool
    violations: list[str] = field(default_factory=list)

    original_phase_durations: dict[str, float] = field(default_factory=dict)
    patched_phase_durations: dict[str, float] = field(default_factory=dict)

    reveal_window_preserved: bool = True
    total_round_budget_preserved: bool = True
    non_stackable_respected: bool = True

    expected_reveal_offset_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "violations": self.violations,
            "original_phase_durations": self.original_phase_durations,
            "patched_phase_durations": self.patched_phase_durations,
            "reveal_window_preserved": self.reveal_window_preserved,
            "total_round_budget_preserved": self.total_round_budget_preserved,
            "non_stackable_respected": self.non_stackable_respected,
            "expected_reveal_offset_ms": self.expected_reveal_offset_ms,
        }


class ChoreographyValidator:
    """Validates candidate patches against a contract + timing model."""

    def __init__(self, contract: ControlChoreographyContract) -> None:
        self._contract = contract

    @property
    def contract(self) -> ControlChoreographyContract:
        return self._contract

    def validate(
        self,
        patch: dict[str, Any],
        model: TimingModel,
        *,
        phase: str = "between_rounds",
    ) -> ChoreographyValidation:
        """Allow/Block a candidate patch against the contract (v4 §8.4)."""
        violations: list[str] = []
        contract = self._contract

        # 1) Parameter space: forbidden and unknown keys are hard blocks.
        for name, value in patch.items():
            if name in contract.forbidden_parameters:
                violations.append(f"{V_FORBIDDEN_PARAMETER}:{name}")
                continue
            rule = contract.parameter_rule(name)
            if rule is None:
                violations.append(f"{V_UNKNOWN_PARAMETER}:{name}")
                continue
            if not self._value_in_range(value, rule):
                violations.append(f"{V_OUT_OF_RANGE}:{name}={value!r}")
            if phase == "between_rounds" and (
                contract.between_round_only and name not in contract.between_round_only
            ):
                violations.append(f"{V_NOT_PATCHABLE_PHASE}:{name}")

        # 2) Stacking: a non-stackable parameter conflicts when the current
        #    model already carries a different value for another
        #    non-stackable parameter (run1 stacked speed+delay changes).
        non_stackable_respected = True
        touched = [name for name in patch if name in contract.non_stackable_parameters]
        if len(touched) > 1:
            non_stackable_respected = False
            violations.append(f"{V_NON_STACKABLE}:{'+'.join(touched)}")
        if touched:
            current = model.parameters.get("inter_round_cooldown_sec")
            if (
                "inter_round_cooldown_sec" not in touched
                and current
                and contract.non_stackable_parameters
            ):
                non_stackable_respected = False
                violations.append(f"{V_NON_STACKABLE}:cooldown_already_active")

        patched = apply_patch(model, patch, contract)

        # 3) Reveal window: the patched reveal offset must stay inside the
        #    contract window (between-round patches never move it; any
        #    timing-affecting patch that did would be forbidden above).
        reveal_preserved = True
        if (
            contract.reveal_window_start_ms is not None
            and contract.reveal_window_end_ms is not None
            and patched.reveal_offset_ms is not None
        ):
            reveal_preserved = (
                contract.reveal_window_start_ms
                <= patched.reveal_offset_ms
                <= contract.reveal_window_end_ms
            )
            if not reveal_preserved:
                violations.append(
                    f"{V_REVEAL_WINDOW}:offset={patched.reveal_offset_ms:.0f}ms "
                    f"outside [{contract.reveal_window_start_ms:.0f}, "
                    f"{contract.reveal_window_end_ms:.0f}]ms"
                )

        # 4) Round budget.
        budget_preserved = patched.round_duration_ms() <= contract.maximum_round_duration_ms
        if not budget_preserved:
            violations.append(
                f"{V_ROUND_BUDGET}:{patched.round_duration_ms():.0f}ms > "
                f"{contract.maximum_round_duration_ms:.0f}ms"
            )

        # 5) Phase durations vs contract bounds.
        for phase in contract.phases:
            patched_ms = patched.phase_durations_ms.get(phase.name, 0.0)
            if phase.name == "cooldown":
                # Cooldown is patchable; its bound is the round budget,
                # already checked above.
                continue
            if not (phase.minimum_duration_ms <= patched_ms <= phase.maximum_duration_ms):
                violations.append(
                    f"{V_PHASE_DURATION}:{phase.name}={patched_ms:.0f}ms outside "
                    f"[{phase.minimum_duration_ms:.0f}, {phase.maximum_duration_ms:.0f}]ms"
                )

        return ChoreographyValidation(
            allowed=not violations,
            violations=violations,
            original_phase_durations=dict(model.phase_durations_ms),
            patched_phase_durations=dict(patched.phase_durations_ms),
            reveal_window_preserved=reveal_preserved,
            total_round_budget_preserved=budget_preserved,
            non_stackable_respected=non_stackable_respected,
            expected_reveal_offset_ms=patched.reveal_offset_ms,
        )

    @staticmethod
    def _value_in_range(value: Any, rule: dict[str, Any]) -> bool:
        kind = rule.get("type", "number")
        if kind == "boolean":
            return isinstance(value, bool)
        if kind == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            return rule.get("min", value) <= value <= rule.get("max", value)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        return rule.get("min", value) <= value <= rule.get("max", value)
