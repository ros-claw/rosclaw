"""Control choreography contract schema (数据库优化v4 §8.2).

A contract binds one task's control phases, reveal window, round budget,
and the patch space (what may/may not be patched, stacking rules, when
patches may apply).  It is the machine-checkable form of the run1 lesson:
slow_down + delay broke the reveal window and invalid rate went from ~4%
to 21–52%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ControlPhase:
    name: str
    start_event: str
    end_event: str
    maximum_duration_ms: float
    minimum_duration_ms: float = 0.0
    interruptible: bool = False


@dataclass(frozen=True)
class ControlChoreographyContract:
    """v4 §8.2 contract schema."""

    contract_id: str
    task_id: str
    skill_id: str | None
    version: str

    phases: tuple[ControlPhase, ...]

    reveal_window_start_ms: float | None
    reveal_window_end_ms: float | None
    maximum_round_duration_ms: float

    patchable_parameters: dict[str, dict[str, Any]]
    forbidden_parameters: tuple[str, ...]
    non_stackable_parameters: tuple[str, ...]

    between_round_only: tuple[str, ...]
    before_session_only: tuple[str, ...]

    required_invariants: tuple[str, ...] = field(
        default=("reveal_window_preserved", "round_budget_preserved", "non_stackable_respected")
    )

    def parameter_rule(self, name: str) -> dict[str, Any] | None:
        return self.patchable_parameters.get(name)


def contract_from_dict(raw: dict[str, Any]) -> ControlChoreographyContract:
    section = raw.get("contract", raw)
    phases = tuple(
        ControlPhase(
            name=str(item["name"]),
            start_event=str(item["start_event"]),
            end_event=str(item["end_event"]),
            maximum_duration_ms=float(item["maximum_duration_ms"]),
            minimum_duration_ms=float(item.get("minimum_duration_ms") or 0.0),
            interruptible=bool(item.get("interruptible", False)),
        )
        for item in section["phases"]
    )
    return ControlChoreographyContract(
        contract_id=str(section["contract_id"]),
        task_id=str(section["task_id"]),
        skill_id=section.get("skill_id"),
        version=str(section.get("version", "1.0")),
        phases=phases,
        reveal_window_start_ms=section.get("reveal_window_start_ms"),
        reveal_window_end_ms=section.get("reveal_window_end_ms"),
        maximum_round_duration_ms=float(section["maximum_round_duration_ms"]),
        patchable_parameters=dict(section.get("patchable_parameters") or {}),
        forbidden_parameters=tuple(section.get("forbidden_parameters") or ()),
        non_stackable_parameters=tuple(section.get("non_stackable_parameters") or ()),
        between_round_only=tuple(section.get("between_round_only") or ()),
        before_session_only=tuple(section.get("before_session_only") or ()),
        required_invariants=tuple(section.get("required_invariants") or ()),
    )


def load_contract(path: str) -> ControlChoreographyContract:
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return contract_from_dict(raw)
