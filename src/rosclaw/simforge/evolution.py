"""Auditable state machine for autonomous simulation evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EvolutionState(StrEnum):
    FAILURE_DISCOVERED = "FAILURE_DISCOVERED"
    FAILURE_CLUSTERED = "FAILURE_CLUSTERED"
    DIAGNOSED = "DIAGNOSED"
    CANDIDATES_GENERATED = "CANDIDATES_GENERATED"
    SAME_SEED_RETRY = "SAME_SEED_RETRY"
    DEVELOPMENT_EVAL = "DEVELOPMENT_EVAL"
    VALIDATION_EVAL = "VALIDATION_EVAL"
    HOLDOUT_EVAL = "HOLDOUT_EVAL"
    FALSIFICATION = "FALSIFICATION"
    CROSS_BACKEND_CHECK = "CROSS_BACKEND_CHECK"
    SIM_CHAMPION = "SIM_CHAMPION"
    REJECTED = "REJECTED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    DEAD_END = "DEAD_END"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    SIMULATOR_DISAGREEMENT = "SIMULATOR_DISAGREEMENT"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"

    @property
    def terminal(self) -> bool:
        return self in {
            EvolutionState.SIM_CHAMPION,
            EvolutionState.REJECTED,
            EvolutionState.NEED_MORE_EVIDENCE,
            EvolutionState.DEAD_END,
            EvolutionState.RESOURCE_EXHAUSTED,
            EvolutionState.SIMULATOR_DISAGREEMENT,
            EvolutionState.HUMAN_REVIEW_REQUIRED,
        }


_PIPELINE = (
    EvolutionState.FAILURE_DISCOVERED,
    EvolutionState.FAILURE_CLUSTERED,
    EvolutionState.DIAGNOSED,
    EvolutionState.CANDIDATES_GENERATED,
    EvolutionState.SAME_SEED_RETRY,
    EvolutionState.DEVELOPMENT_EVAL,
    EvolutionState.VALIDATION_EVAL,
    EvolutionState.HOLDOUT_EVAL,
    EvolutionState.FALSIFICATION,
    EvolutionState.CROSS_BACKEND_CHECK,
    EvolutionState.SIM_CHAMPION,
)
_FAILURE_STATES = {state for state in EvolutionState if state.terminal} - {
    EvolutionState.SIM_CHAMPION
}


@dataclass
class EvolutionRun:
    run_id: str
    task_id: str
    state: EvolutionState = EvolutionState.FAILURE_DISCOVERED
    history: list[tuple[str, str]] = field(default_factory=list)

    def transition(self, target: EvolutionState, *, reason: str) -> None:
        if not reason:
            raise ValueError("evolution transitions require an audit reason")
        if self.state.terminal:
            raise RuntimeError(f"cannot transition terminal evolution state {self.state.value}")
        current_index = _PIPELINE.index(self.state)
        expected = _PIPELINE[current_index + 1]
        if target is not expected and target not in _FAILURE_STATES:
            raise RuntimeError(f"invalid evolution transition {self.state.value} -> {target.value}")
        self.history.append((target.value, reason))
        self.state = target


__all__ = ["EvolutionRun", "EvolutionState"]
