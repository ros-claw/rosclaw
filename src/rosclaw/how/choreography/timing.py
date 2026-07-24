"""Timing model — phase timeline computation + patch application (v4 §8.4).

The model reconstructs a task's phase timeline from observed practice
rounds (median durations) and the contract's phase bounds.  A candidate
patch is applied to the model and the timeline recomputed; the validator
then checks the reveal window, the round budget, and stacking rules
against the contract.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .contract import ControlChoreographyContract


@dataclass
class RoundTiming:
    """Observed timing of one round (from practice events)."""

    started_at: float
    ended_at: float | None
    reveal_at: float | None = None  # gesture presented (when recorded)

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return max(0.0, (self.ended_at - self.started_at) * 1000.0)


@dataclass
class TimingModel:
    """The reconstructed task timing (median of observed rounds)."""

    phase_durations_ms: dict[str, float]
    round_interval_ms: float
    cooldown_ms: float
    reveal_offset_ms: float | None
    parameters: dict[str, Any] = field(default_factory=dict)

    def round_duration_ms(self) -> float:
        return sum(self.phase_durations_ms.values())


def build_timing_model(
    contract: ControlChoreographyContract,
    rounds: list[RoundTiming],
    *,
    current_parameters: dict[str, Any] | None = None,
) -> TimingModel:
    """Reconstruct the timing model from observed rounds.

    Phase durations come from the contract's typical bounds mid-point when
    the session does not record phase boundaries, blended with observed
    round durations when those exist (never invented finer granularity
    than the evidence carries).
    """
    durations = [r.duration_ms for r in rounds if r.duration_ms is not None]
    intervals = [
        (b.started_at - a.started_at) * 1000.0 for a, b in zip(rounds, rounds[1:], strict=False)
    ]
    median_round = statistics.median(durations) if durations else None
    median_interval = statistics.median(intervals) if intervals else None

    phase_durations: dict[str, float] = {}
    fixed_phases = [p for p in contract.phases if p.name != "cooldown"]
    for phase in fixed_phases:
        # Observed phase boundaries are not recorded in v1 sessions; use the
        # contract's midpoint as the model value (declared, not measured).
        phase_durations[phase.name] = (phase.minimum_duration_ms + phase.maximum_duration_ms) / 2.0

    cooldown_ms = 0.0
    if median_interval is not None and median_round is not None:
        cooldown_ms = max(0.0, median_interval - median_round)
    elif median_interval is not None:
        in_round = sum(phase_durations.values())
        cooldown_ms = max(0.0, median_interval - in_round)
    phase_durations["cooldown"] = cooldown_ms

    reveal_offset = None
    reveals = [(r.reveal_at - r.started_at) * 1000.0 for r in rounds if r.reveal_at is not None]
    if reveals:
        reveal_offset = statistics.median(reveals)
    elif contract.reveal_window_start_ms is not None and contract.reveal_window_end_ms is not None:
        reveal_offset = (contract.reveal_window_start_ms + contract.reveal_window_end_ms) / 2.0

    return TimingModel(
        phase_durations_ms=phase_durations,
        round_interval_ms=median_interval or sum(phase_durations.values()),
        cooldown_ms=cooldown_ms,
        reveal_offset_ms=reveal_offset,
        parameters=dict(current_parameters or {}),
    )


def apply_patch(
    model: TimingModel,
    patch: dict[str, Any],
    contract: ControlChoreographyContract,
) -> TimingModel:
    """Apply a candidate patch to the timing model (returns a new model).

    Only between-round parameters influence the timeline; telemetry_hz is
    timing-neutral.  Unknown parameters raise — the validator rejects
    unknown/forbidden keys BEFORE this function is called.
    """
    known_between_round = {
        "inter_round_cooldown_sec",
        "cooldown_every_n_rounds",
        "rehome_between_blocks",
        "neutral_pose_between_blocks",
        "telemetry_hz",
    }
    unknown = sorted(set(patch) - known_between_round)
    if unknown:
        # Defense in depth (the validator rejects these first): the model
        # refuses to guess the timing effect of an unknown parameter.
        raise ValueError(f"unknown timing effect for parameters: {unknown}")

    new_phases = dict(model.phase_durations_ms)
    parameters = dict(model.parameters)
    parameters.update(patch)

    # Current effective cooldown: the larger of the observed interval and
    # the explicitly parameterized one — re-patching while a cooldown is
    # already active must not under-count the round total.
    param_cooldown_ms = float(model.parameters.get("inter_round_cooldown_sec") or 0.0) * 1000.0
    current_effective_ms = max(model.cooldown_ms, param_cooldown_ms)

    extra_cooldown_ms = 0.0
    if "inter_round_cooldown_sec" in patch:
        # The patch SETS the cooldown parameter (not adds to it).
        extra_cooldown_ms = max(
            0.0, float(patch["inter_round_cooldown_sec"]) * 1000.0 - param_cooldown_ms
        )
    if patch.get("rehome_between_blocks") or patch.get("neutral_pose_between_blocks"):
        # A between-block recovery action occupies cooldown time but must
        # fit inside it; modeled as +800 ms of cooldown occupancy.
        extra_cooldown_ms = max(extra_cooldown_ms, 800.0)

    new_phases["cooldown"] = max(0.0, current_effective_ms + extra_cooldown_ms)
    return TimingModel(
        phase_durations_ms=new_phases,
        round_interval_ms=model.round_interval_ms + extra_cooldown_ms,
        cooldown_ms=new_phases["cooldown"],
        reveal_offset_ms=model.reveal_offset_ms,  # between-round patches never move reveal
        parameters=parameters,
    )
