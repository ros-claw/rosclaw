"""Support-bound IQL torque residual for the G1 approach-to-strike window."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.approach_strike_contracts import EVENT_PHASE_NAMES, STATE_FEATURES
from rosclaw.growth.learners.iql import (
    IQLResidualDecision,
    IQLResidualGuardConfig,
    SupportBoundIQLResidualActor,
)

_EVENT_PHASE_COUNT = len(EVENT_PHASE_NAMES)
_DEFAULT_EVENT_PHASE_IDS = (1, 2, 3, 4)
_ALLOWED_EVENT_PHASE_IDS = frozenset((0, 1, 2, 3, 4))
_APPROACH_EVENT_PHASE_ID = 0


@dataclass(frozen=True)
class G1ApproachStrikeResidualConfig:
    """A deliberately small, reversible residual authority envelope."""

    residual_fraction: float = 0.20
    maximum_residual_nm: float = 5.0
    maximum_standardized_rms: float = 6.0
    maximum_standardized_abs: float = 30.0
    joint_group: str = "whole_body"
    active_event_phase_ids: tuple[int, ...] = _DEFAULT_EVENT_PHASE_IDS
    approach_release_distance_m: float = 0.0
    approach_full_authority_distance_m: float = 0.0
    schema_version: str = "rosclaw.growth.g1_approach_strike_residual_config.v3"

    def __post_init__(self) -> None:
        values = (
            self.residual_fraction,
            self.maximum_residual_nm,
            self.maximum_standardized_rms,
            self.maximum_standardized_abs,
            self.approach_release_distance_m,
            self.approach_full_authority_distance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("approach-strike residual config must be finite")
        if (
            not self.active_event_phase_ids
            or len(set(self.active_event_phase_ids)) != len(self.active_event_phase_ids)
            or not set(self.active_event_phase_ids).issubset(_ALLOWED_EVENT_PHASE_IDS)
        ):
            raise ValueError(
                "approach-strike active phases must be unique ids drawn from [0, 1, 2, 3, 4]"
            )
        if not (
            0.0
            <= self.approach_release_distance_m
            <= self.approach_full_authority_distance_m
            <= 5.0
        ):
            raise ValueError(
                "approach residual release distances must satisfy 0 <= release <= full <= 5 m"
            )
        # Reuse the shared guard's stricter contract validation.
        self.guard_config()

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))

    def guard_config(self) -> IQLResidualGuardConfig:
        return IQLResidualGuardConfig(
            residual_fraction=self.residual_fraction,
            maximum_residual_nm=self.maximum_residual_nm,
            maximum_standardized_rms=self.maximum_standardized_rms,
            maximum_standardized_abs=self.maximum_standardized_abs,
            joint_group=self.joint_group,
        )


class G1ApproachStrikeResidualController:
    """Load one unevaluated candidate without granting activation authority."""

    def __init__(
        self,
        candidate_path: Path,
        config: G1ApproachStrikeResidualConfig | None = None,
    ) -> None:
        self.config = config or G1ApproachStrikeResidualConfig()
        self.actor = SupportBoundIQLResidualActor.load(
            candidate_path,
            self.config.guard_config(),
        )
        if self.actor.actor.task_id != "g1_approach_strike_transition":
            raise ValueError("IQL candidate task is not approach-to-strike")
        if self.actor.actor.state_features != tuple(STATE_FEATURES):
            raise ValueError("IQL candidate state features do not match approach-to-strike")

    @property
    def candidate_hash(self) -> str:
        return self.actor.candidate_hash

    def propose(
        self,
        *,
        data: Any,
        ids: Any,
        target: np.ndarray,
        event_phase: int,
        baseline_torque: np.ndarray,
    ) -> IQLResidualDecision:
        phase_id = int(event_phase)
        if phase_id not in self.config.active_event_phase_ids:
            return IQLResidualDecision(
                residual_torque=np.zeros(29, dtype=np.float64),
                accepted=False,
                confidence=0.0,
                standardized_rms=0.0,
                standardized_abs=0.0,
                peak_residual_nm=0.0,
                reason="outside_approach_strike_event_window",
            )
        state = build_online_approach_strike_state(
            data=data,
            ids=ids,
            target=target,
            event_phase=event_phase,
        )
        authority = self._approach_authority(data=data, ids=ids, event_phase=phase_id)
        if authority <= 0.0:
            return IQLResidualDecision(
                residual_torque=np.zeros(29, dtype=np.float64),
                accepted=False,
                confidence=0.0,
                standardized_rms=0.0,
                standardized_abs=0.0,
                peak_residual_nm=0.0,
                reason="inside_approach_residual_release_distance",
            )
        decision = self.actor.action(state, baseline_torque)
        if not decision.accepted or authority >= 1.0:
            return decision
        residual = decision.residual_torque * authority
        return replace(
            decision,
            residual_torque=residual,
            confidence=decision.confidence * authority,
            peak_residual_nm=float(np.max(np.abs(residual))),
            reason="accepted_with_approach_distance_taper",
        )

    def _approach_authority(self, *, data: Any, ids: Any, event_phase: int) -> float:
        release = self.config.approach_release_distance_m
        full = self.config.approach_full_authority_distance_m
        if event_phase != _APPROACH_EVENT_PHASE_ID or full <= release:
            return 1.0
        pelvis_x = float(data.qpos[0])
        ball_x = float(data.qpos[ids.ball_qpos])
        return g1_approach_distance_authority(
            forward_distance_m=ball_x - pelvis_x,
            release_distance_m=release,
            full_authority_distance_m=full,
        )


def g1_approach_distance_authority(
    *,
    forward_distance_m: float,
    release_distance_m: float,
    full_authority_distance_m: float,
) -> float:
    """Return a bounded residual blend that releases authority before handoff."""

    values = (forward_distance_m, release_distance_m, full_authority_distance_m)
    if not all(math.isfinite(value) for value in values):
        return 0.0
    if not 0.0 <= release_distance_m < full_authority_distance_m:
        return 1.0
    return float(
        np.clip(
            (forward_distance_m - release_distance_m)
            / (full_authority_distance_m - release_distance_m),
            0.0,
            1.0,
        )
    )


def build_online_approach_strike_state(
    *,
    data: Any,
    ids: Any,
    target: np.ndarray,
    event_phase: int,
) -> np.ndarray:
    """Construct the deployment-side version of the frozen 110-D contract."""

    pelvis = np.asarray(data.qpos[:7], dtype=np.float64)
    ball = np.asarray(data.qpos[ids.ball_qpos : ids.ball_qpos + 3], dtype=np.float64)
    one_hot: np.ndarray = np.zeros(_EVENT_PHASE_COUNT, dtype=np.float64)
    one_hot[int(event_phase)] = 1.0
    state = np.concatenate(
        (
            np.asarray(data.qpos[7:36], dtype=np.float64),
            np.asarray(data.qvel[6:35], dtype=np.float64),
            pelvis[2:3],
            np.asarray(data.qvel[:3], dtype=np.float64),
            np.asarray(data.xquat[ids.torso], dtype=np.float64),
            ball - pelvis[:3],
            np.asarray(data.qvel[ids.ball_qvel : ids.ball_qvel + 3], dtype=np.float64),
            one_hot,
            np.asarray(target, dtype=np.float64),
        )
    )
    if state.shape != (len(STATE_FEATURES),) or not np.all(np.isfinite(state)):
        raise ValueError("online approach-strike state violates the frozen feature contract")
    return state


__all__ = [
    "G1ApproachStrikeResidualConfig",
    "G1ApproachStrikeResidualController",
    "build_online_approach_strike_state",
    "g1_approach_distance_authority",
]
