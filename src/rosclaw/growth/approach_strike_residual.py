"""Support-bound IQL torque residual for the G1 approach-to-strike window."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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
_ACTIVE_EVENT_PHASE_IDS = frozenset((1, 2, 3, 4))


@dataclass(frozen=True)
class G1ApproachStrikeResidualConfig:
    """A deliberately small, reversible residual authority envelope."""

    residual_fraction: float = 0.20
    maximum_residual_nm: float = 5.0
    maximum_standardized_rms: float = 6.0
    maximum_standardized_abs: float = 30.0
    joint_group: str = "whole_body"
    schema_version: str = "rosclaw.growth.g1_approach_strike_residual_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.residual_fraction,
            self.maximum_residual_nm,
            self.maximum_standardized_rms,
            self.maximum_standardized_abs,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("approach-strike residual config must be finite")
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
        if int(event_phase) not in _ACTIVE_EVENT_PHASE_IDS:
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
        return self.actor.action(state, baseline_torque)


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
    one_hot = np.zeros(_EVENT_PHASE_COUNT, dtype=np.float64)
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
]
