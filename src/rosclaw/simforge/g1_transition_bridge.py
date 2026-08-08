"""Auditable velocity-matched transition between two G1 motion experts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class G1TransitionBridgeConfig:
    """Boundary authority for a quintic approach-to-strike transition."""

    duration_sec: float = 0.60
    entry_velocity_scale: float = 0.0
    exit_velocity_scale: float = 0.0
    maximum_boundary_velocity_rad_s: float = 2.0
    schema_version: str = "rosclaw.simforge.g1_transition_bridge_config.v2"

    def __post_init__(self) -> None:
        values = (
            self.duration_sec,
            self.entry_velocity_scale,
            self.exit_velocity_scale,
            self.maximum_boundary_velocity_rad_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("G1 transition bridge config must be finite")
        if not 0.16 <= self.duration_sec <= 2.00:
            raise ValueError("G1 transition bridge duration must be in [0.16, 2.00] s")
        if not 0.0 <= self.entry_velocity_scale <= 1.0:
            raise ValueError("G1 transition entry velocity scale must be in [0, 1]")
        if not 0.0 <= self.exit_velocity_scale <= 1.25:
            raise ValueError("G1 transition exit velocity scale must be in [0, 1.25]")
        if not 0.5 <= self.maximum_boundary_velocity_rad_s <= 4.0:
            raise ValueError("G1 transition boundary velocity limit must be in [0.5, 4] rad/s")

    @property
    def config_hash(self) -> str:
        return hash_json(asdict(self))

    @property
    def velocity_matched(self) -> bool:
        return self.entry_velocity_scale > 0.0 or self.exit_velocity_scale > 0.0


@dataclass(frozen=True)
class G1TransitionBridgeSample:
    """Position, velocity and acceleration reference at one bridge time."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class G1VelocityMatchedTransitionBridge:
    """Quintic joint reference with bounded position/velocity boundaries.

    Both endpoint accelerations are zero.  With zero velocity scales this is
    exactly the legacy minimum-jerk position bridge, which makes the rollout
    change explicit and permits controller-equivalent baseline comparisons.
    """

    def __init__(
        self,
        *,
        entry_position: np.ndarray,
        entry_velocity: np.ndarray,
        exit_position: np.ndarray,
        exit_velocity: np.ndarray,
        config: G1TransitionBridgeConfig,
    ) -> None:
        arrays = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (entry_position, entry_velocity, exit_position, exit_velocity)
        )
        if any(value.shape != (29,) for value in arrays):
            raise ValueError("G1 transition bridge boundaries must contain 29 joints")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("G1 transition bridge boundaries must be finite")
        self.config = config
        self.entry_position = arrays[0].copy()
        self.exit_position = arrays[2].copy()
        limit = config.maximum_boundary_velocity_rad_s
        self.entry_velocity = np.clip(
            arrays[1] * config.entry_velocity_scale,
            -limit,
            limit,
        )
        self.exit_velocity = np.clip(
            arrays[3] * config.exit_velocity_scale,
            -limit,
            limit,
        )
        self.boundary_velocity_projection_applied = bool(
            np.any(np.abs(self.entry_velocity - arrays[1] * config.entry_velocity_scale) > 1e-12)
            or np.any(np.abs(self.exit_velocity - arrays[3] * config.exit_velocity_scale) > 1e-12)
        )
        duration = config.duration_sec
        displacement = self.exit_position - self.entry_position
        entry_rate = duration * self.entry_velocity
        exit_rate = duration * self.exit_velocity
        self._coefficients = np.stack(
            (
                self.entry_position,
                entry_rate,
                np.zeros(29, dtype=np.float64),
                10.0 * displacement - 6.0 * entry_rate - 4.0 * exit_rate,
                -15.0 * displacement + 8.0 * entry_rate + 7.0 * exit_rate,
                6.0 * displacement - 3.0 * entry_rate - 3.0 * exit_rate,
            ),
            axis=0,
        )

    def sample(self, elapsed_sec: float) -> G1TransitionBridgeSample:
        if not math.isfinite(elapsed_sec):
            raise ValueError("G1 transition bridge sample time must be finite")
        duration = self.config.duration_sec
        phase = float(np.clip(elapsed_sec / duration, 0.0, 1.0))
        if not self.config.velocity_matched:
            blend = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
            acceleration_scale = (60.0 * phase - 180.0 * phase**2 + 120.0 * phase**3) / (
                duration**2
            )
            return G1TransitionBridgeSample(
                position=(1.0 - blend) * self.entry_position + blend * self.exit_position,
                # The compatibility path deliberately preserves the legacy
                # zero-velocity PD target, not the analytic blend derivative.
                velocity=np.zeros(29, dtype=np.float64),
                acceleration=acceleration_scale * (self.exit_position - self.entry_position),
            )
        powers = np.asarray((1.0, phase, phase**2, phase**3, phase**4, phase**5))
        derivative = np.asarray(
            (0.0, 1.0, 2.0 * phase, 3.0 * phase**2, 4.0 * phase**3, 5.0 * phase**4)
        )
        second = np.asarray((0.0, 0.0, 2.0, 6.0 * phase, 12.0 * phase**2, 20.0 * phase**3))
        return G1TransitionBridgeSample(
            position=powers @ self._coefficients,
            velocity=(derivative @ self._coefficients) / duration,
            acceleration=(second @ self._coefficients) / (duration**2),
        )

    def audit_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config.config_hash,
            "velocity_matched": self.config.velocity_matched,
            "entry_velocity_rms_rad_s": _rms(self.entry_velocity),
            "exit_velocity_rms_rad_s": _rms(self.exit_velocity),
            "boundary_velocity_projection_applied": self.boundary_velocity_projection_applied,
            "schema_version": "rosclaw.simforge.g1_transition_bridge_audit.v1",
        }


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


__all__ = [
    "G1TransitionBridgeConfig",
    "G1TransitionBridgeSample",
    "G1VelocityMatchedTransitionBridge",
]
