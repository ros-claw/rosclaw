"""Bounded SIM-only direct-torque residual for G1 ball contact learning.

The six-dimensional action is intentionally small and contact centred.  It is
added to the qualified PD command before the existing joint-boundary guard and
hard torque projector; it cannot increase actuator authority or reach a real
robot path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from rosclaw.feedback.contracts import canonical_hash

_RIGHT_LEG_INDICES = np.arange(6, 12)
_COUNTERBALANCE_INDICES = np.asarray((0, 1, 2, 4, 12, 14), dtype=np.int64)


@dataclass(frozen=True)
class G1BallisticContactTorqueResidualConfig:
    """One replay-learnable torque pulse around the measured contact event."""

    right_leg_residual_nm: tuple[float, ...] = (0.0,) * 6
    right_leg_preload_nm: tuple[float, ...] = (0.0,) * 6
    right_leg_phase_offset_sec: tuple[float, ...] = (0.0,) * 6
    counterbalance_residual_nm: tuple[float, ...] = (0.0,) * 6
    contact_policy_frame: int = 256
    lead_duration_sec: float = 0.16
    trail_duration_sec: float = 0.08
    maximum_joint_residual_nm: float = 12.0
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.growth.g1_ballistic_contact_torque_residual.v4"

    def __post_init__(self) -> None:
        if len(self.right_leg_residual_nm) != 6 or not all(
            math.isfinite(value) for value in self.right_leg_residual_nm
        ):
            raise ValueError("ballistic contact torque residual requires six finite values")
        if any(abs(value) > self.maximum_joint_residual_nm for value in self.right_leg_residual_nm):
            raise ValueError("ballistic contact torque residual exceeds its SIM-only limit")
        if len(self.right_leg_preload_nm) != 6 or not all(
            math.isfinite(value) for value in self.right_leg_preload_nm
        ):
            raise ValueError("ballistic contact torque preload requires six finite values")
        if any(
            abs(residual) + abs(preload) > self.maximum_joint_residual_nm
            for residual, preload in zip(
                self.right_leg_residual_nm,
                self.right_leg_preload_nm,
                strict=True,
            )
        ):
            raise ValueError("ballistic contact torque combined pulse exceeds its SIM-only limit")
        if len(self.right_leg_phase_offset_sec) != 6 or not all(
            math.isfinite(value) for value in self.right_leg_phase_offset_sec
        ):
            raise ValueError("ballistic contact torque phase requires six finite values")
        if any(abs(value) > 0.04 for value in self.right_leg_phase_offset_sec):
            raise ValueError("ballistic contact torque phase exceeds 0.04 s")
        if len(self.counterbalance_residual_nm) != 6 or not all(
            math.isfinite(value) for value in self.counterbalance_residual_nm
        ):
            raise ValueError("ballistic counterbalance torque requires six finite values")
        if any(abs(value) > 6.0 for value in self.counterbalance_residual_nm):
            raise ValueError("ballistic counterbalance torque exceeds 6 Nm")
        if not 240 <= self.contact_policy_frame <= 300:
            raise ValueError("ballistic contact torque frame must be in [240, 300]")
        if not 0.08 <= self.lead_duration_sec <= 0.24:
            raise ValueError("ballistic contact torque lead must be in [0.08, 0.24] s")
        if not 0.04 <= self.trail_duration_sec <= 0.16:
            raise ValueError("ballistic contact torque trail must be in [0.04, 0.16] s")
        if not 1.0 <= self.maximum_joint_residual_nm <= 12.0:
            raise ValueError("ballistic contact torque limit must be in [1, 12] Nm")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("ballistic contact torque residual is SIM_ONLY")

    @property
    def enabled(self) -> bool:
        return any(
            abs(value) > 1e-12
            for value in (
                *self.right_leg_residual_nm,
                *self.right_leg_preload_nm,
                *self.counterbalance_residual_nm,
            )
        )

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))


def g1_ballistic_contact_torque_residual(
    *,
    policy_frame: int,
    control_dt_sec: float,
    config: G1BallisticContactTorqueResidualConfig,
) -> tuple[np.ndarray, bool]:
    """Evaluate the smooth torque pulse without bypassing runtime projection."""

    if not math.isfinite(control_dt_sec) or control_dt_sec <= 0.0:
        raise ValueError("ballistic contact torque control clock must be positive")
    torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    if not config.enabled:
        return torque, False
    nominal_time = (policy_frame - config.contact_policy_frame) * control_dt_sec
    envelopes: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
    preload_envelopes: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
    for index, phase_offset in enumerate(config.right_leg_phase_offset_sec):
        relative_time = nominal_time - phase_offset
        if relative_time < -config.lead_duration_sec or relative_time > config.trail_duration_sec:
            continue
        if relative_time <= 0.0:
            progress = (relative_time + config.lead_duration_sec) / config.lead_duration_sec
            envelopes[index] = math.sin(0.5 * math.pi * progress) ** 2
            if 0.0 < progress < 1.0:
                preload_envelopes[index] = math.sin(math.pi * progress) ** 2
        else:
            progress = relative_time / config.trail_duration_sec
            envelopes[index] = math.cos(0.5 * math.pi * progress) ** 2
    torque[_RIGHT_LEG_INDICES] = envelopes * np.asarray(
        config.right_leg_residual_nm, dtype=np.float64
    ) + preload_envelopes * np.asarray(config.right_leg_preload_nm, dtype=np.float64)
    if -config.lead_duration_sec <= nominal_time <= config.trail_duration_sec:
        if nominal_time <= 0.0:
            progress = (nominal_time + config.lead_duration_sec) / config.lead_duration_sec
            counterbalance_envelope = math.sin(0.5 * math.pi * progress) ** 2
        else:
            progress = nominal_time / config.trail_duration_sec
            counterbalance_envelope = math.cos(0.5 * math.pi * progress) ** 2
        torque[_COUNTERBALANCE_INDICES] = counterbalance_envelope * np.asarray(
            config.counterbalance_residual_nm,
            dtype=np.float64,
        )
    return torque, bool(np.any(np.abs(torque) > 1e-12))


__all__ = [
    "G1BallisticContactTorqueResidualConfig",
    "g1_ballistic_contact_torque_residual",
]
