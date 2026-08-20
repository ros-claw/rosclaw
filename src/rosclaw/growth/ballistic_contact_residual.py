"""Bounded contact-centred joint target residual for G1 ballistic learning.

This is a deliberately small action surface for online SIM curricula.  The
residual never writes torque directly: it modifies six right-leg joint targets
inside a short contact-centred window and remains behind the qualified PD and
hard torque projectors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

G1_BALLISTIC_CONTACT_JOINT_NAMES = G1_DDS_JOINT_NAMES[6:12]
_RIGHT_LEG_INDICES = np.arange(6, 12)


@dataclass(frozen=True)
class G1BallisticContactResidualConfig:
    """One support-bounded action around the measured ball-contact phase."""

    right_leg_residual_rad: tuple[float, ...] = (0.0,) * 6
    contact_policy_frame: int = 256
    lead_duration_sec: float = 0.16
    trail_duration_sec: float = 0.08
    # This is an exploration ceiling, not an actuator-authority increase.  The
    # target still passes through the qualified PD controller, hard torque
    # projection and joint-boundary guard.  The 0.25 rad ceiling was retained
    # after strict 0.26--0.29 rad probes all crossed into rejected contact
    # islands; failures contract rather than expand this SIM-only curriculum.
    maximum_joint_residual_rad: float = 0.25
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.growth.g1_ballistic_contact_residual_config.v2"

    def __post_init__(self) -> None:
        if len(self.right_leg_residual_rad) != 6 or not all(
            math.isfinite(value) for value in self.right_leg_residual_rad
        ):
            raise ValueError("ballistic contact residual must contain six finite joints")
        if not 240 <= self.contact_policy_frame <= 300:
            raise ValueError("ballistic contact frame must be in [240, 300]")
        if not 0.08 <= self.lead_duration_sec <= 0.24:
            raise ValueError("ballistic contact lead duration must be in [0.08, 0.24] s")
        if not 0.04 <= self.trail_duration_sec <= 0.16:
            raise ValueError("ballistic contact trail duration must be in [0.04, 0.16] s")
        if not 0.05 <= self.maximum_joint_residual_rad <= 0.25:
            raise ValueError("ballistic contact residual limit must be in [0.05, 0.25] rad")
        if any(
            abs(value) > self.maximum_joint_residual_rad for value in self.right_leg_residual_rad
        ):
            raise ValueError("ballistic contact residual exceeds its joint limit")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("ballistic contact residual is SIM_ONLY")

    @property
    def enabled(self) -> bool:
        return any(abs(value) > 1e-12 for value in self.right_leg_residual_rad)

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))


def blend_g1_ballistic_contact_target(
    *,
    target: np.ndarray,
    policy_frame: int,
    control_dt_sec: float,
    config: G1BallisticContactResidualConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply a smooth, contact-centred target pulse to the right leg."""

    value = np.asarray(target, dtype=np.float64)
    if value.shape != (29,) or not np.all(np.isfinite(value)):
        raise ValueError("ballistic contact target must contain 29 finite joints")
    if not math.isfinite(control_dt_sec) or control_dt_sec <= 0.0:
        raise ValueError("ballistic contact control clock must be positive")
    delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    relative_time = (policy_frame - config.contact_policy_frame) * control_dt_sec
    if (
        not config.enabled
        or relative_time < -config.lead_duration_sec
        or relative_time > config.trail_duration_sec
    ):
        return value.copy(), delta, False
    if relative_time <= 0.0:
        progress = (relative_time + config.lead_duration_sec) / config.lead_duration_sec
        envelope = math.sin(0.5 * math.pi * progress) ** 2
    else:
        progress = relative_time / config.trail_duration_sec
        envelope = math.cos(0.5 * math.pi * progress) ** 2
    delta[_RIGHT_LEG_INDICES] = envelope * np.asarray(
        config.right_leg_residual_rad,
        dtype=np.float64,
    )
    return value + delta, delta, envelope > 1e-12


__all__ = [
    "G1_BALLISTIC_CONTACT_JOINT_NAMES",
    "G1BallisticContactResidualConfig",
    "blend_g1_ballistic_contact_target",
]
