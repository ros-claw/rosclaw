"""Bounded residual-based estimator for the GoalForge digital twin."""

from __future__ import annotations

from dataclasses import replace

from rosclaw.twin.g1_kick.belief import KickTwinBelief, ScalarBelief
from rosclaw.twin.g1_kick.prediction_error import KickPredictionError
from rosclaw.twin.g1_kick.update import KickTwinUpdate


class KickTwinEstimator:
    """Update beliefs only from disclosed physical residuals.

    The estimator intentionally accepts no scenario object, which makes reading
    hidden mass/friction/latency truth impossible at this boundary.
    """

    def update(
        self,
        belief: KickTwinBelief,
        error: KickPredictionError,
    ) -> tuple[KickTwinBelief, KickTwinUpdate]:
        changed: list[str] = []
        reasons: list[str] = []

        support = belief.support_ground_friction
        if error.support_foot_slip_m > 0.025:
            support = _move(support, delta=-min(0.14, error.support_foot_slip_m * 2.0))
            changed.append("support_ground_friction")
            reasons.append("observed_support_slip")

        ball_friction = belief.ball_ground_friction
        ball_mass = belief.ball_mass
        speed_residual = error.observed_ball_speed_mps - error.predicted_ball_speed_mps
        if abs(speed_residual) >= 0.15:
            ball_friction = _move(ball_friction, delta=-0.012 * speed_residual)
            ball_mass = _move(ball_mass, delta=-0.0025 * speed_residual)
            changed.extend(("ball_ground_friction", "ball_mass"))
            reasons.append("ball_speed_residual")

        latency = belief.control_latency_ms
        timing_residual = error.observed_contact_time_sec - error.predicted_contact_time_sec
        if abs(timing_residual) >= 0.015:
            latency = _move(latency, delta=180.0 * timing_residual)
            changed.append("control_latency_ms")
            reasons.append("contact_timing_residual")

        contact_mean = belief.foot_contact_offset_mean
        covariance = belief.foot_contact_offset_covariance
        if error.observed_target_error_m > error.predicted_target_error_m + 0.05:
            sign = 1.0 if timing_residual >= 0.0 else -1.0
            contact_mean = (
                max(-0.10, min(0.10, contact_mean[0] + sign * 0.004)),
                contact_mean[1],
            )
            covariance = (
                (max(0.0002, covariance[0][0] * 0.92), covariance[0][1]),
                (covariance[1][0], max(0.0002, covariance[1][1] * 0.96)),
            )
            changed.append("foot_contact_offset")
            reasons.append("landing_residual")

        if not changed:
            # An evidence-bearing calibration always contracts at least one
            # uncertainty, even when its mean is unchanged.
            support = replace(support, std=max(0.01, support.std * 0.98))
            changed.append("support_ground_friction_uncertainty")
            reasons.append("consistent_observation")

        child = KickTwinBelief(
            ball_mass=ball_mass,
            ball_ground_friction=ball_friction,
            support_ground_friction=support,
            restitution=belief.restitution,
            control_latency_ms=latency,
            foot_contact_offset_mean=contact_mean,
            foot_contact_offset_covariance=covariance,
            update_count=belief.update_count + 1,
            parent_hash=belief.belief_hash,
            evidence_hashes=(*belief.evidence_hashes[-31:], error.error_hash),
        )
        update = KickTwinUpdate(
            parent_belief_hash=belief.belief_hash,
            child_belief_hash=child.belief_hash,
            prediction_error_hash=error.error_hash,
            changed_parameters=tuple(dict.fromkeys(changed)),
            reasons=tuple(dict.fromkeys(reasons)),
        )
        return child, update


def _move(belief: ScalarBelief, *, delta: float) -> ScalarBelief:
    mean = max(belief.minimum, min(belief.maximum, belief.mean + delta))
    return replace(belief, mean=mean, std=max(1e-4, belief.std * 0.92))


__all__ = ["KickTwinEstimator"]
