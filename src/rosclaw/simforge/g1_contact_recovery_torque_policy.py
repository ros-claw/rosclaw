"""Contact-causal candidate overlay for G1 post-kick torque learning.

The retained hierarchical policy remains bit-for-bit responsible for the kick.
A candidate actor may replace it only after observed ball motion proves contact,
the robot remains upright, and a support foot is available.  This separates
shot execution from recovery credit assignment without adding a hardware path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from rosclaw.simforge.g1_neural_torque import (
    G1NeuralTorquePolicy,
    G1TorqueControlFrame,
    G1TorquePolicyReceipt,
    build_g1_neural_torque_observation,
)
from rosclaw.simforge.g1_stability_plasticity_policy import (
    G1StabilityPlasticityTorquePolicy,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class G1ContactRecoveryGateConfig:
    """Causal SIM-only envelope for a learned recovery candidate."""

    minimum_policy_phase: float = 0.40
    minimum_ball_speed_mps: float = 0.50
    minimum_pelvis_height_m: float = 0.70
    maximum_projected_gravity_z: float = -0.88
    eligibility_warmup_steps: int = 5
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if not 0.02 <= self.minimum_policy_phase <= 1.0:
            raise ValueError("contact recovery phase must be in [0.02, 1]")
        if not math.isfinite(self.minimum_ball_speed_mps) or not (
            0.05 <= self.minimum_ball_speed_mps <= 5.0
        ):
            raise ValueError("contact recovery ball-speed threshold must be in [0.05, 5]")
        if not 0.55 <= self.minimum_pelvis_height_m <= 0.95:
            raise ValueError("contact recovery pelvis height must be in [0.55, 0.95] m")
        if not -0.999 <= self.maximum_projected_gravity_z <= -0.75:
            raise ValueError("contact recovery gravity gate must be in [-0.999, -0.75]")
        if not 1 <= self.eligibility_warmup_steps <= 500:
            raise ValueError("contact recovery warmup must be in [1, 500]")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("contact recovery candidate is restricted to SIM_ONLY")


@dataclass(frozen=True)
class G1RecoveryContextSnapshot:
    """Proprioceptive self-state frozen at the first causal ball-contact frame."""

    policy_phase: float
    pelvis_height_m: float
    projected_gravity_x: float
    projected_gravity_y: float
    projected_gravity_z: float
    base_linear_velocity_x_mps: float
    base_linear_velocity_y_mps: float
    base_linear_velocity_z_mps: float
    base_angular_velocity_x_rps: float
    base_angular_velocity_y_rps: float
    base_angular_velocity_z_rps: float
    ball_speed_mps: float
    ball_direction_x: float
    ball_direction_y: float
    ball_direction_z: float
    left_contact: bool
    right_contact: bool
    schema_version: str = "rosclaw.simforge.g1_recovery_context.v1"

    def __post_init__(self) -> None:
        numeric = (
            self.policy_phase,
            self.pelvis_height_m,
            self.projected_gravity_x,
            self.projected_gravity_y,
            self.projected_gravity_z,
            self.base_linear_velocity_x_mps,
            self.base_linear_velocity_y_mps,
            self.base_linear_velocity_z_mps,
            self.base_angular_velocity_x_rps,
            self.base_angular_velocity_y_rps,
            self.base_angular_velocity_z_rps,
            self.ball_speed_mps,
            self.ball_direction_x,
            self.ball_direction_y,
            self.ball_direction_z,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("recovery context must contain only finite values")
        if not 0.0 <= self.policy_phase <= 1.0:
            raise ValueError("recovery context policy phase must be in [0, 1]")
        if not 0.0 < self.pelvis_height_m <= 2.0:
            raise ValueError("recovery context pelvis height must be in (0, 2] m")
        if self.ball_speed_mps <= 0.0:
            raise ValueError("recovery context requires observed ball motion")
        direction_norm = math.sqrt(
            self.ball_direction_x**2 + self.ball_direction_y**2 + self.ball_direction_z**2
        )
        if not math.isclose(direction_norm, 1.0, abs_tol=1e-5):
            raise ValueError("recovery context ball direction must be unit length")

    @classmethod
    def from_frame(
        cls,
        frame: G1TorqueControlFrame,
        *,
        ball_speed_mps: float,
    ) -> G1RecoveryContextSnapshot:
        observation = build_g1_neural_torque_observation(
            frame,
            np.zeros(29, dtype=np.float64),
        )
        ball_velocity = np.asarray(frame.ball_velocity, dtype=np.float64)
        direction = ball_velocity / ball_speed_mps
        return cls(
            policy_phase=float(frame.policy_phase),
            pelvis_height_m=float(frame.pelvis_position[2]),
            projected_gravity_x=float(observation[58]),
            projected_gravity_y=float(observation[59]),
            projected_gravity_z=float(observation[60]),
            base_linear_velocity_x_mps=float(observation[61]),
            base_linear_velocity_y_mps=float(observation[62]),
            base_linear_velocity_z_mps=float(observation[63]),
            base_angular_velocity_x_rps=float(observation[64]),
            base_angular_velocity_y_rps=float(observation[65]),
            base_angular_velocity_z_rps=float(observation[66]),
            ball_speed_mps=ball_speed_mps,
            ball_direction_x=float(direction[0]),
            ball_direction_y=float(direction[1]),
            ball_direction_z=float(direction[2]),
            left_contact=bool(frame.left_contact),
            right_contact=bool(frame.right_contact),
        )

    @property
    def context_hash(self) -> str:
        return hash_json(asdict(self))


class G1DelayedRecoveryContextObserver:
    """Read-only observer that freezes an observable post-impact self-state."""

    def __init__(
        self,
        *,
        minimum_ball_speed_mps: float = 0.50,
        delay_steps: int = 50,
    ) -> None:
        if not math.isfinite(minimum_ball_speed_mps) or minimum_ball_speed_mps <= 0.0:
            raise ValueError("recovery context observer speed threshold must be positive")
        if not 1 <= delay_steps <= 250:
            raise ValueError("recovery context observer delay must be in [1, 250] steps")
        self.minimum_ball_speed_mps = minimum_ball_speed_mps
        self.delay_steps = delay_steps
        self.reset()

    def reset(self) -> None:
        self._contact_steps = 0
        self._context: G1RecoveryContextSnapshot | None = None
        self._invalid_context = False

    def observe(self, frame: G1TorqueControlFrame, applied_torque: np.ndarray) -> None:
        torque = np.asarray(applied_torque, dtype=np.float64)
        if torque.shape != (29,) or not np.all(np.isfinite(torque)):
            self._invalid_context = True
            return
        ball_speed = float(np.linalg.norm(np.asarray(frame.ball_velocity, dtype=np.float64)))
        contact_ready = bool(
            frame.ball_contact_observed
            and math.isfinite(ball_speed)
            and ball_speed >= self.minimum_ball_speed_mps
        )
        if contact_ready:
            self._contact_steps += 1
        if self._context is None and self._contact_steps >= self.delay_steps:
            try:
                self._context = G1RecoveryContextSnapshot.from_frame(
                    frame,
                    ball_speed_mps=ball_speed,
                )
            except ValueError:
                self._invalid_context = True

    def context(self) -> G1RecoveryContextSnapshot:
        if self._invalid_context:
            raise ValueError("recovery context observer encountered invalid state")
        if self._context is None:
            raise ValueError("recovery context observer did not reach the delayed contact state")
        return self._context


@dataclass(frozen=True)
class G1ContactRecoveryReceipt(G1TorquePolicyReceipt):
    baseline_artifact_hash: str = ""
    candidate_artifact_hash: str = ""
    gate_hash: str = ""
    ball_contact_latched: bool = False
    contact_context: G1RecoveryContextSnapshot | None = None
    contact_context_hash: str | None = None
    candidate_activation_count: int = 0
    candidate_activation_fraction: float = 0.0
    phase_rejection_count: int = 0
    contact_rejection_count: int = 0
    posture_rejection_count: int = 0
    support_rejection_count: int = 0
    candidate_projection_rejection_count: int = 0
    router_artifact_hash: str | None = None
    routed_expert_id: str | None = None
    routed_expert_artifact_hash: str | None = None
    router_fallback_reason: str | None = None
    schema_version: str = "rosclaw.simforge.g1_contact_recovery_receipt.v3"


class G1ContactRecoveryTorquePolicy:
    """Overlay a recovery candidate only after a causal contact latch.

    Both policies receive every observation and the actually applied torque.
    Therefore an artifact identical to the retained recovery actor is a
    behaviorally exact no-op parent before online learning starts.
    """

    def __init__(
        self,
        baseline: G1StabilityPlasticityTorquePolicy,
        candidate: G1NeuralTorquePolicy,
        *,
        config: G1ContactRecoveryGateConfig | None = None,
    ) -> None:
        if baseline.stable.artifact.body_hash != candidate.artifact.body_hash:
            raise ValueError("contact recovery actors target different bodies")
        if baseline.stable.artifact.parent_policy_hash != candidate.artifact.parent_policy_hash:
            raise ValueError("contact recovery actors have different parents")
        self.baseline = baseline
        self.candidate = candidate
        self.config = config or G1ContactRecoveryGateConfig()
        self.artifact_hash = hash_json(
            {
                "baseline_artifact_hash": baseline.artifact_hash,
                "candidate_artifact_hash": candidate.artifact.artifact_hash,
                "gate": asdict(self.config),
            }
        )
        self.reset()

    def reset(self) -> None:
        self.baseline.reset()
        self.candidate.reset()
        self._pending = False
        self._contact_latched = False
        self._contact_context: G1RecoveryContextSnapshot | None = None
        self._eligible_steps = 0
        self._inference_count = 0
        self._activation_count = 0
        self._phase_rejections = 0
        self._contact_rejections = 0
        self._posture_rejections = 0
        self._support_rejections = 0
        self._projection_rejections = 0
        self._activation_mask: list[bool] = []

    def command(self, frame: G1TorqueControlFrame, parent_torque: np.ndarray) -> np.ndarray:
        if self._pending:
            raise RuntimeError("contact recovery policy did not receive note_applied")
        baseline_torque = self.baseline.command(frame, parent_torque)
        ball_speed = float(np.linalg.norm(np.asarray(frame.ball_velocity, dtype=np.float64)))
        contact_now = bool(
            frame.ball_contact_observed
            and math.isfinite(ball_speed)
            and ball_speed >= self.config.minimum_ball_speed_mps
        )
        if contact_now and not self._contact_latched:
            self._contact_latched = True
            try:
                self._contact_context = G1RecoveryContextSnapshot.from_frame(
                    frame,
                    ball_speed_mps=ball_speed,
                )
            except ValueError:
                # A malformed context can never become applicability evidence.
                # The existing posture/safety gates still fail closed below.
                self._contact_context = None
        phase_ok = frame.policy_phase >= self.config.minimum_policy_phase
        posture_ok = self._posture_eligible(frame)
        support_ok = bool(frame.left_contact or frame.right_contact)
        self._phase_rejections += int(not phase_ok)
        self._contact_rejections += int(phase_ok and not self._contact_latched)
        self._posture_rejections += int(phase_ok and self._contact_latched and not posture_ok)
        self._support_rejections += int(
            phase_ok and self._contact_latched and posture_ok and not support_ok
        )
        eligible = phase_ok and self._contact_latched and posture_ok and support_ok
        self._eligible_steps = self._eligible_steps + 1 if eligible else 0
        ready = self._eligible_steps >= self.config.eligibility_warmup_steps
        candidate_torque = self.candidate.command(
            frame,
            parent_torque,
            allow_exploration=ready,
        )
        projection_ok = self.candidate.pending_projection.reason is None
        active = bool(ready and projection_ok)
        self._projection_rejections += int(ready and not projection_ok)
        self._activation_count += int(active)
        self._activation_mask.append(active)
        self._inference_count += 1
        self._pending = True
        return (candidate_torque if active else baseline_torque).copy()

    def note_applied(self, torque: np.ndarray) -> None:
        if not self._pending:
            raise RuntimeError("contact recovery policy has no pending command")
        self.baseline.note_applied(torque)
        self.candidate.note_applied(torque)
        self._pending = False

    def candidate_activation_mask(self) -> np.ndarray:
        if self._pending or not self._activation_mask:
            raise ValueError("contact recovery activation mask is empty or incomplete")
        return np.asarray(self._activation_mask, dtype=np.bool_)

    def build_receipt(self) -> G1ContactRecoveryReceipt:
        if self._pending or not self._inference_count:
            raise ValueError("cannot build an incomplete contact recovery receipt")
        baseline = self.baseline.build_receipt()
        candidate = self.candidate.build_receipt()
        route = getattr(candidate, "route", None)
        # Baseline fallback totals are deliberately retained as a conservative
        # upper bound: the unused baseline branch is still evaluated while the
        # candidate is active, and under-reporting fallback is worse than
        # counting one that did not reach the plant.
        fallback_count = min(
            self._inference_count,
            baseline.fallback_count + self._projection_rejections,
        )
        return G1ContactRecoveryReceipt(
            artifact_hash=self.artifact_hash,
            body_hash=candidate.body_hash,
            parent_policy_hash=candidate.parent_policy_hash,
            inference_count=self._inference_count,
            learned_output_count=self._inference_count - fallback_count,
            fallback_count=fallback_count,
            nonfinite_fallback_count=baseline.nonfinite_fallback_count,
            projection_fallback_count=(
                baseline.projection_fallback_count + self._projection_rejections
            ),
            out_of_distribution_fallback_count=baseline.out_of_distribution_fallback_count,
            warmup_fallback_count=baseline.warmup_fallback_count,
            state_guard_fallback_count=baseline.state_guard_fallback_count,
            projected_joint_count=(
                baseline.projected_joint_count + candidate.projected_joint_count
            ),
            maximum_limit_ratio=max(
                baseline.maximum_limit_ratio,
                candidate.maximum_limit_ratio,
            ),
            peak_mechanical_power_w=max(
                baseline.peak_mechanical_power_w,
                candidate.peak_mechanical_power_w,
            ),
            direct_torque_output=True,
            activation_ceiling="SIM_ONLY",
            exploration_config_hash=candidate.exploration_config_hash,
            exploration_attempt_count=candidate.exploration_attempt_count,
            exploration_applied_count=candidate.exploration_applied_count,
            exploration_rejection_count=candidate.exploration_rejection_count,
            exploration_noise_rms_ratio=candidate.exploration_noise_rms_ratio,
            exploration_noise_peak_ratio=candidate.exploration_noise_peak_ratio,
            baseline_artifact_hash=self.baseline.artifact_hash,
            candidate_artifact_hash=self.candidate.artifact.artifact_hash,
            gate_hash=hash_json(asdict(self.config)),
            ball_contact_latched=self._contact_latched,
            contact_context=self._contact_context,
            contact_context_hash=(
                self._contact_context.context_hash if self._contact_context is not None else None
            ),
            candidate_activation_count=self._activation_count,
            candidate_activation_fraction=(self._activation_count / self._inference_count),
            phase_rejection_count=self._phase_rejections,
            contact_rejection_count=self._contact_rejections,
            posture_rejection_count=self._posture_rejections,
            support_rejection_count=self._support_rejections,
            candidate_projection_rejection_count=self._projection_rejections,
            router_artifact_hash=getattr(candidate, "router_artifact_hash", None),
            routed_expert_id=getattr(candidate, "selected_expert_id", None),
            routed_expert_artifact_hash=getattr(
                candidate,
                "selected_expert_artifact_hash",
                None,
            ),
            router_fallback_reason=(
                getattr(route, "fallback_reason", None) if route is not None else None
            ),
        )

    def _posture_eligible(self, frame: G1TorqueControlFrame) -> bool:
        try:
            observation = build_g1_neural_torque_observation(
                frame,
                np.zeros(29, dtype=np.float64),
            )
        except ValueError:
            return False
        gravity_z = float(observation[60])
        return bool(
            math.isfinite(gravity_z)
            and gravity_z <= self.config.maximum_projected_gravity_z
            and float(frame.pelvis_position[2]) >= self.config.minimum_pelvis_height_m
        )


__all__ = [
    "G1ContactRecoveryGateConfig",
    "G1ContactRecoveryReceipt",
    "G1ContactRecoveryTorquePolicy",
    "G1DelayedRecoveryContextObserver",
    "G1RecoveryContextSnapshot",
]
