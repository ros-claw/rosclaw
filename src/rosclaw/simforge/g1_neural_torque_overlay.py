"""Contact-causal trust overlay for neural muscle memory on a strong parent.

The target-space controller remains the physical parent.  A recurrent neural
actor runs in shadow, then contributes only a bounded, smoothly ramped torque
residual after MuJoCo has observed ball contact and the body remains upright.
This gives teacher distillation and online RL a safe bridge without asking an
immature actor to replace the complete whole-body controller in one step.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from rosclaw.simforge.g1_contact_recovery_torque_policy import G1RecoveryContextSnapshot
from rosclaw.simforge.g1_neural_torque import (
    G1NeuralTorquePolicy,
    G1TeacherTorqueEpisode,
    G1TorqueControlFrame,
    G1TorquePolicyReceipt,
    build_g1_neural_torque_observation,
)
from rosclaw.simforge.g1_recovery_expert_router import (
    G1RecoveryExpertRoute,
    G1RecoveryExpertRouterArtifact,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class G1NeuralTorqueOverlayConfig:
    """Fail-closed envelope around a distilled neural torque residual."""

    trust_fraction: float = 0.05
    minimum_policy_phase: float = 0.40
    minimum_ball_speed_mps: float = 0.50
    minimum_pelvis_height_m: float = 0.70
    maximum_projected_gravity_z: float = -0.88
    eligibility_warmup_steps: int = 5
    ramp_steps: int = 250
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if not math.isfinite(self.trust_fraction) or not 0.0 < self.trust_fraction <= 0.25:
            raise ValueError("neural torque overlay trust must be in (0, 0.25]")
        if not 0.02 <= self.minimum_policy_phase <= 1.0:
            raise ValueError("neural torque overlay phase must be in [0.02, 1]")
        if not math.isfinite(self.minimum_ball_speed_mps) or not (
            0.05 <= self.minimum_ball_speed_mps <= 5.0
        ):
            raise ValueError("neural torque overlay ball-speed gate must be in [0.05, 5]")
        if not 0.55 <= self.minimum_pelvis_height_m <= 0.95:
            raise ValueError("neural torque overlay pelvis gate must be in [0.55, 0.95] m")
        if not -0.999 <= self.maximum_projected_gravity_z <= -0.75:
            raise ValueError("neural torque overlay gravity gate must be in [-0.999, -0.75]")
        if not 1 <= self.eligibility_warmup_steps <= 500:
            raise ValueError("neural torque overlay warmup must be in [1, 500]")
        if not 1 <= self.ramp_steps <= 2_500:
            raise ValueError("neural torque overlay ramp must be in [1, 2500] steps")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("neural torque overlay is restricted to SIM_ONLY")


@dataclass(frozen=True)
class G1NeuralTorqueOverlayReceipt(G1TorquePolicyReceipt):
    candidate_artifact_hash: str = ""
    config_hash: str = ""
    ball_contact_latched: bool = False
    activation_count: int = 0
    activation_fraction: float = 0.0
    maximum_applied_trust_fraction: float = 0.0
    maximum_residual_ratio: float = 0.0
    phase_rejection_count: int = 0
    contact_rejection_count: int = 0
    posture_rejection_count: int = 0
    support_rejection_count: int = 0
    context_rejection_count: int = 0
    projection_rejection_count: int = 0
    exploration_enabled: bool = False
    applicability_artifact_hash: str | None = None
    applicability_expert_id: str | None = None
    applicability_route: G1RecoveryExpertRoute | None = None
    applicability_context: G1RecoveryContextSnapshot | None = None
    applicability_context_hash: str | None = None
    applicability_delay_steps: int = 0
    schema_version: str = "rosclaw.simforge.g1_neural_torque_overlay_receipt.v3"


@dataclass(frozen=True)
class G1NeuralTorqueOverlayEpisode:
    """Credit-assignment trace for the action proposed before trust blending."""

    policy_episode: G1TeacherTorqueEpisode
    applied_actions: np.ndarray
    trust_fractions: np.ndarray
    activation_mask: np.ndarray
    schema_version: str = "rosclaw.simforge.g1_neural_torque_overlay_episode.v1"

    def __post_init__(self) -> None:
        rows = len(self.policy_episode.actions)
        applied = np.asarray(self.applied_actions, dtype=np.float32)
        trusts = np.asarray(self.trust_fractions, dtype=np.float32)
        mask = np.asarray(self.activation_mask)
        binary_mask = mask.astype(np.bool_)
        if applied.shape != self.policy_episode.actions.shape:
            raise ValueError("overlay applied actions are misaligned")
        if trusts.shape != (rows,) or mask.shape != (rows,):
            raise ValueError("overlay credit-assignment arrays are misaligned")
        if not np.all(np.isfinite(applied)) or not np.all(np.isfinite(trusts)):
            raise ValueError("overlay credit-assignment arrays must be finite")
        if np.any((trusts < 0.0) | (trusts > 0.25)):
            raise ValueError("overlay trust fractions must be in [0, 0.25]")
        if mask.dtype.kind not in {"b", "i", "u"} or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("overlay activation mask must be binary")
        if np.any((trusts > 0.0) != binary_mask):
            raise ValueError("overlay trust and activation mask disagree")
        for value in (applied, trusts, binary_mask):
            value.setflags(write=False)
        object.__setattr__(self, "applied_actions", applied)
        object.__setattr__(self, "trust_fractions", trusts)
        object.__setattr__(self, "activation_mask", binary_mask)


class G1NeuralTorqueOverlayPolicy:
    """Blend a guarded candidate into the target-space parent after contact."""

    def __init__(
        self,
        candidate: G1NeuralTorquePolicy,
        *,
        config: G1NeuralTorqueOverlayConfig | None = None,
        exploration_enabled: bool = False,
        applicability_router: G1RecoveryExpertRouterArtifact | None = None,
        applicability_expert_id: str | None = None,
        applicability_delay_steps: int = 50,
    ) -> None:
        if exploration_enabled and candidate.exploration is None:
            raise ValueError("overlay exploration requires a configured candidate explorer")
        self.candidate = candidate
        self.config = config or G1NeuralTorqueOverlayConfig()
        self.exploration_enabled = bool(exploration_enabled)
        if applicability_router is None:
            if applicability_expert_id is not None:
                raise ValueError("overlay applicability expert requires a router")
        else:
            if not applicability_expert_id:
                raise ValueError("overlay applicability router requires an expert id")
            expected = dict(applicability_router.expert_artifact_hashes)
            if expected != {applicability_expert_id: candidate.artifact.artifact_hash}:
                raise ValueError("overlay applicability router does not bind the candidate")
            if applicability_router.body_hash != candidate.artifact.body_hash:
                raise ValueError("overlay applicability router body mismatch")
            if applicability_router.parent_policy_hash != candidate.artifact.parent_policy_hash:
                raise ValueError("overlay applicability router parent mismatch")
        if not 1 <= applicability_delay_steps <= 250:
            raise ValueError("overlay applicability delay must be in [1, 250] steps")
        self.applicability_router = applicability_router
        self.applicability_expert_id = applicability_expert_id
        self.applicability_delay_steps = applicability_delay_steps
        self.artifact_hash = hash_json(
            {
                "candidate_artifact_hash": candidate.artifact.artifact_hash,
                "config": asdict(self.config),
                "exploration_enabled": self.exploration_enabled,
                "applicability_artifact_hash": (
                    applicability_router.artifact_hash if applicability_router else None
                ),
                "applicability_expert_id": applicability_expert_id,
                "applicability_delay_steps": applicability_delay_steps,
            }
        )
        self.reset()

    def reset(self) -> None:
        self.candidate.reset()
        self._pending = False
        self._contact_latched = False
        self._eligible_steps = 0
        self._inference_count = 0
        self._activation_count = 0
        self._maximum_trust = 0.0
        self._maximum_residual_ratio = 0.0
        self._phase_rejections = 0
        self._contact_rejections = 0
        self._posture_rejections = 0
        self._support_rejections = 0
        self._context_rejections = 0
        self._projection_rejections = 0
        self._contact_steps = 0
        self._applicability_route: G1RecoveryExpertRoute | None = None
        self._applicability_context: G1RecoveryContextSnapshot | None = None
        self._pending_proposal: np.ndarray | None = None
        self._pending_parent: np.ndarray | None = None
        self._pending_trust = 0.0
        self._pending_active = False
        self._proposals: list[np.ndarray] = []
        self._parents: list[np.ndarray] = []
        self._applied: list[np.ndarray] = []
        self._trusts: list[float] = []
        self._active: list[bool] = []

    def command(self, frame: G1TorqueControlFrame, parent_torque: np.ndarray) -> np.ndarray:
        if self._pending:
            raise RuntimeError("neural torque overlay did not receive note_applied")
        parent = np.asarray(parent_torque, dtype=np.float64)
        ball_speed = float(np.linalg.norm(np.asarray(frame.ball_velocity, dtype=np.float64)))
        contact_now = bool(
            frame.ball_contact_observed
            and math.isfinite(ball_speed)
            and ball_speed >= self.config.minimum_ball_speed_mps
        )
        if contact_now:
            self._contact_latched = True
            self._contact_steps += 1
        if (
            self.applicability_router is not None
            and self._applicability_route is None
            and self._contact_steps >= self.applicability_delay_steps
        ):
            try:
                context = G1RecoveryContextSnapshot.from_frame(
                    frame,
                    ball_speed_mps=ball_speed,
                )
                self._applicability_context = context
                self._applicability_route = self.applicability_router.route(context)
            except ValueError:
                self._applicability_route = G1RecoveryExpertRoute(
                    context_hash="",
                    expert_id=None,
                    normalized_distance=math.inf,
                    expected_naturalness_gain=0.0,
                    eligible=False,
                    fallback_reason="invalid_contact_context",
                )
        phase_ok = frame.policy_phase >= self.config.minimum_policy_phase
        posture_ok = self._posture_eligible(frame)
        support_ok = bool(frame.left_contact or frame.right_contact)
        context_ok = bool(
            self.applicability_router is None
            or (
                self._applicability_route is not None
                and self._applicability_route.eligible
                and self._applicability_route.expert_id == self.applicability_expert_id
            )
        )
        proposal = self.candidate.command(
            frame,
            parent,
            allow_exploration=bool(
                self.exploration_enabled
                and phase_ok
                and self._contact_latched
                and posture_ok
                and support_ok
                and context_ok
            ),
        )
        projection_ok = self.candidate.pending_projection.reason is None
        self._phase_rejections += int(not phase_ok)
        self._contact_rejections += int(phase_ok and not self._contact_latched)
        self._posture_rejections += int(phase_ok and self._contact_latched and not posture_ok)
        self._support_rejections += int(
            phase_ok and self._contact_latched and posture_ok and not support_ok
        )
        self._context_rejections += int(
            phase_ok and self._contact_latched and posture_ok and support_ok and not context_ok
        )
        eligible = phase_ok and self._contact_latched and posture_ok and support_ok and context_ok
        self._eligible_steps = self._eligible_steps + 1 if eligible else 0
        ready = self._eligible_steps >= self.config.eligibility_warmup_steps
        self._projection_rejections += int(ready and not projection_ok)
        active = bool(ready and projection_ok)
        if active:
            ramp_progress = min(
                1.0,
                (self._eligible_steps - self.config.eligibility_warmup_steps + 1)
                / self.config.ramp_steps,
            )
            smooth = ramp_progress * ramp_progress * (3.0 - 2.0 * ramp_progress)
            trust = self.config.trust_fraction * smooth
            command = parent + trust * (np.asarray(proposal, dtype=np.float64) - parent)
            limits = np.asarray(self.candidate.artifact.action_limits, dtype=np.float64)
            residual_ratio = float(np.max(np.abs(command - parent) / limits))
            self._activation_count += 1
            self._maximum_trust = max(self._maximum_trust, trust)
            self._maximum_residual_ratio = max(self._maximum_residual_ratio, residual_ratio)
        else:
            command = parent.copy()
        self._inference_count += 1
        self._pending = True
        self._pending_proposal = np.asarray(proposal, dtype=np.float32).copy()
        self._pending_parent = np.asarray(parent, dtype=np.float32).copy()
        self._pending_trust = trust if active else 0.0
        self._pending_active = active
        return np.asarray(command, dtype=np.float64).copy()

    def note_applied(self, torque: np.ndarray) -> None:
        if not self._pending:
            raise RuntimeError("neural torque overlay has no pending command")
        applied = np.asarray(torque, dtype=np.float64)
        self.candidate.note_applied(applied)
        if self._pending_proposal is None or self._pending_parent is None:
            raise RuntimeError("neural torque overlay credit assignment is incomplete")
        self._proposals.append(self._pending_proposal)
        self._parents.append(self._pending_parent)
        self._applied.append(np.asarray(applied, dtype=np.float32).copy())
        self._trusts.append(self._pending_trust)
        self._active.append(self._pending_active)
        self._pending_proposal = None
        self._pending_parent = None
        self._pending_trust = 0.0
        self._pending_active = False
        self._pending = False

    def policy_episode(self) -> G1NeuralTorqueOverlayEpisode:
        """Return proposals, not blended torques, for valid actor credit assignment."""

        if self._pending or not self._proposals:
            raise ValueError("neural torque overlay episode is empty or incomplete")
        observed = self.candidate.episode()
        return G1NeuralTorqueOverlayEpisode(
            policy_episode=G1TeacherTorqueEpisode(
                observations=observed.observations,
                actions=np.asarray(self._proposals, dtype=np.float32),
                parent_actions=np.asarray(self._parents, dtype=np.float32),
            ),
            applied_actions=np.asarray(self._applied, dtype=np.float32),
            trust_fractions=np.asarray(self._trusts, dtype=np.float32),
            activation_mask=np.asarray(self._active, dtype=np.bool_),
        )

    def build_receipt(self) -> G1NeuralTorqueOverlayReceipt:
        if self._pending or not self._inference_count:
            raise ValueError("cannot build an incomplete neural torque overlay receipt")
        candidate = self.candidate.build_receipt()
        return G1NeuralTorqueOverlayReceipt(
            artifact_hash=self.artifact_hash,
            body_hash=candidate.body_hash,
            parent_policy_hash=candidate.parent_policy_hash,
            inference_count=self._inference_count,
            learned_output_count=self._activation_count,
            fallback_count=self._inference_count - self._activation_count,
            nonfinite_fallback_count=candidate.nonfinite_fallback_count,
            projection_fallback_count=candidate.projection_fallback_count,
            out_of_distribution_fallback_count=candidate.out_of_distribution_fallback_count,
            warmup_fallback_count=candidate.warmup_fallback_count,
            state_guard_fallback_count=candidate.state_guard_fallback_count,
            projected_joint_count=candidate.projected_joint_count,
            maximum_limit_ratio=candidate.maximum_limit_ratio,
            peak_mechanical_power_w=candidate.peak_mechanical_power_w,
            direct_torque_output=True,
            activation_ceiling="SIM_ONLY",
            candidate_artifact_hash=self.candidate.artifact.artifact_hash,
            config_hash=hash_json(asdict(self.config)),
            ball_contact_latched=self._contact_latched,
            activation_count=self._activation_count,
            activation_fraction=self._activation_count / self._inference_count,
            maximum_applied_trust_fraction=self._maximum_trust,
            maximum_residual_ratio=self._maximum_residual_ratio,
            phase_rejection_count=self._phase_rejections,
            contact_rejection_count=self._contact_rejections,
            posture_rejection_count=self._posture_rejections,
            support_rejection_count=self._support_rejections,
            context_rejection_count=self._context_rejections,
            projection_rejection_count=self._projection_rejections,
            exploration_enabled=self.exploration_enabled,
            applicability_artifact_hash=(
                self.applicability_router.artifact_hash if self.applicability_router else None
            ),
            applicability_expert_id=self.applicability_expert_id,
            applicability_route=self._applicability_route,
            applicability_context=self._applicability_context,
            applicability_context_hash=(
                self._applicability_context.context_hash
                if self._applicability_context is not None
                else None
            ),
            applicability_delay_steps=(
                self.applicability_delay_steps if self.applicability_router else 0
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
    "G1NeuralTorqueOverlayConfig",
    "G1NeuralTorqueOverlayEpisode",
    "G1NeuralTorqueOverlayPolicy",
    "G1NeuralTorqueOverlayReceipt",
]
