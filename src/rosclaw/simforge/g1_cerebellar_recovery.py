"""Contact-gated post-kick recovery motion for the G1 GoalForge sandbox.

The qualified RoboNaldo policy remains responsible for the kick.  This module
only reshapes the late recovery segment after observed ball contact and kick-
foot landing.  It never produces torque commands or opens a robot transport.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from rosclaw.simforge.g1_contextual_recovery import (
    G1ContextualRecoveryArtifact,
    G1ContextualRecoveryPolicy,
    G1ContextualRecoveryPrimitive,
)
from rosclaw.simforge.g1_muscle_memory import (
    G1MuscleMemoryArtifact,
    G1MuscleMemoryPolicy,
)
from rosclaw.simforge.g1_recovery_state_memory import (
    G1RecoveryStateArtifact,
    G1RecoveryStatePolicy,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    hash_bytes,
    hash_json,
)

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class G1CerebellarRecoveryConfig:
    """Bounded target-space recovery segment discovered by matched SIM A/B.

    The first smoothstep unloads the kick, the second settles into a more
    upright posture, and an optional third envelope adds bounded terminal PD
    damping.  Every stage is contact- and landing-gated, so the kick itself is
    never modified.
    """

    start_policy_frame: int = 420
    blend_frames: int = 100
    standing_pose_blend: float = 0.30
    roll_posture_bias_rad: float = -0.05
    settling_start_policy_frame: int | None = None
    settling_blend_frames: int = 80
    settling_standing_pose_blend: float | None = None
    settling_roll_posture_bias_rad: float | None = None
    settling_waist_pitch_bias_rad: float = 0.0
    target_smoothing_alpha: float = 1.0
    target_smoothing_start_policy_frame: int = 400
    target_smoothing_joint_group: str = "upper_body"
    terminal_damping_start_policy_frame: int | None = None
    terminal_damping_blend_frames: int = 80
    terminal_kp_scale: float = 1.0
    terminal_kd_scale: float = 1.0
    terminal_damping_joint_group: str = "whole_body"
    contact_required: bool = True
    kick_foot_landing_required: bool = True
    minimum_calibrated_support_friction: float = 0.95
    maximum_calibrated_control_latency_ms: float = 5.0
    minimum_calibrated_disturbance_n: float = 70.0
    maximum_calibrated_disturbance_n: float = 80.0

    def __post_init__(self) -> None:
        if self.start_policy_frame < 0:
            raise ValueError("start_policy_frame must be non-negative")
        if self.blend_frames <= 0:
            raise ValueError("blend_frames must be positive")
        if not 0.0 <= self.standing_pose_blend <= 0.50:
            raise ValueError("standing_pose_blend must be in [0, 0.50]")
        if not math.isfinite(self.roll_posture_bias_rad) or not (
            -0.08 <= self.roll_posture_bias_rad <= 0.08
        ):
            raise ValueError("roll_posture_bias_rad must be finite and in [-0.08, 0.08]")
        settling_values = (
            self.settling_start_policy_frame,
            self.settling_standing_pose_blend,
            self.settling_roll_posture_bias_rad,
        )
        if any(value is None for value in settling_values) and not all(
            value is None for value in settling_values
        ):
            raise ValueError("settling recovery parameters must be all set or all disabled")
        if self.settling_blend_frames <= 0:
            raise ValueError("settling_blend_frames must be positive")
        if self.settling_start_policy_frame is not None:
            if (
                self.settling_standing_pose_blend is None
                or self.settling_roll_posture_bias_rad is None
            ):
                raise ValueError("enabled settling recovery requires complete parameters")
            if self.settling_start_policy_frame < self.start_policy_frame + self.blend_frames:
                raise ValueError("settling recovery cannot overlap the unloading blend")
            if not 0.0 <= self.settling_standing_pose_blend <= 0.50:
                raise ValueError("settling_standing_pose_blend must be in [0, 0.50]")
            settling_roll = self.settling_roll_posture_bias_rad
            if not math.isfinite(settling_roll) or not -0.08 <= settling_roll <= 0.08:
                raise ValueError(
                    "settling_roll_posture_bias_rad must be finite and in [-0.08, 0.08]"
                )
        if not math.isfinite(self.settling_waist_pitch_bias_rad) or not (
            -0.12 <= self.settling_waist_pitch_bias_rad <= 0.12
        ):
            raise ValueError("settling_waist_pitch_bias_rad must be finite and in [-0.12, 0.12]")
        if self.settling_start_policy_frame is None and self.settling_waist_pitch_bias_rad != 0.0:
            raise ValueError("settling waist pitch bias requires settling recovery")
        if not 0.25 <= self.target_smoothing_alpha <= 1.0:
            raise ValueError("target_smoothing_alpha must be in [0.25, 1.0]")
        if self.target_smoothing_start_policy_frame < 0:
            raise ValueError("target_smoothing_start_policy_frame must be non-negative")
        if self.target_smoothing_joint_group not in {"upper_body", "arms"}:
            raise ValueError("target_smoothing_joint_group must be upper_body or arms")
        if self.terminal_damping_blend_frames <= 0:
            raise ValueError("terminal_damping_blend_frames must be positive")
        if self.terminal_damping_start_policy_frame is None:
            if self.terminal_kp_scale != 1.0 or self.terminal_kd_scale != 1.0:
                raise ValueError("terminal gain scaling requires terminal damping")
        else:
            earliest = self.start_policy_frame + self.blend_frames
            if self.settling_start_policy_frame is not None:
                earliest = max(
                    earliest,
                    self.settling_start_policy_frame + self.settling_blend_frames,
                )
            if self.terminal_damping_start_policy_frame < earliest:
                raise ValueError("terminal damping cannot overlap recovery blending")
        if not 0.75 <= self.terminal_kp_scale <= 1.0:
            raise ValueError("terminal_kp_scale must be in [0.75, 1.0]")
        if not 1.0 <= self.terminal_kd_scale <= 2.0:
            raise ValueError("terminal_kd_scale must be in [1.0, 2.0]")
        if self.terminal_damping_joint_group not in {"whole_body", "legs", "upper_body"}:
            raise ValueError("terminal_damping_joint_group must be whole_body, legs, or upper_body")
        if not 0.0 < self.minimum_calibrated_support_friction <= 2.0:
            raise ValueError("minimum calibrated support friction must be in (0, 2]")
        if not 0.0 <= self.maximum_calibrated_control_latency_ms <= 100.0:
            raise ValueError("maximum calibrated control latency must be in [0, 100]")
        if not (
            0.0 < self.minimum_calibrated_disturbance_n <= self.maximum_calibrated_disturbance_n
        ):
            raise ValueError("calibrated disturbance bounds must be positive and ordered")


def shared_post_impact_recovery_config() -> G1CerebellarRecoveryConfig:
    """Return the trained recovery contract shared by football roles.

    Keeping the contract beside the controller prevents pass, shot and
    long-run-up entry points from silently drifting to separate "cerebella".
    """

    return G1CerebellarRecoveryConfig(
        start_policy_frame=280,
        blend_frames=80,
        standing_pose_blend=0.02,
        roll_posture_bias_rad=0.0,
        target_smoothing_alpha=0.60,
        target_smoothing_start_policy_frame=280,
        target_smoothing_joint_group="upper_body",
    )


def _recovery_config_hash(config: G1CerebellarRecoveryConfig) -> str:
    return hash_json(
        {
            "schema_version": "rosclaw.g1_goalforge.cerebellar_recovery_config.v2",
            "config": asdict(config),
        }
    )


@dataclass(frozen=True)
class G1CerebellarRecoveryEffect:
    target: np.ndarray
    active: bool
    blend_fraction: float
    settling_fraction: float
    contact_latched: bool
    kick_foot_landing_latched: bool
    smoothing_active: bool
    smoothing_residual_rms_rad: float
    terminal_damping_fraction: float
    terminal_kp_scale: float
    terminal_kd_scale: float
    terminal_damping_joint_group: str
    muscle_memory_active: bool
    muscle_memory_out_of_distribution: bool
    muscle_memory_residual_rms_rad: float
    muscle_memory_synergy_actions: np.ndarray
    contextual_recovery_active: bool
    contextual_recovery_out_of_distribution: bool
    contextual_recovery_primitive_index: int | None
    recovery_state_active: bool
    recovery_state_pending: bool
    recovery_state_out_of_distribution: bool
    recovery_state_primitive_index: int | None


@dataclass(frozen=True)
class G1CerebellarRecoveryReceipt:
    controller_hash: str
    config_hash: str
    body_hash: str
    motion_hash: str
    standing_pose_hash: str
    regime_commitment: str
    regime_eligible: bool
    regime_reasons: tuple[str, ...]
    contact_latched: bool
    kick_foot_landing_latched: bool
    activation_policy_frame: int | None
    activation_time_sec: float | None
    smoothing_activation_policy_frame: int | None
    smoothing_activation_time_sec: float | None
    settling_activation_policy_frame: int | None
    settling_activation_time_sec: float | None
    terminal_damping_activation_policy_frame: int | None
    terminal_damping_activation_time_sec: float | None
    peak_blend_fraction: float
    peak_settling_fraction: float
    peak_terminal_damping_fraction: float
    peak_smoothing_residual_rms_rad: float
    strict_replay: bool
    evidence_domain: str
    config: dict[str, Any]
    fallback_config_hash: str | None
    fallback_config: dict[str, Any] | None
    fallback_routed_count: int
    expert_route_latched: bool | None
    muscle_memory_receipt: dict[str, Any] | None = None
    contextual_recovery_receipt: dict[str, Any] | None = None
    recovery_state_receipt: dict[str, Any] | None = None
    schema_version: str = "rosclaw.g1_goalforge.cerebellar_recovery_receipt.v9"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["regime_reasons"] = list(self.regime_reasons)
        return value


class G1CerebellarRecoveryController:
    """Blend the late kick target toward a qualified, damped standing pose.

    The controller is stateful only for causal gates and evidence.  It is
    transparent until ball contact and a subsequent right-foot landing are
    observed.  The existing high-rate balance reflex remains the closed-loop
    disturbance layer; this controller supplies the slower recovery segment.
    """

    def __init__(
        self,
        *,
        body_hash: str,
        motion_hash: str,
        regime_commitment: str,
        regime_eligible: bool,
        regime_reasons: tuple[str, ...],
        standing_pose: np.ndarray,
        config: G1CerebellarRecoveryConfig | None = None,
        muscle_memory_artifact: G1MuscleMemoryArtifact | None = None,
        contextual_recovery_artifact: G1ContextualRecoveryArtifact | None = None,
        recovery_state_artifact: G1RecoveryStateArtifact | None = None,
        fallback_config: G1CerebellarRecoveryConfig | None = None,
    ) -> None:
        for label, value in (
            ("body_hash", body_hash),
            ("motion_hash", motion_hash),
            ("regime_commitment", regime_commitment),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        pose = np.asarray(standing_pose, dtype=np.float64)
        if pose.shape != (len(G1_DDS_JOINT_NAMES),) or not np.all(np.isfinite(pose)):
            raise ValueError("standing_pose must be a finite 29-joint G1 target")
        self.body_hash = body_hash
        self.motion_hash = motion_hash
        self.regime_commitment = regime_commitment
        self.regime_eligible = bool(regime_eligible)
        self.regime_reasons = tuple(str(reason) for reason in regime_reasons)
        if self.regime_eligible == bool(self.regime_reasons):
            raise ValueError("eligible regimes must have no rejection reasons")
        self.standing_pose = pose.copy()
        self.standing_pose.setflags(write=False)
        self.standing_pose_hash = hash_bytes(np.ascontiguousarray(pose).tobytes())
        self.config = config or G1CerebellarRecoveryConfig()
        self.fallback_config = fallback_config
        self._roll_pattern = _roll_posture_pattern()
        self._waist_pitch_pattern = _waist_pitch_pattern()
        self.muscle_memory = (
            G1MuscleMemoryPolicy(muscle_memory_artifact)
            if muscle_memory_artifact is not None
            else None
        )
        self.contextual_recovery = (
            G1ContextualRecoveryPolicy(contextual_recovery_artifact)
            if contextual_recovery_artifact is not None
            else None
        )
        self.recovery_state = (
            G1RecoveryStatePolicy(recovery_state_artifact)
            if recovery_state_artifact is not None
            else None
        )
        learned_controllers = sum(
            controller is not None
            for controller in (
                self.muscle_memory,
                self.contextual_recovery,
                self.recovery_state,
            )
        )
        if learned_controllers > 1:
            raise ValueError(
                "only one muscle-memory, contextual, or recovery-state policy may be active"
            )
        if self.muscle_memory is not None:
            if (
                not self.muscle_memory.artifact.schema_version.endswith(".v1")
                and self.fallback_config is None
            ):
                raise ValueError("temporal muscle memory requires a fallback recovery config")
            self.muscle_memory.require_compatible(
                body_hash=self.body_hash,
                motion_hash=self.motion_hash,
                parent_recovery_config_hash=self.config_hash,
                fallback_recovery_config_hash=(self.fallback_config_hash or ""),
            )
            if not self.muscle_memory.artifact.schema_version.endswith(".v1"):
                structured = (
                    float(self.config.settling_standing_pose_blend or 0.0),
                    self.config.settling_waist_pitch_bias_rad,
                    self.config.target_smoothing_alpha,
                )
                if structured != self.muscle_memory.artifact.structured_recovery_parameters:
                    raise ValueError("muscle-memory structured recovery parameters mismatch")
        if self.contextual_recovery is not None:
            if self.fallback_config is None:
                raise ValueError("contextual recovery requires a fallback recovery config")
            self.contextual_recovery.require_compatible(
                body_hash=self.body_hash,
                motion_hash=self.motion_hash,
                baseline_recovery_config_hash=self.config_hash,
                fallback_recovery_config_hash=self.fallback_config_hash or "",
            )
        if self.recovery_state is not None:
            if self.fallback_config is None:
                raise ValueError("recovery-state memory requires a fallback recovery config")
            self.recovery_state.require_compatible(
                body_hash=self.body_hash,
                motion_hash=self.motion_hash,
                baseline_recovery_config_hash=self.config_hash,
                fallback_recovery_config_hash=self.fallback_config_hash or "",
            )
        self.reset()

    @property
    def config_hash(self) -> str:
        return _recovery_config_hash(self.config)

    @property
    def fallback_config_hash(self) -> str | None:
        if self.fallback_config is None:
            return None
        return _recovery_config_hash(self.fallback_config)

    @property
    def base_controller_hash(self) -> str:
        return hash_json(
            {
                "controller_type": "g1_cerebellar_post_kick_recovery",
                "version": 5,
                "body_hash": self.body_hash,
                "motion_hash": self.motion_hash,
                "standing_pose_hash": self.standing_pose_hash,
                "regime_commitment": self.regime_commitment,
                "regime_eligible": self.regime_eligible,
                "regime_reasons": list(self.regime_reasons),
                "config": asdict(self.config),
                "fallback_config": (
                    asdict(self.fallback_config) if self.fallback_config is not None else None
                ),
            }
        )

    @property
    def controller_hash(self) -> str:
        if (
            self.muscle_memory is None
            and self.contextual_recovery is None
            and self.recovery_state is None
        ):
            return self.base_controller_hash
        if self.recovery_state is not None:
            return hash_json(
                {
                    "controller_type": (
                        "g1_cerebellar_post_kick_recovery_with_recovery_state_memory"
                    ),
                    "version": 1,
                    "base_controller_hash": self.base_controller_hash,
                    "recovery_state_artifact_hash": self.recovery_state.artifact.artifact_hash,
                }
            )
        if self.contextual_recovery is not None:
            return hash_json(
                {
                    "controller_type": ("g1_cerebellar_post_kick_recovery_with_contextual_memory"),
                    "version": 1,
                    "base_controller_hash": self.base_controller_hash,
                    "contextual_recovery_artifact_hash": (
                        self.contextual_recovery.artifact.artifact_hash
                    ),
                }
            )
        return hash_json(
            {
                "controller_type": "g1_cerebellar_post_kick_recovery_with_muscle_memory",
                "version": 2,
                "base_controller_hash": self.base_controller_hash,
                "muscle_memory_artifact_hash": self.muscle_memory.artifact.artifact_hash,
            }
        )

    def require_compatible(
        self,
        *,
        body_hash: str,
        motion_hash: str,
        regime_commitment: str,
    ) -> None:
        if body_hash != self.body_hash:
            raise ValueError("cerebellar recovery Body hash mismatch")
        if motion_hash != self.motion_hash:
            raise ValueError("cerebellar recovery motion hash mismatch")
        if regime_commitment != self.regime_commitment:
            raise ValueError("cerebellar recovery regime commitment mismatch")

    def reset(self) -> None:
        self._contact_latched = False
        self._landing_latched = False
        self._activation_policy_frame: int | None = None
        self._activation_time_sec: float | None = None
        self._smoothing_activation_policy_frame: int | None = None
        self._smoothing_activation_time_sec: float | None = None
        self._settling_activation_policy_frame: int | None = None
        self._settling_activation_time_sec: float | None = None
        self._terminal_damping_activation_policy_frame: int | None = None
        self._terminal_damping_activation_time_sec: float | None = None
        self._smoothed_target: np.ndarray | None = None
        self._peak_blend_fraction = 0.0
        self._peak_settling_fraction = 0.0
        self._peak_terminal_damping_fraction = 0.0
        self._peak_smoothing_residual_rms_rad = 0.0
        self._fallback_routed_count = 0
        self._expert_route_latched: bool | None = None
        if self.muscle_memory is not None:
            self.muscle_memory.reset()
        if self.contextual_recovery is not None:
            self.contextual_recovery.reset()
        if self.recovery_state is not None:
            self.recovery_state.reset()

    def adapt_target(
        self,
        *,
        target: np.ndarray,
        policy_frame: int,
        timestamp_sec: float,
        ball_contact_detected: bool,
        left_support: bool,
        right_support: bool,
        muscle_memory_observation: Mapping[str, float] | None = None,
    ) -> G1CerebellarRecoveryEffect:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != self.standing_pose.shape or not np.all(np.isfinite(value)):
            raise ValueError("recovery target must match the finite G1 standing pose")
        if policy_frame < 0 or not math.isfinite(timestamp_sec):
            raise ValueError("recovery phase inputs must be finite and non-negative")

        self._contact_latched = self._contact_latched or bool(ball_contact_detected)
        if self._contact_latched and right_support:
            self._landing_latched = True
        config = self.config
        contextual_primitive_index: int | None = None
        contextual_ood = False
        recovery_state_primitive_index: int | None = None
        recovery_state_pending = False
        recovery_state_ood = False
        if self.contextual_recovery is not None:
            if muscle_memory_observation is None:
                raise ValueError("contextual recovery routing requires proprioceptive observations")
            if (
                self._contact_latched
                and self._landing_latched
                and policy_frame >= self.config.start_policy_frame
            ):
                selection = self.contextual_recovery.select(muscle_memory_observation)
                contextual_primitive_index = selection.primitive_index
                contextual_ood = selection.out_of_distribution
                if selection.primitive_index is None:
                    if self.fallback_config is None:
                        raise RuntimeError(
                            "validated contextual fallback config became unavailable"
                        )
                    config = self.fallback_config
                else:
                    config = _apply_contextual_primitive(
                        self.config,
                        self.contextual_recovery.artifact.primitives[selection.primitive_index],
                    )
        if self.recovery_state is not None:
            if muscle_memory_observation is None:
                raise ValueError("recovery-state routing requires proprioceptive observations")
            # Observe immediately after the causal contact-and-landing gate so
            # the short window is complete before the recovery action opens.
            # Selection alone cannot move the robot: target adaptation remains
            # gated below by the selected config's start_policy_frame.
            if self._contact_latched and self._landing_latched:
                selection = self.recovery_state.select(muscle_memory_observation)
                recovery_state_pending = not selection.ready
                recovery_state_primitive_index = selection.primitive_index
                recovery_state_ood = selection.out_of_distribution
                if selection.primitive_index is None:
                    if self.fallback_config is None:
                        raise RuntimeError(
                            "validated recovery-state fallback config became unavailable"
                        )
                    config = self.fallback_config
                else:
                    config = _apply_contextual_primitive(
                        self.config,
                        self.recovery_state.artifact.primitives[selection.primitive_index],
                    )
        if self.fallback_config is not None and self.muscle_memory is not None:
            if muscle_memory_observation is None:
                raise ValueError("temporal recovery routing requires proprioceptive observations")
            if (
                self._expert_route_latched is None
                and self._contact_latched
                and self._landing_latched
            ):
                self._expert_route_latched = self.muscle_memory.expert_regime_confident(
                    muscle_memory_observation
                )
            if self._expert_route_latched is not True:
                config = self.fallback_config
            if self._expert_route_latched is False:
                self._fallback_routed_count += 1
        causal_gate = (
            self.regime_eligible
            and (self._contact_latched or not config.contact_required)
            and (self._landing_latched or not config.kick_foot_landing_required)
        )
        eligible = causal_gate and policy_frame >= config.start_policy_frame
        if eligible:
            linear = min(
                1.0,
                max(
                    0.0,
                    (policy_frame - config.start_policy_frame) / config.blend_frames,
                ),
            )
            fraction = linear * linear * (3.0 - 2.0 * linear)
            settling_fraction = 0.0
            standing_pose_blend = config.standing_pose_blend
            roll_posture_bias = config.roll_posture_bias_rad
            if (
                config.settling_start_policy_frame is not None
                and policy_frame >= config.settling_start_policy_frame
            ):
                settling_standing_pose_blend = config.settling_standing_pose_blend
                settling_roll_posture_bias = config.settling_roll_posture_bias_rad
                if settling_standing_pose_blend is None or settling_roll_posture_bias is None:
                    raise RuntimeError("validated settling recovery config became incomplete")
                settling_linear = min(
                    1.0,
                    max(
                        0.0,
                        (policy_frame - config.settling_start_policy_frame)
                        / config.settling_blend_frames,
                    ),
                )
                settling_fraction = (
                    settling_linear * settling_linear * (3.0 - 2.0 * settling_linear)
                )
                standing_pose_blend += settling_fraction * (
                    settling_standing_pose_blend - standing_pose_blend
                )
                roll_posture_bias += settling_fraction * (
                    settling_roll_posture_bias - roll_posture_bias
                )
            standing_weight = fraction * standing_pose_blend
            adapted = (
                (1.0 - standing_weight) * value
                + standing_weight * self.standing_pose
                + fraction * roll_posture_bias * self._roll_pattern
                + settling_fraction
                * config.settling_waist_pitch_bias_rad
                * self._waist_pitch_pattern
            )
        else:
            fraction = 0.0
            settling_fraction = 0.0
            adapted = value.copy()
        smoothing_active = bool(
            causal_gate
            and config.target_smoothing_alpha < 1.0
            and policy_frame >= config.target_smoothing_start_policy_frame
        )
        previous = self._smoothed_target if self._smoothed_target is not None else value
        if smoothing_active:
            alpha = config.target_smoothing_alpha
            first_joint = {
                "upper_body": 12,
                "arms": 15,
            }[config.target_smoothing_joint_group]
            smoothed = adapted.copy()
            smoothed[first_joint:] = previous[first_joint:] + alpha * (
                adapted[first_joint:] - previous[first_joint:]
            )
            smoothing_residual = float(np.sqrt(np.mean(np.square(adapted - smoothed))))
            adapted = smoothed
            if self._smoothing_activation_policy_frame is None:
                self._smoothing_activation_policy_frame = policy_frame
                self._smoothing_activation_time_sec = timestamp_sec
            self._peak_smoothing_residual_rms_rad = max(
                self._peak_smoothing_residual_rms_rad,
                smoothing_residual,
            )
        else:
            smoothing_residual = 0.0
        terminal_fraction = 0.0
        if (
            causal_gate
            and config.terminal_damping_start_policy_frame is not None
            and policy_frame >= config.terminal_damping_start_policy_frame
        ):
            terminal_linear = min(
                1.0,
                max(
                    0.0,
                    (policy_frame - config.terminal_damping_start_policy_frame)
                    / config.terminal_damping_blend_frames,
                ),
            )
            terminal_fraction = terminal_linear * terminal_linear * (3.0 - 2.0 * terminal_linear)
        terminal_kp_scale = 1.0 + terminal_fraction * (config.terminal_kp_scale - 1.0)
        terminal_kd_scale = 1.0 + terminal_fraction * (config.terminal_kd_scale - 1.0)
        # Keep the slow structured controller's state independent from the
        # learned residual.  Feeding the combined target back through
        # ``_smoothed_target`` turns a bounded per-frame residual into an
        # unintended integrator and makes even tiny learned actions drift.
        structured_target = adapted.copy()
        muscle_memory_active = False
        muscle_memory_ood = False
        muscle_memory_residual_rms = 0.0
        muscle_memory_actions: np.ndarray = np.zeros(0, dtype=np.float64)
        if self.muscle_memory is not None and causal_gate:
            if muscle_memory_observation is None:
                raise ValueError("learned muscle memory requires proprioceptive observations")
            muscle_effect = self.muscle_memory.infer(muscle_memory_observation)
            adapted = adapted + muscle_effect.residual
            muscle_memory_active = muscle_effect.active
            muscle_memory_ood = muscle_effect.out_of_distribution
            muscle_memory_residual_rms = muscle_effect.residual_rms_rad
            muscle_memory_actions = muscle_effect.synergy_actions.copy()
        self._smoothed_target = structured_target
        active = (
            fraction > 0.0 or smoothing_active or terminal_fraction > 0.0 or muscle_memory_active
        )
        if active and self._activation_policy_frame is None:
            self._activation_policy_frame = policy_frame
            self._activation_time_sec = timestamp_sec
        if settling_fraction > 0.0 and self._settling_activation_policy_frame is None:
            self._settling_activation_policy_frame = policy_frame
            self._settling_activation_time_sec = timestamp_sec
        if terminal_fraction > 0.0 and self._terminal_damping_activation_policy_frame is None:
            self._terminal_damping_activation_policy_frame = policy_frame
            self._terminal_damping_activation_time_sec = timestamp_sec
        self._peak_blend_fraction = max(self._peak_blend_fraction, fraction)
        self._peak_settling_fraction = max(self._peak_settling_fraction, settling_fraction)
        self._peak_terminal_damping_fraction = max(
            self._peak_terminal_damping_fraction,
            terminal_fraction,
        )
        return G1CerebellarRecoveryEffect(
            target=adapted,
            active=active,
            blend_fraction=fraction,
            settling_fraction=settling_fraction,
            contact_latched=self._contact_latched,
            kick_foot_landing_latched=self._landing_latched,
            smoothing_active=smoothing_active,
            smoothing_residual_rms_rad=smoothing_residual,
            terminal_damping_fraction=terminal_fraction,
            terminal_kp_scale=terminal_kp_scale,
            terminal_kd_scale=terminal_kd_scale,
            terminal_damping_joint_group=config.terminal_damping_joint_group,
            muscle_memory_active=muscle_memory_active,
            muscle_memory_out_of_distribution=muscle_memory_ood,
            muscle_memory_residual_rms_rad=muscle_memory_residual_rms,
            muscle_memory_synergy_actions=muscle_memory_actions,
            contextual_recovery_active=(
                self.contextual_recovery is not None
                and contextual_primitive_index is not None
                and causal_gate
            ),
            contextual_recovery_out_of_distribution=contextual_ood,
            contextual_recovery_primitive_index=contextual_primitive_index,
            recovery_state_active=(
                self.recovery_state is not None
                and recovery_state_primitive_index is not None
                and eligible
            ),
            recovery_state_pending=recovery_state_pending,
            recovery_state_out_of_distribution=recovery_state_ood,
            recovery_state_primitive_index=recovery_state_primitive_index,
        )

    def build_receipt(
        self,
        *,
        strict_replay: bool,
        evidence_domain: str = "SIM",
    ) -> G1CerebellarRecoveryReceipt:
        return G1CerebellarRecoveryReceipt(
            controller_hash=self.controller_hash,
            config_hash=self.config_hash,
            body_hash=self.body_hash,
            motion_hash=self.motion_hash,
            standing_pose_hash=self.standing_pose_hash,
            regime_commitment=self.regime_commitment,
            regime_eligible=self.regime_eligible,
            regime_reasons=self.regime_reasons,
            contact_latched=self._contact_latched,
            kick_foot_landing_latched=self._landing_latched,
            activation_policy_frame=self._activation_policy_frame,
            activation_time_sec=self._activation_time_sec,
            smoothing_activation_policy_frame=self._smoothing_activation_policy_frame,
            smoothing_activation_time_sec=self._smoothing_activation_time_sec,
            settling_activation_policy_frame=self._settling_activation_policy_frame,
            settling_activation_time_sec=self._settling_activation_time_sec,
            terminal_damping_activation_policy_frame=(
                self._terminal_damping_activation_policy_frame
            ),
            terminal_damping_activation_time_sec=self._terminal_damping_activation_time_sec,
            peak_blend_fraction=self._peak_blend_fraction,
            peak_settling_fraction=self._peak_settling_fraction,
            peak_terminal_damping_fraction=self._peak_terminal_damping_fraction,
            peak_smoothing_residual_rms_rad=self._peak_smoothing_residual_rms_rad,
            strict_replay=strict_replay,
            evidence_domain=evidence_domain,
            config=asdict(self.config),
            fallback_config_hash=self.fallback_config_hash,
            fallback_config=(
                asdict(self.fallback_config) if self.fallback_config is not None else None
            ),
            fallback_routed_count=self._fallback_routed_count,
            expert_route_latched=self._expert_route_latched,
            muscle_memory_receipt=(
                self.muscle_memory.build_receipt().to_dict()
                if self.muscle_memory is not None
                else None
            ),
            contextual_recovery_receipt=(
                self.contextual_recovery.build_receipt().to_dict()
                if self.contextual_recovery is not None
                else None
            ),
            recovery_state_receipt=(
                self.recovery_state.build_receipt().to_dict()
                if self.recovery_state is not None
                else None
            ),
        )


def _apply_contextual_primitive(
    config: G1CerebellarRecoveryConfig,
    primitive: G1ContextualRecoveryPrimitive,
) -> G1CerebellarRecoveryConfig:
    """Apply only the bounded fields carried by a learned SIM primitive."""

    return replace(
        config,
        start_policy_frame=primitive.start_policy_frame,
        blend_frames=primitive.blend_frames,
        settling_start_policy_frame=primitive.settling_start_policy_frame,
        settling_blend_frames=primitive.settling_blend_frames,
        settling_standing_pose_blend=primitive.settling_standing_pose_blend,
        settling_waist_pitch_bias_rad=primitive.settling_waist_pitch_bias_rad,
        target_smoothing_alpha=primitive.target_smoothing_alpha,
        target_smoothing_start_policy_frame=primitive.start_policy_frame,
    )


def evaluate_g1_cerebellar_recovery_regime(
    *,
    support_friction: float,
    control_latency_ms: float,
    disturbance_n: float,
    config: G1CerebellarRecoveryConfig,
) -> tuple[bool, tuple[str, ...]]:
    """Fail closed outside the SIM regimes covered by matched validation."""

    values = (support_friction, control_latency_ms, disturbance_n)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("recovery regime inputs must be finite")
    reasons = []
    if support_friction < config.minimum_calibrated_support_friction:
        reasons.append("support_friction_below_calibrated_range")
    if control_latency_ms > config.maximum_calibrated_control_latency_ms:
        reasons.append("control_latency_above_calibrated_range")
    magnitude = abs(disturbance_n)
    if 0.0 < magnitude < config.minimum_calibrated_disturbance_n:
        reasons.append("disturbance_below_calibrated_recovery_range")
    if magnitude > config.maximum_calibrated_disturbance_n:
        reasons.append("disturbance_above_calibrated_recovery_range")
    return not reasons, tuple(reasons)


def _roll_posture_pattern() -> np.ndarray:
    index = {name: position for position, name in enumerate(G1_DDS_JOINT_NAMES)}
    pattern: np.ndarray = np.zeros(len(G1_DDS_JOINT_NAMES), dtype=np.float64)
    for name in ("left_hip_roll_joint", "right_hip_roll_joint"):
        pattern[index[name]] = 1.0
    for name in ("left_ankle_roll_joint", "right_ankle_roll_joint"):
        pattern[index[name]] = -0.65
    pattern[index["waist_roll_joint"]] = 0.45
    pattern.setflags(write=False)
    return pattern


def _waist_pitch_pattern() -> np.ndarray:
    index = {name: position for position, name in enumerate(G1_DDS_JOINT_NAMES)}
    pattern: np.ndarray = np.zeros(len(G1_DDS_JOINT_NAMES), dtype=np.float64)
    pattern[index["waist_pitch_joint"]] = 1.0
    pattern.setflags(write=False)
    return pattern


__all__ = [
    "G1CerebellarRecoveryConfig",
    "G1CerebellarRecoveryController",
    "G1CerebellarRecoveryEffect",
    "G1CerebellarRecoveryReceipt",
    "evaluate_g1_cerebellar_recovery_regime",
]
