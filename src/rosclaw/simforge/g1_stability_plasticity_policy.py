"""Context-gated composition of stable and plastic direct-torque actors."""

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
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class G1StabilityPlasticityGateConfig:
    minimum_recovery_phase: float = 0.85
    minimum_anticipatory_phase: float | None = None
    anticipatory_blend_fraction: float = 0.0
    minimum_pelvis_height_m: float = 0.76
    maximum_projected_gravity_z: float = -0.94
    eligibility_warmup_steps: int = 20
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if not 0.02 <= self.minimum_recovery_phase <= 1.0:
            raise ValueError("plastic recovery phase must be in [0.02, 1]")
        if self.minimum_anticipatory_phase is None:
            if self.anticipatory_blend_fraction != 0.0:
                raise ValueError("anticipatory blend requires an anticipatory phase")
        elif (
            not 0.02 <= self.minimum_anticipatory_phase < self.minimum_recovery_phase
            or not 0.0 < self.anticipatory_blend_fraction <= 0.5
        ):
            raise ValueError("anticipatory blend phase or fraction is invalid")
        if not 0.55 <= self.minimum_pelvis_height_m <= 0.95:
            raise ValueError("plastic pelvis-height gate must be in [0.55, 0.95] m")
        if not -0.999 <= self.maximum_projected_gravity_z <= -0.75:
            raise ValueError("plastic gravity gate must be in [-0.999, -0.75]")
        if not 1 <= self.eligibility_warmup_steps <= 500:
            raise ValueError("plastic eligibility warmup must be in [1, 500]")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("stability-plasticity torque policy is SIM_ONLY")


@dataclass(frozen=True)
class G1StabilityPlasticityReceipt(G1TorquePolicyReceipt):
    stable_artifact_hash: str = ""
    plastic_artifact_hash: str = ""
    gate_hash: str = ""
    plastic_activation_count: int = 0
    plastic_activation_fraction: float = 0.0
    anticipatory_blend_count: int = 0
    phase_rejection_count: int = 0
    posture_rejection_count: int = 0
    contact_rejection_count: int = 0
    schema_version: str = "rosclaw.simforge.g1_stability_plasticity_receipt.v3"


class G1StabilityPlasticityTorquePolicy:
    """Use learned plasticity only in its post-kick stability regime."""

    def __init__(
        self,
        stable: G1NeuralTorquePolicy,
        plastic: G1NeuralTorquePolicy,
        *,
        config: G1StabilityPlasticityGateConfig | None = None,
    ) -> None:
        if stable.artifact.body_hash != plastic.artifact.body_hash:
            raise ValueError("stable and plastic torque actors target different bodies")
        if stable.artifact.parent_policy_hash != plastic.artifact.parent_policy_hash:
            raise ValueError("stable and plastic torque actors have different parents")
        self.stable = stable
        self.plastic = plastic
        self.config = config or G1StabilityPlasticityGateConfig()
        self.artifact_hash = hash_json(
            {
                "stable_artifact_hash": stable.artifact.artifact_hash,
                "plastic_artifact_hash": plastic.artifact.artifact_hash,
                "gate": asdict(self.config),
            }
        )
        self._pending = False
        self._eligible_steps = 0
        self._inference_count = 0
        self._activation_count = 0
        self._anticipatory_blend_count = 0
        self._phase_rejections = 0
        self._posture_rejections = 0
        self._contact_rejections = 0
        self._plastic_activation_mask: list[bool] = []
        self._fallback_reasons: list[str] = []
        self._projected_joint_count = 0
        self._maximum_limit_ratio = 0.0
        self._peak_power = 0.0

    def reset(self) -> None:
        self.stable.reset()
        self.plastic.reset()
        self._pending = False
        self._eligible_steps = 0
        self._inference_count = 0
        self._activation_count = 0
        self._anticipatory_blend_count = 0
        self._phase_rejections = 0
        self._posture_rejections = 0
        self._contact_rejections = 0
        self._plastic_activation_mask.clear()
        self._fallback_reasons.clear()
        self._projected_joint_count = 0
        self._maximum_limit_ratio = 0.0
        self._peak_power = 0.0

    def command(self, frame: G1TorqueControlFrame, parent_torque: np.ndarray) -> np.ndarray:
        if self._pending:
            raise RuntimeError("stability-plasticity policy did not receive note_applied")
        recovery_phase_ok = frame.policy_phase >= self.config.minimum_recovery_phase
        anticipatory_phase_ok = bool(
            self.config.minimum_anticipatory_phase is not None
            and self.config.minimum_anticipatory_phase
            <= frame.policy_phase
            < self.config.minimum_recovery_phase
        )
        phase_ok = recovery_phase_ok or anticipatory_phase_ok
        posture_ok = self._posture_eligible(frame)
        contact_ok = frame.left_contact or frame.right_contact
        self._phase_rejections += int(not phase_ok)
        self._posture_rejections += int(phase_ok and not posture_ok)
        self._contact_rejections += int(phase_ok and posture_ok and not contact_ok)
        eligible = phase_ok and posture_ok and contact_ok
        self._eligible_steps = self._eligible_steps + 1 if eligible else 0
        plastic_ready = self._eligible_steps >= self.config.eligibility_warmup_steps
        stable_torque = self.stable.command(
            frame,
            parent_torque,
            allow_exploration=False,
        )
        plastic_torque = self.plastic.command(
            frame,
            parent_torque,
            allow_exploration=plastic_ready,
        )
        plastic_projection = self.plastic.pending_projection
        stable_projection = self.stable.pending_projection
        anticipatory_active = bool(
            plastic_ready
            and anticipatory_phase_ok
            and plastic_projection.reason is None
            and stable_projection.reason is None
        )
        recovery_active = plastic_ready and recovery_phase_ok
        selected = self.plastic if recovery_active else self.stable
        projection = selected.pending_projection
        plastic_contributed = bool(
            anticipatory_active or (recovery_active and projection.reason is None)
        )
        self._activation_count += int(plastic_contributed)
        self._anticipatory_blend_count += int(anticipatory_active)
        self._plastic_activation_mask.append(plastic_contributed)
        self._inference_count += 1
        if projection.reason is not None:
            self._fallback_reasons.append(projection.reason)
        self._projected_joint_count += projection.projected_joint_count
        self._maximum_limit_ratio = max(
            self._maximum_limit_ratio, projection.maximum_limit_ratio
        )
        self._peak_power = max(self._peak_power, projection.mechanical_power_w)
        self._pending = True
        if anticipatory_active:
            fraction = self.config.anticipatory_blend_fraction
            return (stable_torque + fraction * (plastic_torque - stable_torque)).copy()
        return (plastic_torque if recovery_active else stable_torque).copy()

    def note_applied(self, torque: np.ndarray) -> None:
        if not self._pending:
            raise RuntimeError("stability-plasticity policy has no pending command")
        self.stable.note_applied(torque)
        self.plastic.note_applied(torque)
        self._pending = False

    def build_receipt(self) -> G1StabilityPlasticityReceipt:
        if self._pending or not self._inference_count:
            raise ValueError("cannot build an incomplete stability-plasticity receipt")
        fallback_count = len(self._fallback_reasons)
        plastic_receipt = self.plastic.build_receipt()
        return G1StabilityPlasticityReceipt(
            artifact_hash=self.artifact_hash,
            body_hash=self.stable.artifact.body_hash,
            parent_policy_hash=self.stable.artifact.parent_policy_hash,
            inference_count=self._inference_count,
            learned_output_count=self._inference_count - fallback_count,
            fallback_count=fallback_count,
            nonfinite_fallback_count=sum(
                "invalid_proposal" in reason or "observation_or_inference" in reason
                for reason in self._fallback_reasons
            ),
            projection_fallback_count=sum(
                reason == "projection_ratio_exceeded" for reason in self._fallback_reasons
            ),
            out_of_distribution_fallback_count=sum(
                reason == "observation_out_of_distribution" for reason in self._fallback_reasons
            ),
            warmup_fallback_count=sum(
                reason == "warmup_parent" for reason in self._fallback_reasons
            ),
            state_guard_fallback_count=sum(
                reason
                in {"state_recovery_parent", "recovery_cooldown_parent", "joint_limit_guard"}
                for reason in self._fallback_reasons
            ),
            projected_joint_count=self._projected_joint_count,
            maximum_limit_ratio=self._maximum_limit_ratio,
            peak_mechanical_power_w=self._peak_power,
            direct_torque_output=True,
            activation_ceiling="SIM_ONLY",
            stable_artifact_hash=self.stable.artifact.artifact_hash,
            plastic_artifact_hash=self.plastic.artifact.artifact_hash,
            gate_hash=hash_json(asdict(self.config)),
            plastic_activation_count=self._activation_count,
            plastic_activation_fraction=self._activation_count / self._inference_count,
            anticipatory_blend_count=self._anticipatory_blend_count,
            phase_rejection_count=self._phase_rejections,
            posture_rejection_count=self._posture_rejections,
            contact_rejection_count=self._contact_rejections,
            exploration_config_hash=plastic_receipt.exploration_config_hash,
            exploration_attempt_count=plastic_receipt.exploration_attempt_count,
            exploration_applied_count=plastic_receipt.exploration_applied_count,
            exploration_rejection_count=plastic_receipt.exploration_rejection_count,
            exploration_noise_rms_ratio=plastic_receipt.exploration_noise_rms_ratio,
            exploration_noise_peak_ratio=plastic_receipt.exploration_noise_peak_ratio,
        )

    def plastic_activation_mask(self) -> np.ndarray:
        if self._pending or not self._plastic_activation_mask:
            raise ValueError("plastic activation mask is empty or incomplete")
        return np.asarray(self._plastic_activation_mask, dtype=np.bool_)

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
    "G1StabilityPlasticityGateConfig",
    "G1StabilityPlasticityReceipt",
    "G1StabilityPlasticityTorquePolicy",
]
