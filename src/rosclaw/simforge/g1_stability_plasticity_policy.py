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
    minimum_pelvis_height_m: float = 0.76
    maximum_projected_gravity_z: float = -0.94
    eligibility_warmup_steps: int = 20
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if not 0.5 <= self.minimum_recovery_phase <= 1.0:
            raise ValueError("plastic recovery phase must be in [0.5, 1]")
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
    phase_rejection_count: int = 0
    posture_rejection_count: int = 0
    contact_rejection_count: int = 0
    schema_version: str = "rosclaw.simforge.g1_stability_plasticity_receipt.v1"


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
        self._phase_rejections = 0
        self._posture_rejections = 0
        self._contact_rejections = 0
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
        self._phase_rejections = 0
        self._posture_rejections = 0
        self._contact_rejections = 0
        self._fallback_reasons.clear()
        self._projected_joint_count = 0
        self._maximum_limit_ratio = 0.0
        self._peak_power = 0.0

    def command(self, frame: G1TorqueControlFrame, parent_torque: np.ndarray) -> np.ndarray:
        if self._pending:
            raise RuntimeError("stability-plasticity policy did not receive note_applied")
        stable_torque = self.stable.command(frame, parent_torque)
        plastic_torque = self.plastic.command(frame, parent_torque)
        phase_ok = frame.policy_phase >= self.config.minimum_recovery_phase
        posture_ok = self._posture_eligible(frame)
        contact_ok = frame.left_contact or frame.right_contact
        self._phase_rejections += int(not phase_ok)
        self._posture_rejections += int(phase_ok and not posture_ok)
        self._contact_rejections += int(phase_ok and posture_ok and not contact_ok)
        eligible = phase_ok and posture_ok and contact_ok
        self._eligible_steps = self._eligible_steps + 1 if eligible else 0
        plastic_active = self._eligible_steps >= self.config.eligibility_warmup_steps
        selected = self.plastic if plastic_active else self.stable
        projection = selected.pending_projection
        self._activation_count += int(plastic_active)
        self._inference_count += 1
        if projection.reason is not None:
            self._fallback_reasons.append(projection.reason)
        self._projected_joint_count += projection.projected_joint_count
        self._maximum_limit_ratio = max(
            self._maximum_limit_ratio, projection.maximum_limit_ratio
        )
        self._peak_power = max(self._peak_power, projection.mechanical_power_w)
        self._pending = True
        return (plastic_torque if plastic_active else stable_torque).copy()

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
            phase_rejection_count=self._phase_rejections,
            posture_rejection_count=self._posture_rejections,
            contact_rejection_count=self._contact_rejections,
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
    "G1StabilityPlasticityGateConfig",
    "G1StabilityPlasticityReceipt",
    "G1StabilityPlasticityTorquePolicy",
]
