"""Risk-gated hierarchy for G1 balance and recovery torque actors.

The balance and recovery actors are deliberately separate recurrent policies.
They may learn from disjoint replay windows without asking one residual head to
solve pre-contact balance, ball contact, and post-contact settling at once.
Every head remains behind the independent ``G1TorqueSafetyProjector`` and this
composer is simulator-only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from rosclaw.simforge.g1_neural_torque import (
    G1NeuralTorquePolicy,
    G1TorqueControlFrame,
    G1TorquePolicyReceipt,
    G1TorqueProjection,
    build_g1_neural_torque_observation,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class G1HierarchicalTorqueGateConfig:
    """Immutable routing and posture gates for the two plastic heads."""

    balance_start_phase: float = 0.02
    balance_end_phase: float = 0.20
    recovery_start_phase: float = 0.20
    balance_minimum_pelvis_height_m: float = 0.62
    balance_maximum_projected_gravity_z: float = -0.75
    recovery_minimum_pelvis_height_m: float = 0.70
    recovery_maximum_projected_gravity_z: float = -0.88
    balance_warmup_steps: int = 5
    recovery_warmup_steps: int = 5
    require_foot_contact: bool = True
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.balance_start_phase < self.balance_end_phase <= 0.90
            and self.balance_end_phase <= self.recovery_start_phase <= 1.0
        ):
            raise ValueError("hierarchical torque phase windows are invalid")
        for label, value in (
            ("balance", self.balance_minimum_pelvis_height_m),
            ("recovery", self.recovery_minimum_pelvis_height_m),
        ):
            if not 0.50 <= value <= 0.95:
                raise ValueError(f"{label} pelvis-height gate must be in [0.50, 0.95] m")
        for label, value in (
            ("balance", self.balance_maximum_projected_gravity_z),
            ("recovery", self.recovery_maximum_projected_gravity_z),
        ):
            if not -0.999 <= value <= -0.60:
                raise ValueError(f"{label} gravity gate must be in [-0.999, -0.60]")
        if not 1 <= self.balance_warmup_steps <= 500:
            raise ValueError("balance eligibility warmup must be in [1, 500]")
        if not 1 <= self.recovery_warmup_steps <= 500:
            raise ValueError("recovery eligibility warmup must be in [1, 500]")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("hierarchical direct-torque policy is SIM_ONLY")


@dataclass(frozen=True)
class G1HierarchicalTorqueReceipt(G1TorquePolicyReceipt):
    stable_artifact_hash: str = ""
    balance_artifact_hash: str = ""
    recovery_artifact_hash: str = ""
    gate_hash: str = ""
    balance_activation_count: int = 0
    balance_activation_fraction: float = 0.0
    recovery_activation_count: int = 0
    recovery_activation_fraction: float = 0.0
    balance_phase_rejection_count: int = 0
    recovery_phase_rejection_count: int = 0
    balance_posture_rejection_count: int = 0
    recovery_posture_rejection_count: int = 0
    contact_rejection_count: int = 0
    schema_version: str = "rosclaw.simforge.g1_hierarchical_torque_receipt.v1"


class G1HierarchicalTorquePolicy:
    """Route stable, early-balance, and post-contact recovery actors safely."""

    def __init__(
        self,
        stable: G1NeuralTorquePolicy,
        balance: G1NeuralTorquePolicy,
        recovery: G1NeuralTorquePolicy,
        *,
        config: G1HierarchicalTorqueGateConfig | None = None,
    ) -> None:
        body_hashes = {
            stable.artifact.body_hash,
            balance.artifact.body_hash,
            recovery.artifact.body_hash,
        }
        parent_hashes = {
            stable.artifact.parent_policy_hash,
            balance.artifact.parent_policy_hash,
            recovery.artifact.parent_policy_hash,
        }
        if len(body_hashes) != 1:
            raise ValueError("hierarchical torque actors target different bodies")
        if len(parent_hashes) != 1:
            raise ValueError("hierarchical torque actors have different parents")
        self.stable = stable
        self.balance = balance
        self.recovery = recovery
        self.config = config or G1HierarchicalTorqueGateConfig()
        self.artifact_hash = hash_json(
            {
                "stable_artifact_hash": stable.artifact.artifact_hash,
                "balance_artifact_hash": balance.artifact.artifact_hash,
                "recovery_artifact_hash": recovery.artifact.artifact_hash,
                "gate": asdict(self.config),
            }
        )
        self._pending = False
        self._balance_eligible_steps = 0
        self._recovery_eligible_steps = 0
        self._inference_count = 0
        self._balance_activation_count = 0
        self._recovery_activation_count = 0
        self._balance_phase_rejections = 0
        self._recovery_phase_rejections = 0
        self._balance_posture_rejections = 0
        self._recovery_posture_rejections = 0
        self._contact_rejections = 0
        self._balance_activation_mask: list[bool] = []
        self._recovery_activation_mask: list[bool] = []
        self._fallback_reasons: list[str] = []
        self._projected_joint_count = 0
        self._maximum_limit_ratio = 0.0
        self._peak_power = 0.0

    def reset(self) -> None:
        self.stable.reset()
        self.balance.reset()
        self.recovery.reset()
        self._pending = False
        self._balance_eligible_steps = 0
        self._recovery_eligible_steps = 0
        self._inference_count = 0
        self._balance_activation_count = 0
        self._recovery_activation_count = 0
        self._balance_phase_rejections = 0
        self._recovery_phase_rejections = 0
        self._balance_posture_rejections = 0
        self._recovery_posture_rejections = 0
        self._contact_rejections = 0
        self._balance_activation_mask.clear()
        self._recovery_activation_mask.clear()
        self._fallback_reasons.clear()
        self._projected_joint_count = 0
        self._maximum_limit_ratio = 0.0
        self._peak_power = 0.0

    def command(self, frame: G1TorqueControlFrame, parent_torque: np.ndarray) -> np.ndarray:
        if self._pending:
            raise RuntimeError("hierarchical torque policy did not receive note_applied")
        balance_phase = bool(
            self.config.balance_start_phase
            <= frame.policy_phase
            < self.config.balance_end_phase
        )
        recovery_phase = bool(frame.policy_phase >= self.config.recovery_start_phase)
        contact_ok = bool(
            not self.config.require_foot_contact
            or frame.left_contact
            or frame.right_contact
        )
        balance_posture = self._posture_eligible(
            frame,
            minimum_pelvis_height_m=self.config.balance_minimum_pelvis_height_m,
            maximum_projected_gravity_z=(
                self.config.balance_maximum_projected_gravity_z
            ),
        )
        recovery_posture = self._posture_eligible(
            frame,
            minimum_pelvis_height_m=self.config.recovery_minimum_pelvis_height_m,
            maximum_projected_gravity_z=(
                self.config.recovery_maximum_projected_gravity_z
            ),
        )
        balance_eligible = balance_phase and balance_posture and contact_ok
        recovery_eligible = recovery_phase and recovery_posture and contact_ok
        self._balance_eligible_steps = (
            self._balance_eligible_steps + 1 if balance_eligible else 0
        )
        self._recovery_eligible_steps = (
            self._recovery_eligible_steps + 1 if recovery_eligible else 0
        )
        balance_ready = (
            balance_eligible
            and self._balance_eligible_steps >= self.config.balance_warmup_steps
        )
        recovery_ready = (
            recovery_eligible
            and self._recovery_eligible_steps >= self.config.recovery_warmup_steps
        )
        self._balance_phase_rejections += int(not balance_phase)
        self._recovery_phase_rejections += int(not recovery_phase)
        self._balance_posture_rejections += int(balance_phase and not balance_posture)
        self._recovery_posture_rejections += int(recovery_phase and not recovery_posture)
        self._contact_rejections += int(
            (balance_phase or recovery_phase)
            and (balance_posture or recovery_posture)
            and not contact_ok
        )

        stable_torque = self.stable.command(
            frame,
            parent_torque,
            allow_exploration=False,
        )
        balance_torque = self.balance.command(
            frame,
            parent_torque,
            allow_exploration=balance_ready,
        )
        recovery_torque = self.recovery.command(
            frame,
            parent_torque,
            allow_exploration=recovery_ready,
        )
        if balance_ready:
            selected = self.balance
            output = balance_torque
        elif recovery_ready:
            selected = self.recovery
            output = recovery_torque
        else:
            selected = self.stable
            output = stable_torque
        projection = selected.pending_projection
        balance_contributed = bool(
            selected is self.balance and projection.reason is None
        )
        recovery_contributed = bool(
            selected is self.recovery and projection.reason is None
        )
        self._balance_activation_mask.append(balance_contributed)
        self._recovery_activation_mask.append(recovery_contributed)
        self._balance_activation_count += int(balance_contributed)
        self._recovery_activation_count += int(recovery_contributed)
        self._inference_count += 1
        self._record_projection(projection)
        self._pending = True
        return output.copy()

    def note_applied(self, torque: np.ndarray) -> None:
        if not self._pending:
            raise RuntimeError("hierarchical torque policy has no pending command")
        self.stable.note_applied(torque)
        self.balance.note_applied(torque)
        self.recovery.note_applied(torque)
        self._pending = False

    def build_receipt(self) -> G1HierarchicalTorqueReceipt:
        if self._pending or not self._inference_count:
            raise ValueError("cannot build an incomplete hierarchical torque receipt")
        fallback_count = len(self._fallback_reasons)
        balance_receipt = self.balance.build_receipt()
        recovery_receipt = self.recovery.build_receipt()
        exploration_hash = hash_json(
            {
                "balance": balance_receipt.exploration_config_hash,
                "recovery": recovery_receipt.exploration_config_hash,
            }
        )
        return G1HierarchicalTorqueReceipt(
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
                in {
                    "state_recovery_parent",
                    "recovery_cooldown_parent",
                    "joint_limit_guard",
                }
                for reason in self._fallback_reasons
            ),
            projected_joint_count=self._projected_joint_count,
            maximum_limit_ratio=self._maximum_limit_ratio,
            peak_mechanical_power_w=self._peak_power,
            direct_torque_output=True,
            activation_ceiling="SIM_ONLY",
            exploration_config_hash=exploration_hash,
            exploration_attempt_count=(
                balance_receipt.exploration_attempt_count
                + recovery_receipt.exploration_attempt_count
            ),
            exploration_applied_count=(
                balance_receipt.exploration_applied_count
                + recovery_receipt.exploration_applied_count
            ),
            exploration_rejection_count=(
                balance_receipt.exploration_rejection_count
                + recovery_receipt.exploration_rejection_count
            ),
            exploration_noise_rms_ratio=_combined_exploration_rms(
                balance_receipt,
                recovery_receipt,
            ),
            exploration_noise_peak_ratio=max(
                balance_receipt.exploration_noise_peak_ratio,
                recovery_receipt.exploration_noise_peak_ratio,
            ),
            stable_artifact_hash=self.stable.artifact.artifact_hash,
            balance_artifact_hash=self.balance.artifact.artifact_hash,
            recovery_artifact_hash=self.recovery.artifact.artifact_hash,
            gate_hash=hash_json(asdict(self.config)),
            balance_activation_count=self._balance_activation_count,
            balance_activation_fraction=(
                self._balance_activation_count / self._inference_count
            ),
            recovery_activation_count=self._recovery_activation_count,
            recovery_activation_fraction=(
                self._recovery_activation_count / self._inference_count
            ),
            balance_phase_rejection_count=self._balance_phase_rejections,
            recovery_phase_rejection_count=self._recovery_phase_rejections,
            balance_posture_rejection_count=self._balance_posture_rejections,
            recovery_posture_rejection_count=self._recovery_posture_rejections,
            contact_rejection_count=self._contact_rejections,
        )

    def balance_activation_mask(self) -> np.ndarray:
        return self._activation_mask(self._balance_activation_mask, label="balance")

    def recovery_activation_mask(self) -> np.ndarray:
        return self._activation_mask(self._recovery_activation_mask, label="recovery")

    def _record_projection(self, projection: G1TorqueProjection) -> None:
        if projection.reason is not None:
            self._fallback_reasons.append(projection.reason)
        self._projected_joint_count += projection.projected_joint_count
        self._maximum_limit_ratio = max(
            self._maximum_limit_ratio,
            projection.maximum_limit_ratio,
        )
        self._peak_power = max(self._peak_power, projection.mechanical_power_w)

    def _activation_mask(self, values: list[bool], *, label: str) -> np.ndarray:
        if self._pending or not values:
            raise ValueError(f"{label} activation mask is empty or incomplete")
        return np.asarray(values, dtype=np.bool_)

    @staticmethod
    def _posture_eligible(
        frame: G1TorqueControlFrame,
        *,
        minimum_pelvis_height_m: float,
        maximum_projected_gravity_z: float,
    ) -> bool:
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
            and gravity_z <= maximum_projected_gravity_z
            and float(frame.pelvis_position[2]) >= minimum_pelvis_height_m
        )


def _combined_exploration_rms(
    balance: G1TorquePolicyReceipt,
    recovery: G1TorquePolicyReceipt,
) -> float:
    count = balance.exploration_applied_count + recovery.exploration_applied_count
    if count <= 0:
        return 0.0
    square = (
        balance.exploration_applied_count * balance.exploration_noise_rms_ratio**2
        + recovery.exploration_applied_count * recovery.exploration_noise_rms_ratio**2
    )
    return math.sqrt(square / count)


__all__ = [
    "G1HierarchicalTorqueGateConfig",
    "G1HierarchicalTorquePolicy",
    "G1HierarchicalTorqueReceipt",
]
