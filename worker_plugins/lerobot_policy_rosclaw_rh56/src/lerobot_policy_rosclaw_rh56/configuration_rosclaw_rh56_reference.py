"""Configuration for the deterministic RH56 reference policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig

RH56_STATE_NAMES = ["little", "ring", "middle", "index", "thumb", "thumb_rot"]

REFERENCE_TASKS = (
    "hold_current",
    "open_hand",
    "micro_index_flex",
    "half_close",
    "return_open",
    "countdown_pose",
    "ok_pose_safe",
)


@PreTrainedConfig.register_subclass("rosclaw_rh56_reference")
@dataclass
class RosclawRH56ReferenceConfig(PreTrainedConfig):
    """Config for the deterministic RH56 reference policy.

    The policy consumes ``observation.state`` in raw device units (0-1000) and
    emits a 6-dim joint_position action in the same raw device units.  It is a
    finite-state, receding-horizon single-step controller used to validate the
    LeRobot → ROSClaw → RH56 deployment loop without any training.
    """

    task: str = "hold_current"
    max_step_delta: int = 20
    hold_steps: int = 2
    # Module imported by the ROSClaw worker before get_policy_class() so this
    # config's registration becomes visible inside the persistent runtime.
    plugin_module: str | None = "lerobot_policy_rosclaw_rh56"

    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        }
    )
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        }
    )
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError("Deterministic reference policy has no optimizer")

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        state = self.input_features.get("observation.state") if self.input_features else None
        if state is None or list(state.shape) != [6]:
            raise ValueError("rosclaw_rh56_reference requires observation.state shape [6]")
        action = self.output_features.get("action") if self.output_features else None
        if action is None or list(action.shape) != [6]:
            raise ValueError("rosclaw_rh56_reference requires action shape [6]")
        if self.task not in REFERENCE_TASKS:
            raise ValueError(
                f"Unknown reference task {self.task!r}; expected one of {REFERENCE_TASKS}"
            )
