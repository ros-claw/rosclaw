"""Recurrent direct-torque actor-critic for the G1 MuJoCo sandbox.

The learner supports three stages:

1. behavior-clone the qualified RoboNaldo+PD teacher into a GRU actor;
2. update twin reward and safety-cost critics from versioned simulation replay;
3. update the actor only from fresh transitions while rehearsing historical
   anchors and applying an EWC penalty around the consolidated parent.

Only the actor is exported, using the safe tensor format in
``g1_neural_torque``.  Training checkpoints are service-owned and restored
with ``weights_only=True``.  Neither form can authorize hardware execution.
"""

from __future__ import annotations

import copy
import hashlib
import io
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.g1_neural_torque import (
    G1_NEURAL_TORQUE_OBSERVATIONS,
    G1TeacherTorqueEpisode,
    G1TorqueSafetyConfig,
    serialize_g1_neural_torque_artifact,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

if TYPE_CHECKING:
    from rosclaw.collective.sources.motiondecode.motion_prior import G1MotionPriorArtifact

_RECENT = 0
_ANCHOR = 1
_BOUNDARY = 2
_G1_MOTION_PRIOR_FEATURE_COUNT = 61


@dataclass(frozen=True)
class G1NeuralTorqueLearnerConfig:
    hidden_dim: int = 96
    sequence_length: int = 16
    batch_size: int = 256
    gamma: float = 0.995
    tau: float = 0.01
    actor_lr: float = 2e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 2e-4
    initial_alpha: float = 0.02
    initial_log_std: float = -3.0
    lagrange_lr: float = 1e-3
    maximum_lagrange: float = 100.0
    fall_cost_limit: float = 0.0
    constraint_cost_limit: float = 0.01
    behavior_cloning_weight: float = 1.0
    online_behavior_weight: float = 5.0
    parent_churn_weight: float = 0.5
    ewc_weight: float = 5.0
    observation_clip: float = 8.0
    log_std_min: float = -5.0
    log_std_max: float = 0.5
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if not 8 <= self.hidden_dim <= 1024:
            raise ValueError("neural torque hidden dimension must be in [8, 1024]")
        if not 2 <= self.sequence_length <= 256:
            raise ValueError("neural torque sequence length must be in [2, 256]")
        if self.batch_size <= 0:
            raise ValueError("neural torque batch size must be positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("neural torque gamma and tau must be in (0, 1]")
        if min(self.actor_lr, self.critic_lr, self.alpha_lr, self.lagrange_lr) <= 0.0:
            raise ValueError("neural torque learning rates must be positive")
        if not 1e-4 <= self.initial_alpha <= 1.0:
            raise ValueError("neural torque initial alpha must be in [1e-4, 1]")
        if not self.log_std_min <= self.initial_log_std <= self.log_std_max:
            raise ValueError("initial log standard deviation must be within the configured range")
        if (
            min(
                self.maximum_lagrange,
                self.fall_cost_limit,
                self.constraint_cost_limit,
                self.behavior_cloning_weight,
                self.online_behavior_weight,
                self.parent_churn_weight,
                self.ewc_weight,
            )
            < 0.0
        ):
            raise ValueError("neural torque cost and retention weights must be non-negative")
        if not 1.0 <= self.observation_clip <= 20.0:
            raise ValueError("neural torque observation clip must be in [1, 20]")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("neural torque log standard-deviation range is invalid")


@dataclass(frozen=True)
class G1NeuralTorqueBCMetrics:
    epoch: int
    training_loss: float
    validation_loss: float
    action_limit_fraction: float
    finite: bool
    schema_version: str = "rosclaw.simforge.g1_neural_torque_bc_metrics.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1NeuralTorqueUpdate:
    update_index: int
    actor_updated: bool
    reward_critic_loss: float
    fall_critic_loss: float
    constraint_critic_loss: float
    actor_loss: float
    alpha: float
    fall_lagrange: float
    constraint_lagrange: float
    anchor_loss: float
    online_behavior_loss: float
    parent_churn_loss: float
    ewc_loss: float
    actor_transition_count: int
    critic_transition_count: int
    stale_actor_transition_count: int
    anchor_transition_count: int
    finite: bool
    schema_version: str = "rosclaw.simforge.g1_neural_torque_update.v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1NeuralTorqueReplay:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    fall_costs: np.ndarray
    constraint_costs: np.ndarray
    terminals: np.ndarray
    parent_actions: np.ndarray
    partitions: np.ndarray
    policy_lags: np.ndarray

    def __post_init__(self) -> None:
        count = len(np.asarray(self.observations))
        observation_dim = len(G1_NEURAL_TORQUE_OBSERVATIONS)
        action_dim = len(G1_HARD_TORQUE_LIMITS)
        expected = {
            "observations": (count, None, observation_dim),
            "actions": (count, action_dim),
            "next_observations": (count, None, observation_dim),
            "rewards": (count, 1),
            "fall_costs": (count, 1),
            "constraint_costs": (count, 1),
            "terminals": (count, 1),
            "parent_actions": (count, action_dim),
            "partitions": (count,),
            "policy_lags": (count,),
        }
        arrays = {name: np.asarray(getattr(self, name)) for name in expected}
        if count == 0:
            raise ValueError("neural torque replay must not be empty")
        if (
            arrays["observations"].ndim != 3
            or arrays["next_observations"].shape != arrays["observations"].shape
        ):
            raise ValueError("neural torque replay sequences are misaligned")
        if arrays["observations"].shape[2] != observation_dim:
            raise ValueError("neural torque replay observation dimension is invalid")
        for name, shape in expected.items():
            if len(shape) == 3 and shape[1] is None:
                continue
            if arrays[name].shape != shape:
                raise ValueError(f"neural torque replay {name} has the wrong shape")
        numeric = tuple(value for name, value in arrays.items() if name != "partitions")
        if any(not np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("neural torque replay contains non-finite values")
        partitions = arrays["partitions"].astype(np.int8)
        if not set(map(int, np.unique(partitions))).issubset({_RECENT, _ANCHOR, _BOUNDARY}):
            raise ValueError("neural torque replay has an unknown partition")
        lags = arrays["policy_lags"].astype(np.int64)
        if np.any(lags < 0):
            raise ValueError("neural torque replay policy lags must be non-negative")
        for name, value in arrays.items():
            if name == "partitions":
                value = partitions
            elif name == "policy_lags":
                value = lags
            elif name not in {"partitions", "policy_lags"}:
                value = value.astype(np.float32)
            object.__setattr__(self, name, value)

    @property
    def count(self) -> int:
        return len(self.observations)

    @classmethod
    def combine(cls, *replays: G1NeuralTorqueReplay) -> G1NeuralTorqueReplay:
        if not replays:
            raise ValueError("at least one neural torque replay is required")
        sequence_lengths = {value.observations.shape[1] for value in replays}
        if len(sequence_lengths) != 1:
            raise ValueError("neural torque replay sequence lengths must match")
        return cls(
            **{
                name: np.concatenate([getattr(value, name) for value in replays], axis=0)
                for name in (
                    "observations",
                    "actions",
                    "next_observations",
                    "rewards",
                    "fall_costs",
                    "constraint_costs",
                    "terminals",
                    "parent_actions",
                    "partitions",
                    "policy_lags",
                )
            }
        )


class _RecurrentActor(nn.Module):
    def __init__(self, config: G1NeuralTorqueLearnerConfig, limits: np.ndarray) -> None:
        super().__init__()
        self.gru = nn.GRU(
            len(G1_NEURAL_TORQUE_OBSERVATIONS),
            config.hidden_dim,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_dim, 2 * len(G1_HARD_TORQUE_LIMITS))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[len(G1_HARD_TORQUE_LIMITS) :].fill_(config.initial_log_std)
        self.register_buffer("action_limits", torch.as_tensor(limits, dtype=torch.float32))
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def distribution_parameters(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.gru(observations)
        mean, log_std = self.head(encoded[:, -1]).chunk(2, dim=-1)
        return mean, torch.clamp(log_std, self.log_std_min, self.log_std_max)

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_parameters(observations)
        return torch.tanh(mean) * self.action_limits

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(observations)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        latent = distribution.rsample()
        unit = torch.tanh(latent)
        action = unit * self.action_limits
        # Optimize entropy in normalized action space.  Including the fixed
        # physical torque scale adds a large constant (29 joints) to every
        # target and destabilizes early online critic updates.
        jacobian = 1.0 - unit.square()
        log_probability = (distribution.log_prob(latent) - torch.log(jacobian.clamp_min(1e-6))).sum(
            dim=-1, keepdim=True
        )
        return action, log_probability


class _SequenceQ(nn.Module):
    def __init__(self, config: G1NeuralTorqueLearnerConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            len(G1_NEURAL_TORQUE_OBSERVATIONS),
            config.hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim + len(G1_HARD_TORQUE_LIMITS), config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor, action_ratio: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(observations)
        return self.head(torch.cat((encoded[:, -1], action_ratio), dim=-1))


class _TwinSequenceCritic(nn.Module):
    def __init__(self, config: G1NeuralTorqueLearnerConfig) -> None:
        super().__init__()
        self.q1 = _SequenceQ(config)
        self.q2 = _SequenceQ(config)

    def forward(
        self,
        observations: torch.Tensor,
        action_ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observations, action_ratio), self.q2(observations, action_ratio)


class G1ContinualTorqueActorCritic:
    """End-to-end recurrent torque learner with retention and cost critics."""

    def __init__(
        self,
        config: G1NeuralTorqueLearnerConfig,
        *,
        safety: G1TorqueSafetyConfig | None = None,
    ) -> None:
        self.config = config
        self.safety = safety or G1TorqueSafetyConfig()
        torch.manual_seed(config.seed)
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA neural torque learner requested without CUDA")
        self.device = torch.device(config.device)
        limits = np.asarray(G1_HARD_TORQUE_LIMITS) * self.safety.torque_guard_scale
        self.action_limits = limits.astype(np.float32)
        self.actor = _RecurrentActor(config, limits).to(self.device)
        self.parent_actor = copy.deepcopy(self.actor).eval()
        self._freeze_parent()
        self.reward_critic = _TwinSequenceCritic(config).to(self.device)
        self.fall_critic = _TwinSequenceCritic(config).to(self.device)
        self.constraint_critic = _TwinSequenceCritic(config).to(self.device)
        self.reward_target = copy.deepcopy(self.reward_critic).eval()
        self.fall_target = copy.deepcopy(self.fall_critic).eval()
        self.constraint_target = copy.deepcopy(self.constraint_critic).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        critic_parameters = (
            list(self.reward_critic.parameters())
            + list(self.fall_critic.parameters())
            + list(self.constraint_critic.parameters())
        )
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=config.critic_lr)
        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha),
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=config.alpha_lr)
        self.target_entropy = -float(len(G1_HARD_TORQUE_LIMITS))
        self.fall_lagrange = 0.0
        self.constraint_lagrange = 0.0
        self.update_index = 0
        self.observation_mean: np.ndarray = np.zeros(
            len(G1_NEURAL_TORQUE_OBSERVATIONS), dtype=np.float32
        )
        self.observation_std: np.ndarray = np.ones(
            len(G1_NEURAL_TORQUE_OBSERVATIONS), dtype=np.float32
        )
        self._anchor_parameters: dict[str, torch.Tensor] = {}
        self._fisher: dict[str, torch.Tensor] = {}
        self._pending_motion_prior: G1MotionPriorArtifact | None = None
        self._pending_motion_prior_fraction: float = 0.0
        self.motion_prior_artifact_hash: str | None = None
        self.motion_prior_transfer_fraction: float = 0.0
        # ``deepcopy`` does not preserve cuDNN's packed recurrent-weight
        # layout. Pack every actor/critic once up front so the first online
        # update does not silently take the slow fallback path.
        self._flatten_recurrent_parameters()

    def install_motion_prior(
        self,
        artifact: G1MotionPriorArtifact,
        *,
        expected_body_hash: str,
        fraction: float = 1.0,
    ) -> None:
        """Stage an audited kinematic representation for the next BC pass.

        The prior contributes GRU representation weights only.  Its prediction
        head cannot become a torque head, and the torque actor still requires
        teacher BC plus sealed physics validation.
        """

        if self.update_index or self._anchor_parameters:
            raise ValueError("motion prior must be installed before torque consolidation")
        if artifact.body_hash != expected_body_hash:
            raise ValueError("motion-prior body hash does not match the torque learner")
        if artifact.hidden_dim != self.config.hidden_dim:
            raise ValueError("motion-prior hidden dimension does not match the torque learner")
        feature_count = len(artifact.feature_names)
        if (
            feature_count != _G1_MOTION_PRIOR_FEATURE_COUNT
            or artifact.feature_names != G1_NEURAL_TORQUE_OBSERVATIONS[:feature_count]
        ):
            raise ValueError("motion-prior feature order does not match torque proprioception")
        if artifact.source_truth_level != "T4" or artifact.action_semantics != "ABSENT":
            raise ValueError("motion-prior source semantics are not eligible")
        if artifact.activation_ceiling != "SIM_ONLY_REPRESENTATION_INITIALIZATION":
            raise ValueError("motion-prior activation ceiling is not SIM_ONLY")
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("motion-prior transfer fraction must be in (0, 1]")
        self._pending_motion_prior = artifact
        self._pending_motion_prior_fraction = fraction

    def pretrain_behavior(
        self,
        training: tuple[G1TeacherTorqueEpisode, ...],
        *,
        validation: tuple[G1TeacherTorqueEpisode, ...] = (),
        epochs: int = 10,
        stride: int = 4,
        minimum_end_fraction: float = 0.0,
    ) -> tuple[G1NeuralTorqueBCMetrics, ...]:
        if not training or epochs <= 0 or stride <= 0:
            raise ValueError("neural torque BC requires data, positive epochs, and stride")
        if not 0.0 <= minimum_end_fraction <= 0.9:
            raise ValueError("neural torque BC end fraction must be in [0, 0.9]")
        self.actor.gru.flatten_parameters()
        all_observations = np.concatenate([item.observations for item in training], axis=0)
        self.observation_mean = all_observations.mean(axis=0).astype(np.float32)
        self.observation_std = np.maximum(all_observations.std(axis=0), 1e-3).astype(np.float32)
        if self._pending_motion_prior is not None:
            self._apply_motion_prior(
                self._pending_motion_prior,
                fraction=self._pending_motion_prior_fraction,
            )
            self.motion_prior_artifact_hash = self._pending_motion_prior.artifact_hash
            self.motion_prior_transfer_fraction = self._pending_motion_prior_fraction
            self._pending_motion_prior = None
            self._pending_motion_prior_fraction = 0.0
        train_sequences, train_actions = _teacher_sequences(
            training,
            sequence_length=self.config.sequence_length,
            stride=stride,
            minimum_end_fraction=minimum_end_fraction,
        )
        if validation:
            validation_sequences, validation_actions = _teacher_sequences(
                validation,
                sequence_length=self.config.sequence_length,
                stride=stride,
                minimum_end_fraction=minimum_end_fraction,
            )
        else:
            validation_sequences, validation_actions = train_sequences, train_actions
        rng = np.random.default_rng(self.config.seed)
        metrics: list[G1NeuralTorqueBCMetrics] = []
        for epoch in range(epochs):
            order = rng.permutation(len(train_sequences))
            losses: list[float] = []
            for start in range(0, len(order), self.config.batch_size):
                indices = order[start : start + self.config.batch_size]
                observations = self._observation_tensor(train_sequences[indices])
                targets = torch.as_tensor(
                    train_actions[indices], dtype=torch.float32, device=self.device
                )
                targets = torch.clamp(
                    targets,
                    -self.actor.action_limits,
                    self.actor.action_limits,
                )
                prediction = self.actor.deterministic(observations)
                loss = torch.nn.functional.mse_loss(
                    prediction / self.actor.action_limits,
                    targets / self.actor.action_limits,
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
                self.actor_optimizer.step()
                losses.append(float(loss.detach().item()))
            validation_loss, saturation = self._behavior_loss(
                validation_sequences,
                validation_actions,
            )
            training_loss = float(sum(losses) / len(losses))
            metrics.append(
                G1NeuralTorqueBCMetrics(
                    epoch=epoch,
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                    action_limit_fraction=saturation,
                    finite=all(math.isfinite(value) for value in (training_loss, validation_loss)),
                )
            )
        self._consolidate_parent(train_sequences, train_actions)
        return tuple(metrics)

    def update(
        self,
        replay: G1NeuralTorqueReplay,
        *,
        update_actor: bool = True,
    ) -> G1NeuralTorqueUpdate:
        if replay.observations.shape[1] != self.config.sequence_length:
            raise ValueError("neural torque replay sequence length does not match learner")
        fresh = np.flatnonzero((replay.partitions == _RECENT) & (replay.policy_lags <= 1))
        anchors = np.flatnonzero(replay.partitions == _ANCHOR)
        if update_actor and len(fresh) < self.config.batch_size:
            raise ValueError("neural torque actor requires enough fresh online transitions")
        if update_actor and len(anchors) == 0:
            raise ValueError("neural torque actor update requires historical anchors")
        rng = np.random.default_rng(self.config.seed + self.update_index)
        critic_indices = rng.choice(
            replay.count,
            self.config.batch_size,
            replace=replay.count < self.config.batch_size,
        )
        observations = self._observation_tensor(replay.observations[critic_indices])
        next_observations = self._observation_tensor(replay.next_observations[critic_indices])
        actions = self._tensor(replay.actions[critic_indices])
        rewards = self._tensor(replay.rewards[critic_indices])
        fall_costs = self._tensor(replay.fall_costs[critic_indices])
        constraint_costs = self._tensor(replay.constraint_costs[critic_indices])
        terminals = self._tensor(replay.terminals[critic_indices])
        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(next_observations)
            next_ratio = next_actions / self.actor.action_limits
            reward_target = rewards + self.config.gamma * (1.0 - terminals) * (
                torch.minimum(*self.reward_target(next_observations, next_ratio))
                - self.log_alpha.exp().detach() * next_log_probability
            )
            fall_target = fall_costs + self.config.gamma * (1.0 - terminals) * torch.maximum(
                *self.fall_target(next_observations, next_ratio)
            )
            constraint_target = constraint_costs + self.config.gamma * (
                1.0 - terminals
            ) * torch.maximum(*self.constraint_target(next_observations, next_ratio))
        action_ratio = actions / self.actor.action_limits
        reward_values = self.reward_critic(observations, action_ratio)
        fall_values = self.fall_critic(observations, action_ratio)
        constraint_values = self.constraint_critic(observations, action_ratio)
        reward_loss = sum(
            torch.nn.functional.mse_loss(value, reward_target) for value in reward_values
        )
        fall_loss = sum(torch.nn.functional.mse_loss(value, fall_target) for value in fall_values)
        constraint_loss = sum(
            torch.nn.functional.mse_loss(value, constraint_target) for value in constraint_values
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        (reward_loss + fall_loss + constraint_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.reward_critic.parameters())
            + list(self.fall_critic.parameters())
            + list(self.constraint_critic.parameters()),
            max_norm=10.0,
        )
        self.critic_optimizer.step()

        actor_loss = torch.zeros((), device=self.device)
        anchor_loss = torch.zeros((), device=self.device)
        online_behavior_loss = torch.zeros((), device=self.device)
        parent_churn = torch.zeros((), device=self.device)
        ewc_loss = torch.zeros((), device=self.device)
        if update_actor:
            actor_indices = rng.choice(fresh, self.config.batch_size, replace=False)
            anchor_indices = rng.choice(
                anchors,
                min(self.config.batch_size, len(anchors)),
                replace=False,
            )
            actor_observations = self._observation_tensor(replay.observations[actor_indices])
            anchor_observations = self._observation_tensor(replay.observations[anchor_indices])
            anchor_actions = self._tensor(replay.parent_actions[anchor_indices])
            sampled_actions, log_probability = self.actor.sample(actor_observations)
            behavior_actions = self._tensor(replay.actions[actor_indices])
            sampled_ratio = sampled_actions / self.actor.action_limits
            reward_q = torch.minimum(*self.reward_critic(actor_observations, sampled_ratio))
            fall_q = torch.maximum(*self.fall_critic(actor_observations, sampled_ratio))
            constraint_q = torch.maximum(*self.constraint_critic(actor_observations, sampled_ratio))
            anchor_prediction = self.actor.deterministic(anchor_observations)
            with torch.no_grad():
                parent_prediction = self.parent_actor.deterministic(anchor_observations)
            anchor_loss = torch.nn.functional.mse_loss(
                anchor_prediction / self.actor.action_limits,
                torch.clamp(
                    anchor_actions,
                    -self.actor.action_limits,
                    self.actor.action_limits,
                )
                / self.actor.action_limits,
            )
            parent_churn = torch.nn.functional.mse_loss(
                anchor_prediction / self.actor.action_limits,
                parent_prediction / self.actor.action_limits,
            )
            online_behavior_loss = torch.nn.functional.mse_loss(
                self.actor.deterministic(actor_observations) / self.actor.action_limits,
                torch.clamp(
                    behavior_actions,
                    -self.actor.action_limits,
                    self.actor.action_limits,
                )
                / self.actor.action_limits,
            )
            ewc_loss = self._ewc_loss()
            actor_loss = (
                (
                    self.log_alpha.exp().detach() * log_probability
                    - reward_q
                    + self.fall_lagrange * fall_q
                    + self.constraint_lagrange * constraint_q
                ).mean()
                + self.config.behavior_cloning_weight * anchor_loss
                + self.config.online_behavior_weight * online_behavior_loss
                + self.config.parent_churn_weight * parent_churn
                + self.config.ewc_weight * ewc_loss
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_optimizer.step()
            alpha_loss = -(self.log_alpha * (log_probability.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        self.fall_lagrange = _lagrange_update(
            self.fall_lagrange,
            float(fall_costs.mean().item()) - self.config.fall_cost_limit,
            self.config,
        )
        self.constraint_lagrange = _lagrange_update(
            self.constraint_lagrange,
            float(constraint_costs.mean().item()) - self.config.constraint_cost_limit,
            self.config,
        )
        _soft_update(self.reward_target, self.reward_critic, self.config.tau)
        _soft_update(self.fall_target, self.fall_critic, self.config.tau)
        _soft_update(self.constraint_target, self.constraint_critic, self.config.tau)
        values = (
            reward_loss,
            fall_loss,
            constraint_loss,
            actor_loss,
            self.log_alpha.exp(),
            anchor_loss,
            online_behavior_loss,
            parent_churn,
            ewc_loss,
        )
        finite = all(bool(torch.isfinite(value).all().item()) for value in values)
        result = G1NeuralTorqueUpdate(
            update_index=self.update_index,
            actor_updated=update_actor,
            reward_critic_loss=float(reward_loss.detach().item()),
            fall_critic_loss=float(fall_loss.detach().item()),
            constraint_critic_loss=float(constraint_loss.detach().item()),
            actor_loss=float(actor_loss.detach().item()),
            alpha=float(self.log_alpha.exp().detach().item()),
            fall_lagrange=self.fall_lagrange,
            constraint_lagrange=self.constraint_lagrange,
            anchor_loss=float(anchor_loss.detach().item()),
            online_behavior_loss=float(online_behavior_loss.detach().item()),
            parent_churn_loss=float(parent_churn.detach().item()),
            ewc_loss=float(ewc_loss.detach().item()),
            actor_transition_count=len(fresh) if update_actor else 0,
            critic_transition_count=replay.count,
            stale_actor_transition_count=int(
                np.count_nonzero((replay.partitions == _RECENT) & (replay.policy_lags > 1))
            ),
            anchor_transition_count=len(anchors),
            finite=finite,
        )
        self.update_index += 1
        return result

    def deterministic_action(self, observation_sequence: np.ndarray) -> np.ndarray:
        value = np.asarray(observation_sequence, dtype=np.float32)
        expected = (
            self.config.sequence_length,
            len(G1_NEURAL_TORQUE_OBSERVATIONS),
        )
        if value.shape != expected or not np.all(np.isfinite(value)):
            raise ValueError("neural torque inference sequence has the wrong shape")
        with torch.no_grad():
            action = self.actor.deterministic(self._observation_tensor(value[None]))[0]
        return action.cpu().numpy()

    def artifact_bytes(
        self,
        *,
        body_hash: str,
        parent_policy_hash: str,
        dataset_hash: str,
    ) -> bytes:
        state = self.actor.state_dict()
        # Canonical 1e-6 quantization removes backend-dependent last-bit CUDA
        # noise before a policy enters a discontinuous hybrid control loop.
        # In experiments, 1e-12 weight differences could select a different
        # fallback branch several seconds later despite identical losses.
        tensors = {
            "observation_mean": _quantized_export(self.observation_mean),
            "observation_std": _quantized_export(self.observation_std),
            "action_limits": _quantized_export(self.action_limits),
            "actor.gru.weight_ih_l0": _quantized_export(
                state["gru.weight_ih_l0"].detach().cpu().numpy()
            ),
            "actor.gru.weight_hh_l0": _quantized_export(
                state["gru.weight_hh_l0"].detach().cpu().numpy()
            ),
            "actor.gru.bias_ih_l0": _quantized_export(
                state["gru.bias_ih_l0"].detach().cpu().numpy()
            ),
            "actor.gru.bias_hh_l0": _quantized_export(
                state["gru.bias_hh_l0"].detach().cpu().numpy()
            ),
            "actor.head.weight": _quantized_export(
                state["head.weight"].detach().cpu().numpy()
            ),
            "actor.head.bias": _quantized_export(state["head.bias"].detach().cpu().numpy()),
        }
        return serialize_g1_neural_torque_artifact(
            body_hash=body_hash,
            parent_policy_hash=parent_policy_hash,
            dataset_hash=dataset_hash,
            hidden_dim=self.config.hidden_dim,
            observation_clip=self.config.observation_clip,
            update_index=self.update_index,
            safety=self.safety,
            tensors=tensors,
        )

    def actor_snapshot(self) -> dict[str, np.ndarray]:
        """Return a finite CPU snapshot for a simulator-side trust-region search."""

        return {
            name: value.detach().cpu().numpy().copy()
            for name, value in self.actor.state_dict().items()
        }

    def install_interpolated_actor(
        self,
        parent: dict[str, np.ndarray],
        proposal: dict[str, np.ndarray],
        *,
        fraction: float,
    ) -> None:
        """Install one bounded point on the parent-to-proposal actor segment."""

        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("neural torque actor interpolation fraction must be in [0, 1]")
        expected = self.actor.state_dict()
        if set(parent) != set(expected) or set(proposal) != set(expected):
            raise ValueError("neural torque actor snapshot tensor set mismatch")
        state: dict[str, torch.Tensor] = {}
        for name, expected_value in expected.items():
            before = np.asarray(parent[name], dtype=np.float32)
            after = np.asarray(proposal[name], dtype=np.float32)
            shape = tuple(expected_value.shape)
            if before.shape != shape or after.shape != shape:
                raise ValueError(f"neural torque actor snapshot shape mismatch: {name}")
            if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
                raise ValueError("neural torque actor snapshot contains non-finite values")
            value = before.astype(np.float64) + fraction * (
                after.astype(np.float64) - before.astype(np.float64)
            )
            state[name] = torch.as_tensor(value, dtype=expected_value.dtype, device=self.device)
        self.actor.load_state_dict(state)
        self.actor.gru.flatten_parameters()

    def checkpoint_bytes(self) -> bytes:
        payload = {
            "schema_version": "rosclaw.simforge.g1_neural_torque_checkpoint.v1",
            "config_hash": canonical_hash(asdict(self.config)),
            "safety_hash": canonical_hash(asdict(self.safety)),
            "actor": self.actor.state_dict(),
            "parent_actor": self.parent_actor.state_dict(),
            "reward_critic": self.reward_critic.state_dict(),
            "fall_critic": self.fall_critic.state_dict(),
            "constraint_critic": self.constraint_critic.state_dict(),
            "reward_target": self.reward_target.state_dict(),
            "fall_target": self.fall_target.state_dict(),
            "constraint_target": self.constraint_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "fall_lagrange": self.fall_lagrange,
            "constraint_lagrange": self.constraint_lagrange,
            "update_index": self.update_index,
            "observation_mean": torch.from_numpy(self.observation_mean),
            "observation_std": torch.from_numpy(self.observation_std),
            "anchor_parameters": self._anchor_parameters,
            "fisher": self._fisher,
            "motion_prior_artifact_hash": self.motion_prior_artifact_hash,
            "motion_prior_transfer_fraction": self.motion_prior_transfer_fraction,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return buffer.getvalue()

    def restore_checkpoint(self, checkpoint: bytes) -> None:
        if not checkpoint:
            raise ValueError("neural torque checkpoint must not be empty")
        payload = torch.load(io.BytesIO(checkpoint), map_location=self.device, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("neural torque checkpoint payload must be a mapping")
        if payload.get("schema_version") != "rosclaw.simforge.g1_neural_torque_checkpoint.v1":
            raise ValueError("unsupported neural torque checkpoint schema")
        if payload.get("config_hash") != canonical_hash(asdict(self.config)):
            raise ValueError("neural torque checkpoint learner configuration mismatch")
        if payload.get("safety_hash") != canonical_hash(asdict(self.safety)):
            raise ValueError("neural torque checkpoint safety configuration mismatch")
        required = (
            "actor",
            "parent_actor",
            "reward_critic",
            "fall_critic",
            "constraint_critic",
            "reward_target",
            "fall_target",
            "constraint_target",
            "actor_optimizer",
            "critic_optimizer",
            "alpha_optimizer",
            "log_alpha",
            "fall_lagrange",
            "constraint_lagrange",
            "update_index",
            "observation_mean",
            "observation_std",
            "anchor_parameters",
            "fisher",
            "torch_rng_state",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError("neural torque checkpoint is missing: " + ", ".join(missing))
        for name in (
            "actor",
            "parent_actor",
            "reward_critic",
            "fall_critic",
            "constraint_critic",
            "reward_target",
            "fall_target",
            "constraint_target",
        ):
            getattr(self, name).load_state_dict(payload[name])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"])
        self.fall_lagrange = float(payload["fall_lagrange"])
        self.constraint_lagrange = float(payload["constraint_lagrange"])
        self.update_index = int(payload["update_index"])
        self.observation_mean = payload["observation_mean"].cpu().numpy().astype(np.float32)
        self.observation_std = payload["observation_std"].cpu().numpy().astype(np.float32)
        self._anchor_parameters = {
            str(name): value.to(self.device) for name, value in payload["anchor_parameters"].items()
        }
        self._fisher = {
            str(name): value.to(self.device) for name, value in payload["fisher"].items()
        }
        prior_hash = payload.get("motion_prior_artifact_hash")
        self.motion_prior_artifact_hash = str(prior_hash) if prior_hash is not None else None
        self.motion_prior_transfer_fraction = float(
            payload.get("motion_prior_transfer_fraction", 0.0)
        )
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if self.device.type == "cuda":
            states = payload.get("cuda_rng_state")
            if states is None:
                raise ValueError("CUDA neural torque checkpoint lacks CUDA RNG state")
            torch.cuda.set_rng_state_all([value.cpu() for value in states])
        self._freeze_parent()
        self._flatten_recurrent_parameters()

    def _behavior_loss(
        self,
        sequences: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[float, float]:
        losses: list[float] = []
        saturated = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(sequences), self.config.batch_size):
                observations = self._observation_tensor(
                    sequences[start : start + self.config.batch_size]
                )
                targets = self._tensor(actions[start : start + self.config.batch_size])
                targets = torch.clamp(
                    targets,
                    -self.actor.action_limits,
                    self.actor.action_limits,
                )
                prediction = self.actor.deterministic(observations)
                loss = torch.nn.functional.mse_loss(
                    prediction / self.actor.action_limits,
                    targets / self.actor.action_limits,
                )
                losses.append(float(loss.item()) * len(observations))
                saturated += int(
                    torch.count_nonzero(
                        torch.abs(prediction) >= self.actor.action_limits * 0.999
                    ).item()
                )
                total += prediction.numel()
        return float(sum(losses) / len(sequences)), saturated / max(1, total)

    def _apply_motion_prior(
        self,
        artifact: G1MotionPriorArtifact,
        *,
        fraction: float,
    ) -> None:
        feature_count = len(artifact.feature_names)
        prior_mean = np.asarray(artifact.observation_mean, dtype=np.float32)
        prior_std = np.asarray(artifact.observation_std, dtype=np.float32)
        actor_mean = self.observation_mean[:feature_count]
        actor_std = self.observation_std[:feature_count]
        required_tensors = {
            "gru.weight_ih_l0",
            "gru.weight_hh_l0",
            "gru.bias_ih_l0",
            "gru.bias_hh_l0",
        }
        if not required_tensors.issubset(artifact.tensors):
            raise ValueError("motion-prior GRU tensor set is incomplete")
        if (
            prior_mean.shape != (feature_count,)
            or prior_std.shape != (feature_count,)
            or not np.all(np.isfinite(prior_mean))
            or not np.all(np.isfinite(prior_std))
            or np.any(prior_std <= 1e-6)
        ):
            raise ValueError("motion-prior normalization contract is invalid")
        scale = actor_std / prior_std
        offset = (actor_mean - prior_mean) / prior_std
        weight_ih = np.asarray(artifact.tensors["gru.weight_ih_l0"], dtype=np.float32)
        weight_hh = np.asarray(artifact.tensors["gru.weight_hh_l0"], dtype=np.float32)
        bias_ih = np.asarray(artifact.tensors["gru.bias_ih_l0"], dtype=np.float32)
        bias_hh = np.asarray(artifact.tensors["gru.bias_hh_l0"], dtype=np.float32)
        expected = (3 * self.config.hidden_dim, feature_count)
        if (
            weight_ih.shape != expected
            or weight_hh.shape
            != (3 * self.config.hidden_dim, self.config.hidden_dim)
            or bias_ih.shape != (3 * self.config.hidden_dim,)
            or bias_hh.shape != (3 * self.config.hidden_dim,)
        ):
            raise ValueError("motion-prior GRU tensor shape is invalid")
        if any(
            not np.all(np.isfinite(value))
            for value in (scale, offset, weight_ih, weight_hh, bias_ih, bias_hh)
        ):
            raise ValueError("motion-prior transfer contains non-finite values")
        transformed_weight = weight_ih * scale[None, :]
        transformed_bias = bias_ih + weight_ih @ offset
        with torch.no_grad():
            actor_weight = self.actor.gru.weight_ih_l0
            transfers = (
                (actor_weight[:, :feature_count], transformed_weight),
                (self.actor.gru.weight_hh_l0, weight_hh),
                (self.actor.gru.bias_ih_l0, transformed_bias),
                (self.actor.gru.bias_hh_l0, bias_hh),
            )
            for target, source in transfers:
                prior_value = torch.as_tensor(
                    source,
                    dtype=actor_weight.dtype,
                    device=self.device,
                )
                target.copy_(target * (1.0 - fraction) + prior_value * fraction)
        self.actor.gru.flatten_parameters()

    def _consolidate_parent(self, sequences: np.ndarray, actions: np.ndarray) -> None:
        self.parent_actor.load_state_dict(copy.deepcopy(self.actor.state_dict()))
        self._freeze_parent()
        sample_count = min(len(sequences), max(self.config.batch_size, 1024))
        rng = np.random.default_rng(self.config.seed ^ 0xE0C)
        indices = rng.choice(len(sequences), sample_count, replace=False)
        observations = self._observation_tensor(sequences[indices])
        targets = self._tensor(actions[indices])
        prediction = self.actor.deterministic(observations)
        loss = torch.nn.functional.mse_loss(
            prediction / self.actor.action_limits,
            torch.clamp(targets, -self.actor.action_limits, self.actor.action_limits)
            / self.actor.action_limits,
        )
        self.actor.zero_grad(set_to_none=True)
        loss.backward()
        self._anchor_parameters = {
            name: parameter.detach().clone() for name, parameter in self.actor.named_parameters()
        }
        self._fisher = {
            name: (
                parameter.grad.detach().square().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
            for name, parameter in self.actor.named_parameters()
        }
        self.actor.zero_grad(set_to_none=True)

    def _ewc_loss(self) -> torch.Tensor:
        if not self._anchor_parameters or not self._fisher:
            return torch.zeros((), device=self.device)
        numerator = torch.zeros((), device=self.device)
        count = 0
        for name, parameter in self.actor.named_parameters():
            numerator = (
                numerator
                + (self._fisher[name] * (parameter - self._anchor_parameters[name]).square()).sum()
            )
            count += parameter.numel()
        return numerator / max(1, count)

    def _observation_tensor(self, value: np.ndarray) -> torch.Tensor:
        normalized = np.clip(
            (np.asarray(value, dtype=np.float32) - self.observation_mean) / self.observation_std,
            -self.config.observation_clip,
            self.config.observation_clip,
        )
        return torch.as_tensor(normalized, dtype=torch.float32, device=self.device)

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _freeze_parent(self) -> None:
        self.parent_actor.eval()
        self.parent_actor.gru.flatten_parameters()
        for parameter in self.parent_actor.parameters():
            parameter.requires_grad_(False)

    def _flatten_recurrent_parameters(self) -> None:
        self.actor.gru.flatten_parameters()
        self.parent_actor.gru.flatten_parameters()
        for twin in (
            self.reward_critic,
            self.fall_critic,
            self.constraint_critic,
            self.reward_target,
            self.fall_target,
            self.constraint_target,
        ):
            twin.q1.gru.flatten_parameters()
            twin.q2.gru.flatten_parameters()


def teacher_dataset_hash(episodes: tuple[G1TeacherTorqueEpisode, ...]) -> str:
    if not episodes:
        raise ValueError("neural torque dataset hash requires at least one episode")
    digest = hashlib.sha256()
    for index, episode in enumerate(episodes):
        digest.update(str(index).encode())
        for value in (episode.observations, episode.actions, episode.parent_actions):
            array = np.ascontiguousarray(value, dtype=np.float32)
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def neural_torque_replay_hash(replay: G1NeuralTorqueReplay) -> str:
    """Bind continual updates to the exact replay tensors and partitions."""

    digest = hashlib.sha256()
    for name in (
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "fall_costs",
        "constraint_costs",
        "terminals",
        "parent_actions",
        "partitions",
        "policy_lags",
    ):
        value = np.ascontiguousarray(getattr(replay, name))
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return "sha256:" + digest.hexdigest()


def teacher_replay(
    episodes: tuple[G1TeacherTorqueEpisode, ...],
    *,
    sequence_length: int,
    stride: int = 10,
) -> G1NeuralTorqueReplay:
    sequences, actions, next_sequences, parent_actions, terminals = _episode_transitions(
        episodes,
        sequence_length=sequence_length,
        stride=stride,
    )
    count = len(sequences)
    return G1NeuralTorqueReplay(
        observations=sequences,
        actions=actions,
        next_observations=next_sequences,
        rewards=np.zeros((count, 1), dtype=np.float32),
        fall_costs=np.zeros((count, 1), dtype=np.float32),
        constraint_costs=np.zeros((count, 1), dtype=np.float32),
        terminals=terminals,
        parent_actions=parent_actions,
        partitions=np.full(count, _ANCHOR, dtype=np.int8),
        policy_lags=np.full(count, 2, dtype=np.int64),
    )


def online_replay(
    episode: G1TeacherTorqueEpisode,
    *,
    sequence_length: int,
    task_score: float,
    fell: bool,
    critical_failure: bool,
    projection_fallback_rate: float,
    policy_lag: int = 0,
    stride: int = 10,
) -> G1NeuralTorqueReplay:
    if not math.isfinite(task_score) or not -20.0 <= task_score <= 20.0:
        raise ValueError("neural torque online task score must be finite and bounded")
    if not 0.0 <= projection_fallback_rate <= 1.0:
        raise ValueError("projection fallback rate must be in [0, 1]")
    sequences, actions, next_sequences, parent_actions, terminals = _episode_transitions(
        (episode,),
        sequence_length=sequence_length,
        stride=stride,
    )
    count = len(sequences)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float32) * 0.85
    action_ratio = actions / limits
    parent_ratio = np.clip(parent_actions, -limits, limits) / limits
    imitation = -np.mean(np.square(action_ratio - parent_ratio), axis=1, keepdims=True)
    smoothness: np.ndarray = np.zeros((count, 1), dtype=np.float32)
    if count > 1:
        smoothness[1:] = -np.mean(np.square(np.diff(action_ratio, axis=0)), axis=1)[:, None]
    rewards = (0.05 * imitation + 0.02 * smoothness).astype(np.float32)
    rewards[-1, 0] += task_score
    fall_costs: np.ndarray = np.zeros((count, 1), dtype=np.float32)
    fall_costs[-1, 0] = float(fell)
    constraint_costs: np.ndarray = np.full(
        (count, 1),
        projection_fallback_rate,
        dtype=np.float32,
    )
    if critical_failure:
        constraint_costs[-1, 0] = max(1.0, constraint_costs[-1, 0])
    return G1NeuralTorqueReplay(
        observations=sequences,
        actions=actions,
        next_observations=next_sequences,
        rewards=rewards,
        fall_costs=fall_costs,
        constraint_costs=constraint_costs,
        terminals=terminals,
        parent_actions=parent_actions,
        partitions=np.full(
            count,
            _BOUNDARY if critical_failure else _RECENT,
            dtype=np.int8,
        ),
        policy_lags=np.full(count, policy_lag, dtype=np.int64),
    )


def recovery_online_replay(
    episode: G1TeacherTorqueEpisode,
    *,
    trajectory: Mapping[str, np.ndarray],
    sequence_length: int,
    recovery_score: float,
    fell: bool,
    critical_failure: bool,
    projection_fallback_rate: float,
    recovery_start_phase: float = 0.55,
    policy_lag: int = 0,
    stride: int = 10,
) -> G1NeuralTorqueReplay:
    """Build dense, time-aligned post-kick replay from T1 MuJoCo evidence."""

    if not -20.0 <= recovery_score <= 20.0 or not math.isfinite(recovery_score):
        raise ValueError("recovery score must be finite and bounded")
    if not 0.0 <= projection_fallback_rate <= 1.0:
        raise ValueError("recovery projection fallback rate must be in [0, 1]")
    if not 0.4 <= recovery_start_phase <= 0.95:
        raise ValueError("recovery start phase must be in [0.4, 0.95]")
    if stride <= 0:
        raise ValueError("recovery replay stride must be positive")
    required = (
        "policy_phase",
        "support_foot_slip",
        "com_y_relative",
        "left_foot_contact",
        "right_foot_contact",
    )
    try:
        values = {name: np.asarray(trajectory[name], dtype=np.float64) for name in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("recovery trajectory signals are missing or non-numeric") from exc
    trace_lengths = {len(value) for value in values.values() if value.ndim >= 1}
    if any(value.ndim != 1 for value in values.values()) or len(trace_lengths) != 1:
        raise ValueError("recovery trajectory signals are missing or misaligned")
    trace_count = trace_lengths.pop()
    if trace_count <= 0 or len(episode.observations) != trace_count * 10:
        raise ValueError("recovery replay requires exact 500 Hz to 50 Hz alignment")
    if any(not np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError("recovery trajectory contains non-finite values")

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    parent_actions: list[np.ndarray] = []
    rewards: list[list[float]] = []
    fall_costs: list[list[float]] = []
    constraint_costs: list[list[float]] = []
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float32) * 0.85
    final_end = len(episode.observations) - 2
    for end in range(sequence_length - 1, final_end + 1, stride):
        trace_index = min(end // 10, trace_count - 1)
        if float(values["policy_phase"][trace_index]) < recovery_start_phase:
            continue
        start = end - sequence_length + 1
        observation = episode.observations[start : end + 1]
        action = episode.actions[end]
        parent = episode.parent_actions[end]
        gravity = observation[-1, 58:61]
        tilt_cost = float(np.dot(gravity[:2], gravity[:2]))
        velocity_cost = float(np.mean(np.square(observation[-1, 29:58] / 10.0)))
        action_ratio = np.clip(action, -limits, limits) / limits
        parent_ratio = np.clip(parent, -limits, limits) / limits
        imitation_cost = float(np.mean(np.square(action_ratio - parent_ratio)))
        slip = abs(float(values["support_foot_slip"][trace_index]))
        com_offset = abs(float(values["com_y_relative"][trace_index]))
        double_support = float(
            bool(values["left_foot_contact"][trace_index])
            and bool(values["right_foot_contact"][trace_index])
        )
        reward = (
            0.02 * double_support
            - 0.50 * tilt_cost
            - 0.05 * velocity_cost
            - 1.50 * min(slip, 0.20)
            - 0.50 * min(com_offset, 0.30)
            - 0.05 * imitation_cost
        )
        constraint = max(
            projection_fallback_rate,
            float(slip > 0.04),
            float(tilt_cost > 0.20),
            float(com_offset > 0.12),
        )
        observations.append(observation)
        next_observations.append(episode.observations[start + 1 : end + 2])
        actions.append(action)
        parent_actions.append(parent)
        rewards.append([float(np.clip(reward, -2.0, 0.1))])
        fall_costs.append([0.0])
        constraint_costs.append([constraint])
    if not observations:
        raise ValueError("recovery trajectory contains no eligible transitions")
    rewards[-1][0] += 0.25 * recovery_score
    fall_costs[-1][0] = float(fell)
    if critical_failure:
        constraint_costs[-1][0] = 1.0
    count = len(observations)
    terminals = np.zeros((count, 1), dtype=np.float32)
    terminals[-1, 0] = 1.0
    return G1NeuralTorqueReplay(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        fall_costs=np.asarray(fall_costs, dtype=np.float32),
        constraint_costs=np.asarray(constraint_costs, dtype=np.float32),
        terminals=terminals,
        parent_actions=np.asarray(parent_actions, dtype=np.float32),
        partitions=np.full(count, _BOUNDARY if critical_failure else _RECENT, dtype=np.int8),
        policy_lags=np.full(count, policy_lag, dtype=np.int64),
    )


def _teacher_sequences(
    episodes: tuple[G1TeacherTorqueEpisode, ...],
    *,
    sequence_length: int,
    stride: int,
    minimum_end_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    sequence_rows: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in episodes:
        if len(episode.observations) <= sequence_length:
            continue
        first_end = max(
            sequence_length - 1,
            int(math.ceil((len(episode.observations) - 1) * minimum_end_fraction)),
        )
        for end in range(first_end, len(episode.observations), stride):
            start = end - sequence_length + 1
            sequence_rows.append(episode.observations[start : end + 1])
            actions.append(episode.actions[end])
    if not sequence_rows:
        raise ValueError("teacher episodes are too short for the recurrent sequence length")
    return np.asarray(sequence_rows, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def _episode_transitions(
    episodes: tuple[G1TeacherTorqueEpisode, ...],
    *,
    sequence_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    parent_actions: list[np.ndarray] = []
    terminals: list[list[float]] = []
    for episode in episodes:
        final_end = len(episode.observations) - 2
        for end in range(sequence_length - 1, final_end + 1, stride):
            start = end - sequence_length + 1
            observations.append(episode.observations[start : end + 1])
            next_observations.append(episode.observations[start + 1 : end + 2])
            actions.append(episode.actions[end])
            parent_actions.append(episode.parent_actions[end])
            terminals.append([float(end + stride > final_end)])
    if not observations:
        raise ValueError("neural torque episodes are too short for replay")
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(next_observations, dtype=np.float32),
        np.asarray(parent_actions, dtype=np.float32),
        np.asarray(terminals, dtype=np.float32),
    )


def _lagrange_update(
    value: float,
    violation: float,
    config: G1NeuralTorqueLearnerConfig,
) -> float:
    return min(
        config.maximum_lagrange,
        max(0.0, value + config.lagrange_lr * violation),
    )


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_value, source_value in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_value.mul_(1.0 - tau).add_(source_value, alpha=tau)


def _quantized_export(value: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(value, dtype=np.float64), decimals=6).astype(np.float32)


__all__ = [
    "G1ContinualTorqueActorCritic",
    "G1NeuralTorqueBCMetrics",
    "G1NeuralTorqueLearnerConfig",
    "G1NeuralTorqueReplay",
    "G1NeuralTorqueUpdate",
    "online_replay",
    "neural_torque_replay_hash",
    "teacher_dataset_hash",
    "teacher_replay",
]
