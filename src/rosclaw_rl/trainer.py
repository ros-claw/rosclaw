"""
Training orchestrator for ROSClaw RL.

Integrates OpenClaw-RL training methods (GRPO, OPD, Combine) with physical robot data.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .collector import PhysicalDataCollector, RobotTrajectory
from .config import (
    DataConfig,
    ModelConfig,
    RewardConfig,
    ROSClawRLConfig,
    TrainingHyperparameters,
    TrainingMode,
)
from .converter import RLBatch, RLTrainingSample, TrajectoryConverter
from .rewards.base import BaseRewardModel
from .rewards.mprm import MPRMRewardModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics collection."""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    kl_divergence: float = 0.0
    learning_rate: float = 0.0
    num_trajectories: int = 0
    num_samples: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "kl_divergence": self.kl_divergence,
            "learning_rate": self.learning_rate,
            "num_trajectories": self.num_trajectories,
            "num_samples": self.num_samples,
            **self.extra,
        }


@dataclass
class TrainingConfig:
    """Configuration for training run."""
    config: ROSClawRLConfig
    checkpoint_dir: Path
    log_dir: Path
    num_iterations: int = 100
    trajectories_per_iteration: int = 32
    evaluation_frequency: int = 10
    checkpoint_frequency: int = 10


class RLTrainer:
    """
    Main training orchestrator for ROSClaw RL.

    Integrates OpenClaw-RL's training methods with physical robot data collection.
    Supports GRPO, OPD, and Combine training modes.
    """

    def __init__(
        self,
        config: TrainingConfig,
        reward_model: Optional[BaseRewardModel] = None,
        data_collector: Optional[PhysicalDataCollector] = None,
        trajectory_converter: Optional[TrajectoryConverter] = None,
    ):
        self.config = config
        self.rl_config = config.config

        # Initialize components
        self.reward_model = reward_model or self._create_default_reward_model()
        self.data_collector = data_collector or self._create_default_data_collector()
        self.trajectory_converter = trajectory_converter or self._create_default_converter()

        # Training state
        self.iteration = 0
        self.metrics_history: List[TrainingMetrics] = []
        self.current_policy: Optional[Any] = None  # Placeholder for policy model

        # Callbacks
        self.on_iteration_complete: Optional[Callable[[TrainingMetrics], None]] = None
        self.on_checkpoint_saved: Optional[Callable[[Path], None]] = None

    def _create_default_reward_model(self) -> BaseRewardModel:
        """Create default reward model from config."""
        reward_config = self.rl_config.reward
        return MPRMRewardModel(
            mode=reward_config.mode.value,
            success_reward=reward_config.success_reward,
            failure_reward=reward_config.failure_reward,
            step_penalty=reward_config.step_penalty,
            vlm_model=reward_config.vlm_model,
            vlm_api_url=reward_config.vlm_api_url,
            use_proprioception=reward_config.use_proprioception,
            use_vision=reward_config.use_vision,
            use_force=reward_config.use_force,
            use_audio=reward_config.use_audio,
        )

    def _create_default_data_collector(self) -> PhysicalDataCollector:
        """Create default data collector from config."""
        return PhysicalDataCollector(config=self.rl_config)

    def _create_default_converter(self) -> TrajectoryConverter:
        """Create default trajectory converter."""
        return TrajectoryConverter(
            image_size=(224, 224),
            max_images_per_trajectory=8,
        )

    def train(self) -> List[TrainingMetrics]:
        """
        Main training loop.

        Implements the RL training loop:
        1. Collect trajectories using current policy
        2. Compute rewards
        3. Train policy using selected method (GRPO/OPD/Combine)
        4. Update policy and repeat

        Returns:
            List of training metrics for each iteration
        """
        logger.info(f"Starting {self.rl_config.training_mode.value} training")
        logger.info(f"Iterations: {self.config.num_iterations}")
        logger.info(f"Trajectories per iteration: {self.config.trajectories_per_iteration}")

        self.metrics_history = []

        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            logger.info(f"=== Iteration {iteration + 1}/{self.config.num_iterations} ===")

            # 1. Collect trajectories
            trajectories = self._collect_trajectories()
            logger.info(f"Collected {len(trajectories)} trajectories")

            # 2. Compute rewards
            self._compute_rewards(trajectories)

            # 3. Convert to training format
            batch = self.trajectory_converter.convert_batch(trajectories)
            logger.info(f"Converted to {len(batch)} training samples")

            # 4. Train policy
            metrics = self._train_iteration(batch)
            self.metrics_history.append(metrics)

            # 5. Log metrics
            self._log_metrics(metrics)

            # 6. Evaluate if needed
            if (iteration + 1) % self.config.evaluation_frequency == 0:
                self._evaluate()

            # 7. Save checkpoint if needed
            if (iteration + 1) % self.config.checkpoint_frequency == 0:
                checkpoint_path = self._save_checkpoint()
                if self.on_checkpoint_saved:
                    self.on_checkpoint_saved(checkpoint_path)

            # 8. Call iteration callback
            if self.on_iteration_complete:
                self.on_iteration_complete(metrics)

            # 9. Update VLA serving (hot-swap)
            self._update_vla_serving()

        logger.info("Training completed")
        return self.metrics_history

    async def train_async(self) -> List[TrainingMetrics]:
        """Async version of training loop."""
        return await asyncio.to_thread(self.train)

    def _collect_trajectories(self) -> List[RobotTrajectory]:
        """
        Collect trajectories from physical robot.

        Uses the current policy to collect trajectories.
        """
        trajectories = []

        for i in range(self.config.trajectories_per_iteration):
            logger.debug(f"Collecting trajectory {i + 1}/{self.config.trajectories_per_iteration}")

            # Create policy function that queries current VLA
            def policy_fn(observation) -> Any:
                return self._query_policy(observation)

            # Collect trajectory
            trajectory = self.data_collector.collect_trajectory(
                task_description=self._sample_task(),
                policy_fn=policy_fn,
                max_steps=self.rl_config.data.max_trajectory_length,
                metadata={
                    "iteration": self.iteration,
                    "trajectory_idx": i,
                },
            )

            trajectories.append(trajectory)

        return trajectories

    def _compute_rewards(self, trajectories: List[RobotTrajectory]) -> None:
        """Compute rewards for collected trajectories."""
        for trajectory in trajectories:
            reward_output = self.reward_model.compute_reward(
                trajectory,
                task_description=trajectory.task_description,
            )

            # Store reward in trajectory metadata
            trajectory.metadata["computed_reward"] = reward_output.reward
            trajectory.metadata["reward_metadata"] = reward_output.metadata

            # Update step rewards if available
            if reward_output.step_rewards and len(reward_output.step_rewards) == len(trajectory.steps):
                for step, step_reward in zip(trajectory.steps, reward_output.step_rewards):
                    step.reward = step_reward

    def _train_iteration(self, batch: RLBatch) -> TrainingMetrics:
        """
        Train for one iteration using the appropriate method.

        Args:
            batch: Batch of training samples

        Returns:
            Training metrics for this iteration
        """
        mode = self.rl_config.training_mode

        if mode == TrainingMode.GRPO:
            return self._train_grpo(batch)
        elif mode == TrainingMode.OPD:
            return self._train_opd(batch)
        elif mode == TrainingMode.COMBINE:
            return self._train_combine(batch)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def _train_grpo(self, batch: RLBatch) -> TrainingMetrics:
        """
        Train using GRPO (Group Relative Policy Optimization).

        Binary RL method that uses relative rewards within groups.
        """
        logger.info("Training with GRPO")

        # Group samples
        groups = self._group_samples(batch.samples)

        # Compute relative rewards for each group
        all_rewards = []
        for group in groups:
            rewards = [s.reward for s in group if s.reward is not None]
            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                for sample in group:
                    if sample.reward is not None:
                        sample.metadata["relative_reward"] = sample.reward - mean_reward
                all_rewards.extend(rewards)

        # TODO: Integrate with actual OpenClaw-RL GRPO training
        # For now, return placeholder metrics
        loss = self._compute_grpo_loss(batch)

        return TrainingMetrics(
            epoch=self.iteration,
            loss=loss,
            reward_mean=sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            reward_std=self._compute_std(all_rewards) if all_rewards else 0.0,
            num_trajectories=len(groups),
            num_samples=len(batch),
            extra={"method": "grpo"},
        )

    def _train_opd(self, batch: RLBatch) -> TrainingMetrics:
        """
        Train using OPD (On-Policy Distillation).

        Distills from teacher policy using on-policy samples.
        """
        logger.info("Training with OPD")

        # Compute distillation targets
        for sample in batch.samples:
            # Use high-reward samples as positive examples
            if sample.reward and sample.reward > 0:
                sample.metadata["distillation_weight"] = self.rl_config.training.opd_distillation_weight

        # TODO: Integrate with actual OpenClaw-RL OPD training
        loss = self._compute_opd_loss(batch)

        rewards = [s.reward for s in batch.samples if s.reward is not None]

        return TrainingMetrics(
            epoch=self.iteration,
            loss=loss,
            reward_mean=sum(rewards) / len(rewards) if rewards else 0.0,
            reward_std=self._compute_std(rewards) if rewards else 0.0,
            num_samples=len(batch),
            extra={"method": "opd"},
        )

    def _train_combine(self, batch: RLBatch) -> TrainingMetrics:
        """
        Train using Combine method (GRPO + OPD).

        Combines advantages of both methods.
        """
        logger.info("Training with Combine")

        # Split batch for different methods
        # (Could also use same batch with combined loss)
        grpo_weight = self.rl_config.training.combine_grpo_weight
        opd_weight = self.rl_config.training.combine_opd_weight

        # Compute GRPO component
        groups = self._group_samples(batch.samples)
        for group in groups:
            rewards = [s.reward for s in group if s.reward is not None]
            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                for sample in group:
                    if sample.reward is not None:
                        sample.metadata["grpo_advantage"] = sample.reward - mean_reward

        # Compute OPD component
        for sample in batch.samples:
            if sample.reward and sample.reward > self.rl_config.training.opd_topk_threshold:
                sample.metadata["opd_positive"] = True

        # TODO: Integrate with actual OpenClaw-RL Combine training
        loss = grpo_weight * self._compute_grpo_loss(batch) + opd_weight * self._compute_opd_loss(batch)

        rewards = [s.reward for s in batch.samples if s.reward is not None]

        return TrainingMetrics(
            epoch=self.iteration,
            loss=loss,
            reward_mean=sum(rewards) / len(rewards) if rewards else 0.0,
            reward_std=self._compute_std(rewards) if rewards else 0.0,
            num_samples=len(batch),
            extra={
                "method": "combine",
                "grpo_weight": grpo_weight,
                "opd_weight": opd_weight,
            },
        )

    def _group_samples(self, samples: List[RLTrainingSample]) -> List[List[RLTrainingSample]]:
        """Group samples for GRPO training."""
        group_size = self.rl_config.training.grpo_group_size
        groups = []

        for i in range(0, len(samples), group_size):
            group = samples[i:i + group_size]
            groups.append(group)

        return groups

    def _compute_grpo_loss(self, batch: RLBatch) -> float:
        """Compute GRPO loss (placeholder)."""
        # TODO: Integrate with actual OpenClaw-RL loss computation
        return 0.5

    def _compute_opd_loss(self, batch: RLBatch) -> float:
        """Compute OPD loss (placeholder)."""
        # TODO: Integrate with actual OpenClaw-RL loss computation
        return 0.3

    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _query_policy(self, observation: Any) -> Any:
        """
        Query current policy (VLA model) for action.

        Args:
            observation: Current observation

        Returns:
            Action to take
        """
        # TODO: Query VLA serving API
        # This would call the VLA model to get the next action

        # Placeholder: return empty action
        return {"command": "placeholder_action"}

    def _sample_task(self) -> str:
        """Sample a task description for trajectory collection."""
        # TODO: Sample from task distribution
        # This would sample from the set of tasks the robot should learn
        return "Pick up the red block and place it on the blue platform"

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        logger.info("Evaluating policy...")

        # TODO: Run evaluation episodes
        eval_metrics = {
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 0.0,
        }

        return eval_metrics

    def _save_checkpoint(self) -> Path:
        """Save training checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_{self.iteration:04d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO: Save actual model checkpoint
        checkpoint = {
            "iteration": self.iteration,
            "metrics": [m.to_dict() for m in self.metrics_history],
            "config": self.rl_config.to_dict(),
        }

        # torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def _update_vla_serving(self) -> None:
        """Update VLA serving with new policy weights (hot-swap)."""
        logger.info("Updating VLA serving...")

        # TODO: Implement hot-swap to VLA serving layer
        # This would:
        # 1. Export current policy to format expected by serving
        # 2. Signal serving layer to load new weights
        # 3. Verify new weights are active

    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics."""
        logger.info(
            f"Iter {metrics.epoch}: loss={metrics.loss:.4f}, "
            f"reward_mean={metrics.reward_mean:.4f}, "
            f"samples={metrics.num_samples}"
        )

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        # TODO: Load actual checkpoint
        # checkpoint = torch.load(checkpoint_path)
        # self.iteration = checkpoint["iteration"]
        # self.metrics_history = checkpoint["metrics"]

    def export_policy(self, export_path: Path) -> None:
        """Export trained policy for serving."""
        logger.info(f"Exporting policy to: {export_path}")

        # TODO: Export policy in serving format
        export_path.parent.mkdir(parents=True, exist_ok=True)


class TrainingPipeline:
    """
    High-level training pipeline for ROSClaw RL.

    Provides a simple interface for running RL training.
    """

    def __init__(self, config: ROSClawRLConfig):
        self.config = config
        self.trainer: Optional[RLTrainer] = None

    def setup(
        self,
        reward_model: Optional[BaseRewardModel] = None,
        data_collector: Optional[PhysicalDataCollector] = None,
        trajectory_converter: Optional[TrajectoryConverter] = None,
    ) -> "TrainingPipeline":
        """Setup the training pipeline."""
        training_config = TrainingConfig(
            config=self.config,
            checkpoint_dir=self.config.system.checkpoint_dir,
            log_dir=self.config.system.log_dir,
            num_iterations=100,
            trajectories_per_iteration=self.config.data.collection_batch_size,
        )

        self.trainer = RLTrainer(
            config=training_config,
            reward_model=reward_model,
            data_collector=data_collector,
            trajectory_converter=trajectory_converter,
        )

        return self

    def run(self) -> List[TrainingMetrics]:
        """Run the training pipeline."""
        if not self.trainer:
            raise RuntimeError("Training pipeline not set up. Call setup() first.")

        return self.trainer.train()

    async def run_async(self) -> List[TrainingMetrics]:
        """Run the training pipeline asynchronously."""
        if not self.trainer:
            raise RuntimeError("Training pipeline not set up. Call setup() first.")

        return await self.trainer.train_async()
