"""
Base reward model interface.

Defines the interface for computing rewards from robot trajectories.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RewardOutput:
    """Output from reward model computation."""
    reward: float  # Scalar reward value
    metadata: Dict[str, Any]  # Additional information
    step_rewards: Optional[List[float]] = None  # Per-step rewards if available


class BaseRewardModel(ABC):
    """
    Base class for reward models.

    Reward models compute rewards from robot trajectories or states.
    This is a simplified M-PRM (Multi-modal Process Reward Model) interface.
    """

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def compute_reward(
        self,
        trajectory_or_state: Any,
        task_description: Optional[str] = None,
        **kwargs
    ) -> RewardOutput:
        """
        Compute reward for a trajectory or state.

        Args:
            trajectory_or_state: Trajectory or state to evaluate
            task_description: Optional task description for context
            **kwargs: Additional arguments

        Returns:
            RewardOutput with reward value and metadata
        """
        pass

    @abstractmethod
    def compute_step_reward(
        self,
        observation: Any,
        action: Any,
        next_observation: Any,
        task_description: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Compute reward for a single step.

        Args:
            observation: Current observation
            action: Action taken
            next_observation: Next observation
            task_description: Optional task description
            **kwargs: Additional arguments

        Returns:
            Scalar reward value
        """
        pass

    def batch_compute_rewards(
        self,
        trajectories: List[Any],
        task_description: Optional[str] = None,
        **kwargs
    ) -> List[RewardOutput]:
        """
        Compute rewards for multiple trajectories.

        Default implementation calls compute_reward for each trajectory.
        Override for batch-optimized implementations.
        """
        return [
            self.compute_reward(traj, task_description, **kwargs)
            for traj in trajectories
        ]
