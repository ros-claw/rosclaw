"""
ROSClaw RL Training Layer

This package integrates OpenClaw-RL with ROSClaw for physical robot RL training.
It provides data collection, trajectory conversion, training orchestration, and reward modeling.
"""

from .collector import PhysicalDataCollector, TrajectoryCollector
from .converter import TrajectoryConverter, RobotTrajectory
from .trainer import RLTrainer, TrainingConfig
from .rewards.base import BaseRewardModel, RewardOutput
from .rewards.mprm import MPRMRewardModel

__version__ = "0.1.0"

__all__ = [
    # Collector
    "PhysicalDataCollector",
    "TrajectoryCollector",
    # Converter
    "TrajectoryConverter",
    "RobotTrajectory",
    # Trainer
    "RLTrainer",
    "TrainingConfig",
    # Rewards
    "BaseRewardModel",
    "RewardOutput",
    "MPRMRewardModel",
]
