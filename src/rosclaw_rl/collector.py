"""
Physical data collection from robots.

Adapts OpenClaw-RL's rollout concepts for physical robot data collection.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Single observation from robot."""
    timestamp: float
    image: Optional[np.ndarray] = None  # Camera image
    proprioception: Optional[Dict[str, float]] = None  # Joint states, pose
    force: Optional[Dict[str, float]] = None  # Force/torque sensors
    audio: Optional[np.ndarray] = None  # Audio data
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Action sent to robot."""
    timestamp: float
    command: str  # Text command or code
    target_joint_positions: Optional[List[float]] = None
    target_end_effector_pose: Optional[Dict[str, List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    step_id: int
    observation: Observation
    action: Action
    reward: Optional[float] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotTrajectory:
    """Complete robot trajectory."""
    trajectory_id: str
    task_description: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    success: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TrajectoryStep) -> None:
        """Add a step to the trajectory."""
        self.steps.append(step)

    def finalize(self, success: bool) -> None:
        """Finalize the trajectory."""
        self.end_time = time.time()
        self.success = success

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "task_description": self.task_description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "success": self.success,
            "metadata": self.metadata,
            "steps": [
                {
                    "step_id": s.step_id,
                    "timestamp": s.observation.timestamp,
                    "observation": {
                        "proprioception": s.observation.proprioception,
                        "force": s.observation.force,
                        "metadata": s.observation.metadata,
                    },
                    "action": {
                        "command": s.action.command,
                        "target_joint_positions": s.action.target_joint_positions,
                        "target_end_effector_pose": s.action.target_end_effector_pose,
                        "metadata": s.action.metadata,
                    },
                    "reward": s.reward,
                    "done": s.done,
                    "info": s.info,
                }
                for s in self.steps
            ],
        }


class TrajectoryCollector:
    """Collects trajectory data from robot runtime."""

    def __init__(
        self,
        roboclaw_api_url: str = "http://localhost:8001",
        collection_frequency_hz: float = 10.0,
        max_trajectory_length: int = 1000,
    ):
        self.roboclaw_api_url = roboclaw_api_url
        self.collection_frequency_hz = collection_frequency_hz
        self.max_trajectory_length = max_trajectory_length
        self.collection_period = 1.0 / collection_frequency_hz

        self._current_trajectory: Optional[RobotTrajectory] = None
        self._step_counter: int = 0
        self._running: bool = False
        self._collection_thread: Optional[Thread] = None
        self._lock = Lock()

        # Callbacks
        self._observation_callback: Optional[Callable[[], Observation]] = None
        self._action_callback: Optional[Callable[[], Action]] = None

    def register_observation_callback(self, callback: Callable[[], Observation]) -> None:
        """Register callback to get observations from robot."""
        self._observation_callback = callback

    def register_action_callback(self, callback: Callable[[], Action]) -> None:
        """Register callback to get actions sent to robot."""
        self._action_callback = callback

    def start_trajectory(self, task_description: str, metadata: Optional[Dict] = None) -> str:
        """Start collecting a new trajectory."""
        with self._lock:
            trajectory_id = str(uuid4())
            self._current_trajectory = RobotTrajectory(
                trajectory_id=trajectory_id,
                task_description=task_description,
                metadata=metadata or {},
            )
            self._step_counter = 0
            self._running = True

            # Start collection thread
            self._collection_thread = Thread(target=self._collection_loop, daemon=True)
            self._collection_thread.start()

            logger.info(f"Started trajectory collection: {trajectory_id}")
            return trajectory_id

    def stop_trajectory(self, success: bool) -> Optional[RobotTrajectory]:
        """Stop collecting current trajectory."""
        with self._lock:
            self._running = False

            if self._collection_thread and self._collection_thread.is_alive():
                self._collection_thread.join(timeout=2.0)

            if self._current_trajectory:
                self._current_trajectory.finalize(success)
                trajectory = self._current_trajectory
                self._current_trajectory = None
                logger.info(f"Finalized trajectory: {trajectory.trajectory_id}, success={success}")
                return trajectory

            return None

    def _collection_loop(self) -> None:
        """Main collection loop running in background thread."""
        while self._running:
            loop_start = time.time()

            try:
                self._collect_step()
            except Exception as e:
                logger.error(f"Error collecting step: {e}")

            # Maintain collection frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.collection_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _collect_step(self) -> None:
        """Collect a single step."""
        if not self._current_trajectory:
            return

        if self._step_counter >= self.max_trajectory_length:
            logger.warning(f"Max trajectory length reached: {self.max_trajectory_length}")
            return

        # Get observation
        observation = None
        if self._observation_callback:
            try:
                observation = self._observation_callback()
            except Exception as e:
                logger.error(f"Observation callback failed: {e}")
                observation = Observation(timestamp=time.time())
        else:
            observation = Observation(timestamp=time.time())

        # Get action
        action = None
        if self._action_callback:
            try:
                action = self._action_callback()
            except Exception as e:
                logger.error(f"Action callback failed: {e}")
                action = Action(timestamp=time.time(), command="")
        else:
            action = Action(timestamp=time.time(), command="")

        # Create step
        step = TrajectoryStep(
            step_id=self._step_counter,
            observation=observation,
            action=action,
        )

        with self._lock:
            if self._current_trajectory:
                self._current_trajectory.add_step(step)
                self._step_counter += 1

    def get_current_trajectory(self) -> Optional[RobotTrajectory]:
        """Get current in-progress trajectory."""
        with self._lock:
            return self._current_trajectory


class PhysicalDataCollector:
    """
    High-level data collector for physical robot RL.

    Adapts OpenClaw-RL's rollout worker pattern for physical robots.
    """

    def __init__(
        self,
        config: "ROSClawRLConfig",
        output_queue: Optional[Queue] = None,
    ):
        self.config = config
        self.output_queue = output_queue or Queue(maxsize=10000)

        self.trajectory_collector = TrajectoryCollector(
            roboclaw_api_url=config.roboclaw_api_url,
            collection_frequency_hz=config.data.collection_frequency_hz,
            max_trajectory_length=config.data.max_trajectory_length,
        )

        self._completed_trajectories: List[RobotTrajectory] = []
        self._lock = Lock()
        self._running = False

    def start(self) -> None:
        """Start the data collector."""
        self._running = True
        logger.info("PhysicalDataCollector started")

    def stop(self) -> None:
        """Stop the data collector."""
        self._running = False
        logger.info("PhysicalDataCollector stopped")

    def collect_trajectory(
        self,
        task_description: str,
        policy_fn: Callable[[Observation], Action],
        max_steps: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> RobotTrajectory:
        """
        Collect a single trajectory using the provided policy.

        Args:
            task_description: Description of the task
            policy_fn: Function that takes observation and returns action
            max_steps: Maximum steps (defaults to config)
            metadata: Additional metadata

        Returns:
            Collected trajectory
        """
        max_steps = max_steps or self.config.data.max_trajectory_length

        # Setup callbacks
        current_observation: Optional[Observation] = None
        current_action: Optional[Action] = None

        def observation_callback() -> Observation:
            return current_observation or Observation(timestamp=time.time())

        def action_callback() -> Action:
            return current_action or Action(timestamp=time.time(), command="")

        self.trajectory_collector.register_observation_callback(observation_callback)
        self.trajectory_collector.register_action_callback(action_callback)

        # Start trajectory
        trajectory_id = self.trajectory_collector.start_trajectory(
            task_description=task_description,
            metadata=metadata,
        )

        # Execute policy
        step_count = 0
        done = False
        success = False

        try:
            while step_count < max_steps and not done and self._running:
                # Get observation from robot
                current_observation = self._get_observation()

                # Get action from policy
                current_action = policy_fn(current_observation)

                # Send action to robot
                done, success = self._execute_action(current_action)

                step_count += 1

                # Small delay to prevent overwhelming the system
                time.sleep(0.01)

        except Exception as e:
            logger.error(f"Error during trajectory collection: {e}")
            success = False

        finally:
            # Stop and get trajectory
            trajectory = self.trajectory_collector.stop_trajectory(success=success)

            if trajectory:
                with self._lock:
                    self._completed_trajectories.append(trajectory)

                # Add to output queue
                try:
                    self.output_queue.put_nowait((trajectory_id, trajectory))
                except Exception:
                    logger.warning("Output queue full, dropping trajectory")

                return trajectory

            # Return empty trajectory if something went wrong
            return RobotTrajectory(
                trajectory_id=trajectory_id,
                task_description=task_description,
                success=False,
            )

    def _get_observation(self) -> Observation:
        """Get current observation from robot."""
        # TODO: Integrate with RoboClaw runtime API
        # This would call the robot's observation endpoint
        return Observation(
            timestamp=time.time(),
            proprioception={},
            metadata={"source": "robot"},
        )

    def _execute_action(self, action: Action) -> tuple[bool, bool]:
        """
        Execute action on robot.

        Returns:
            (done, success) tuple
        """
        # TODO: Integrate with RoboClaw runtime API
        # This would send the action to the robot and return status
        return False, True

    def get_completed_trajectories(self) -> List[RobotTrajectory]:
        """Get all completed trajectories."""
        with self._lock:
            trajectories = self._completed_trajectories.copy()
            self._completed_trajectories.clear()
            return trajectories

    def get_queue_size(self) -> int:
        """Get current output queue size."""
        return self.output_queue.qsize()
