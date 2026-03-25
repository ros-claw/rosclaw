"""
Trajectory converter - converts robot trajectories to RL training format.

Adapts physical robot trajectories to OpenClaw-RL's expected format.
"""

import base64
import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from .collector import Observation, RobotTrajectory, TrajectoryStep

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingSample:
    """
    Single sample formatted for RL training.

    Matches OpenClaw-RL's expected format with conversations and metadata.
    """
    index: int
    conversation: List[Dict[str, Any]]  # Chat format messages
    images: List[str]  # Base64 encoded images
    reward: Optional[float] = None
    task: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "index": self.index,
            "conversation": self.conversation,
            "images": self.images,
            "reward": self.reward,
            "task": self.task,
            "metadata": self.metadata,
        }


@dataclass
class RLBatch:
    """Batch of samples for training."""
    samples: List[RLTrainingSample]
    batch_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)


class TrajectoryConverter:
    """
    Converts robot trajectories to RL training format.

    Adapts physical robot data to OpenClaw-RL's expected format.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        image_format: str = "JPEG",
        max_images_per_trajectory: int = 8,
    ):
        self.image_size = image_size
        self.image_format = image_format
        self.max_images_per_trajectory = max_images_per_trajectory

    def convert_trajectory(
        self,
        trajectory: RobotTrajectory,
        index: int = 0,
        include_intermediate_steps: bool = False,
    ) -> List[RLTrainingSample]:
        """
        Convert a robot trajectory to RL training samples.

        Args:
            trajectory: Robot trajectory to convert
            index: Starting index for samples
            include_intermediate_steps: Whether to create samples for each step

        Returns:
            List of RL training samples
        """
        samples = []

        if not trajectory.steps:
            logger.warning(f"Empty trajectory: {trajectory.trajectory_id}")
            return samples

        if include_intermediate_steps:
            # Create samples for intermediate steps (for PRM training)
            samples.extend(self._convert_with_intermediate_steps(trajectory, index))
        else:
            # Create single sample for full trajectory
            sample = self._convert_full_trajectory(trajectory, index)
            if sample:
                samples.append(sample)

        return samples

    def _convert_full_trajectory(
        self,
        trajectory: RobotTrajectory,
        index: int,
    ) -> Optional[RLTrainingSample]:
        """Convert full trajectory to a single training sample."""
        # Build conversation
        conversation = self._build_conversation(trajectory)

        # Collect images from trajectory
        images = self._collect_images(trajectory)

        # Determine reward
        reward = self._compute_trajectory_reward(trajectory)

        return RLTrainingSample(
            index=index,
            conversation=conversation,
            images=images,
            reward=reward,
            task=trajectory.task_description,
            metadata={
                "trajectory_id": trajectory.trajectory_id,
                "num_steps": len(trajectory.steps),
                "duration": trajectory.end_time - trajectory.start_time if trajectory.end_time else 0,
                "success": trajectory.success,
            },
        )

    def _convert_with_intermediate_steps(
        self,
        trajectory: RobotTrajectory,
        start_index: int,
    ) -> List[RLTrainingSample]:
        """Convert trajectory with samples for intermediate steps."""
        samples = []

        for i, step in enumerate(trajectory.steps):
            # Build conversation up to this step
            conversation = self._build_conversation_up_to_step(trajectory, i)

            # Get images up to this step
            images = self._collect_images_up_to_step(trajectory, i)

            # Use step reward if available, otherwise compute
            reward = step.reward
            if reward is None:
                reward = self._compute_step_reward(step, trajectory, i)

            sample = RLTrainingSample(
                index=start_index + i,
                conversation=conversation,
                images=images,
                reward=reward,
                task=trajectory.task_description,
                metadata={
                    "trajectory_id": trajectory.trajectory_id,
                    "step_id": step.step_id,
                    "done": step.done,
                    "success": trajectory.success,
                },
            )
            samples.append(sample)

        return samples

    def _build_conversation(self, trajectory: RobotTrajectory) -> List[Dict[str, Any]]:
        """Build conversation format from trajectory."""
        conversation = []

        # System prompt
        conversation.append({
            "role": "system",
            "content": (
                "You are a robot control assistant. Given a task and observations, "
                "generate the appropriate robot commands to accomplish the task."
            ),
        })

        # User prompt with task
        user_content = self._build_user_prompt(trajectory)
        conversation.append({
            "role": "user",
            "content": user_content,
        })

        # Assistant response with actions
        assistant_content = self._build_assistant_response(trajectory)
        conversation.append({
            "role": "assistant",
            "content": assistant_content,
        })

        return conversation

    def _build_conversation_up_to_step(
        self,
        trajectory: RobotTrajectory,
        step_idx: int,
    ) -> List[Dict[str, Any]]:
        """Build conversation up to a specific step."""
        conversation = []

        # System prompt
        conversation.append({
            "role": "system",
            "content": (
                "You are a robot control assistant. Given a task and observations, "
                "generate the appropriate robot commands to accomplish the task."
            ),
        })

        # User prompt with task
        user_content = self._build_user_prompt_up_to_step(trajectory, step_idx)
        conversation.append({
            "role": "user",
            "content": user_content,
        })

        # Assistant response with actions up to this step
        assistant_content = self._build_assistant_response_up_to_step(trajectory, step_idx)
        conversation.append({
            "role": "assistant",
            "content": assistant_content,
        })

        return conversation

    def _build_user_prompt(self, trajectory: RobotTrajectory) -> Union[str, List[Dict]]:
        """Build user prompt from trajectory observations."""
        # Sample images from trajectory
        sampled_steps = self._sample_steps_with_images(trajectory)

        if not sampled_steps:
            # No images, use text only
            return f"Task: {trajectory.task_description}\n\nExecute this task on the robot."

        # Build multimodal content
        content = [
            {"type": "text", "text": f"Task: {trajectory.task_description}\n\n"},
        ]

        for i, step in enumerate(sampled_steps):
            content.append({
                "type": "text",
                "text": f"[Observation {i+1}]\n",
            })
            content.append({
                "type": "image",
                "image": f"<image_{i}>",  # Placeholder for image
            })

        content.append({
            "type": "text",
            "text": "\nExecute this task on the robot.",
        })

        return content

    def _build_user_prompt_up_to_step(
        self,
        trajectory: RobotTrajectory,
        step_idx: int,
    ) -> Union[str, List[Dict]]:
        """Build user prompt up to a specific step."""
        steps = trajectory.steps[:step_idx + 1]

        # Get steps with images
        steps_with_images = [s for s in steps if s.observation.image is not None]

        if not steps_with_images:
            return f"Task: {trajectory.task_description}\n\nExecute this task on the robot."

        content = [
            {"type": "text", "text": f"Task: {trajectory.task_description}\n\n"},
        ]

        for i, step in enumerate(steps_with_images):
            content.append({
                "type": "text",
                "text": f"[Observation {i+1}]\n",
            })
            content.append({
                "type": "image",
                "image": f"<image_{i}>",
            })

        content.append({
            "type": "text",
            "text": "\nExecute this task on the robot.",
        })

        return content

    def _build_assistant_response(self, trajectory: RobotTrajectory) -> str:
        """Build assistant response from trajectory actions."""
        actions = []

        for step in trajectory.steps:
            if step.action.command:
                actions.append(step.action.command)
            elif step.action.target_joint_positions:
                actions.append(f"move_joints({step.action.target_joint_positions})")
            elif step.action.target_end_effector_pose:
                pose = step.action.target_end_effector_pose
                actions.append(f"move_to({pose})")

        if not actions:
            return "No actions recorded."

        return "\n".join(actions)

    def _build_assistant_response_up_to_step(
        self,
        trajectory: RobotTrajectory,
        step_idx: int,
    ) -> str:
        """Build assistant response up to a specific step."""
        steps = trajectory.steps[:step_idx + 1]
        actions = []

        for step in steps:
            if step.action.command:
                actions.append(step.action.command)
            elif step.action.target_joint_positions:
                actions.append(f"move_joints({step.action.target_joint_positions})")
            elif step.action.target_end_effector_pose:
                pose = step.action.target_end_effector_pose
                actions.append(f"move_to({pose})")

        if not actions:
            return "No actions recorded."

        return "\n".join(actions)

    def _collect_images(self, trajectory: RobotTrajectory) -> List[str]:
        """Collect and encode images from trajectory."""
        sampled_steps = self._sample_steps_with_images(trajectory)

        images = []
        for step in sampled_steps:
            if step.observation.image is not None:
                encoded = self._encode_image(step.observation.image)
                if encoded:
                    images.append(encoded)

        return images

    def _collect_images_up_to_step(
        self,
        trajectory: RobotTrajectory,
        step_idx: int,
    ) -> List[str]:
        """Collect images up to a specific step."""
        steps = trajectory.steps[:step_idx + 1]
        steps_with_images = [s for s in steps if s.observation.image is not None]

        images = []
        for step in steps_with_images[-self.max_images_per_trajectory:]:
            encoded = self._encode_image(step.observation.image)
            if encoded:
                images.append(encoded)

        return images

    def _sample_steps_with_images(self, trajectory: RobotTrajectory) -> List[TrajectoryStep]:
        """Sample steps that have images, evenly distributed."""
        steps_with_images = [s for s in trajectory.steps if s.observation.image is not None]

        if len(steps_with_images) <= self.max_images_per_trajectory:
            return steps_with_images

        # Evenly sample
        indices = np.linspace(0, len(steps_with_images) - 1, self.max_images_per_trajectory, dtype=int)
        return [steps_with_images[i] for i in indices]

    def _encode_image(self, image: np.ndarray) -> Optional[str]:
        """Encode numpy image to base64 string."""
        try:
            # Convert to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)

            # Resize if needed
            if pil_image.size != self.image_size:
                pil_image = pil_image.resize(self.image_size)

            # Encode to base64
            buffer = BytesIO()
            pil_image.save(buffer, format=self.image_format)
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/{self.image_format.lower()};base64,{img_str}"

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _compute_trajectory_reward(self, trajectory: RobotTrajectory) -> float:
        """Compute reward for a complete trajectory."""
        if trajectory.success is None:
            return 0.0

        # Binary reward based on success
        return 1.0 if trajectory.success else -1.0

    def _compute_step_reward(
        self,
        step: TrajectoryStep,
        trajectory: RobotTrajectory,
        step_idx: int,
    ) -> float:
        """Compute reward for a single step."""
        # Use step reward if available
        if step.reward is not None:
            return step.reward

        # For intermediate steps, use a small penalty to encourage efficiency
        # Final step gets the trajectory success signal
        if step.done and trajectory.success is not None:
            return 1.0 if trajectory.success else -1.0

        # Intermediate step penalty
        return -0.01

    def convert_batch(
        self,
        trajectories: List[RobotTrajectory],
        batch_id: str = "",
        include_intermediate_steps: bool = False,
    ) -> RLBatch:
        """Convert a batch of trajectories."""
        all_samples = []
        index = 0

        for trajectory in trajectories:
            samples = self.convert_trajectory(
                trajectory,
                index=index,
                include_intermediate_steps=include_intermediate_steps,
            )
            all_samples.extend(samples)
            index += len(samples)

        return RLBatch(
            samples=all_samples,
            batch_id=batch_id,
            metadata={
                "num_trajectories": len(trajectories),
                "num_samples": len(all_samples),
            },
        )
