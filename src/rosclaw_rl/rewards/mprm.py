"""
M-PRM (Multi-modal Process Reward Model) implementation.

Simplified version for Phase 1:
- Scalar rewards from task success/failure
- Optional VLM-based feedback
- Extensible to multi-modal in Phase 2
"""

import base64
import json
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from ..collector import Observation, RobotTrajectory
from .base import BaseRewardModel, RewardOutput

logger = logging.getLogger(__name__)


class MPRMRewardModel(BaseRewardModel):
    """
    Multi-modal Process Reward Model for robot tasks.

    Phase 1: Simple binary rewards + optional VLM feedback
    Phase 2: Full multi-modal PRM with proprioception, vision, force, audio
    """

    def __init__(
        self,
        name: str = "m_prm",
        mode: str = "scalar",  # scalar, vlm, multimodal
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
        step_penalty: float = -0.01,
        vlm_model: Optional[str] = None,
        vlm_api_url: Optional[str] = None,
        vlm_temperature: float = 0.2,
        vlm_max_tokens: int = 256,
        use_proprioception: bool = True,
        use_vision: bool = True,
        use_force: bool = False,
        use_audio: bool = False,
    ):
        super().__init__(name)
        self.mode = mode
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.step_penalty = step_penalty

        # VLM settings
        self.vlm_model = vlm_model
        self.vlm_api_url = vlm_api_url
        self.vlm_temperature = vlm_temperature
        self.vlm_max_tokens = vlm_max_tokens

        # Multi-modal settings
        self.use_proprioception = use_proprioception
        self.use_vision = use_vision
        self.use_force = use_force
        self.use_audio = use_audio

    def compute_reward(
        self,
        trajectory_or_state: Any,
        task_description: Optional[str] = None,
        **kwargs
    ) -> RewardOutput:
        """
        Compute reward for a trajectory.

        Args:
            trajectory_or_state: RobotTrajectory or observation dict
            task_description: Task description for context
            **kwargs: Additional arguments

        Returns:
            RewardOutput with reward and metadata
        """
        if isinstance(trajectory_or_state, RobotTrajectory):
            return self._compute_trajectory_reward(
                trajectory_or_state,
                task_description,
                **kwargs
            )
        else:
            # Treat as observation/state dict
            return self._compute_state_reward(
                trajectory_or_state,
                task_description,
                **kwargs
            )

    def _compute_trajectory_reward(
        self,
        trajectory: RobotTrajectory,
        task_description: Optional[str],
        **kwargs
    ) -> RewardOutput:
        """Compute reward for a complete trajectory."""
        metadata = {
            "mode": self.mode,
            "trajectory_id": trajectory.trajectory_id,
            "num_steps": len(trajectory.steps),
        }

        if self.mode == "scalar":
            # Simple binary reward based on success
            reward = self.success_reward if trajectory.success else self.failure_reward
            metadata["reward_type"] = "binary_success"

            # Add step penalties for efficiency
            step_penalty_total = len(trajectory.steps) * self.step_penalty
            reward += step_penalty_total
            metadata["step_penalty"] = step_penalty_total

            return RewardOutput(reward=reward, metadata=metadata)

        elif self.mode == "vlm":
            # Use VLM for reward feedback
            return self._compute_vlm_trajectory_reward(
                trajectory,
                task_description,
                metadata,
                **kwargs
            )

        elif self.mode == "multimodal":
            # Full multi-modal reward (Phase 2)
            return self._compute_multimodal_trajectory_reward(
                trajectory,
                task_description,
                metadata,
                **kwargs
            )

        else:
            logger.warning(f"Unknown mode: {self.mode}, using scalar")
            reward = self.success_reward if trajectory.success else self.failure_reward
            return RewardOutput(reward=reward, metadata=metadata)

    def _compute_state_reward(
        self,
        state: Dict[str, Any],
        task_description: Optional[str],
        **kwargs
    ) -> RewardOutput:
        """Compute reward for a single state/observation."""
        metadata = {
            "mode": self.mode,
            "state_keys": list(state.keys()),
        }

        # For state rewards, we'd need more context
        # Default to neutral/penalty reward
        reward = self.step_penalty

        return RewardOutput(reward=reward, metadata=metadata)

    def compute_step_reward(
        self,
        observation: Observation,
        action: Any,
        next_observation: Observation,
        task_description: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Compute reward for a single step.

        Args:
            observation: Current observation
            action: Action taken
            next_observation: Next observation after action
            task_description: Task description
            **kwargs: Additional arguments

        Returns:
            Scalar reward value
        """
        if self.mode == "scalar":
            return self.step_penalty

        elif self.mode == "vlm" and self.vlm_api_url:
            # Use VLM to evaluate step progress
            return self._compute_vlm_step_reward(
                observation,
                action,
                next_observation,
                task_description,
                **kwargs
            )

        # Default to step penalty
        return self.step_penalty

    def _compute_vlm_trajectory_reward(
        self,
        trajectory: RobotTrajectory,
        task_description: Optional[str],
        metadata: Dict,
        **kwargs
    ) -> RewardOutput:
        """Compute trajectory reward using VLM feedback."""
        if not self.vlm_api_url:
            logger.warning("VLM API URL not set, falling back to scalar")
            reward = self.success_reward if trajectory.success else self.failure_reward
            metadata["fallback"] = "scalar"
            return RewardOutput(reward=reward, metadata=metadata)

        try:
            # Build VLM prompt
            prompt = self._build_vlm_trajectory_prompt(
                trajectory,
                task_description
            )

            # Get VLM response
            response = self._call_vlm(prompt, trajectory)

            # Parse reward from response
            reward, feedback = self._parse_vlm_reward_response(response)

            metadata["vlm_feedback"] = feedback
            metadata["vlm_model"] = self.vlm_model

            return RewardOutput(reward=reward, metadata=metadata)

        except Exception as e:
            logger.error(f"VLM reward computation failed: {e}")
            # Fall back to scalar
            reward = self.success_reward if trajectory.success else self.failure_reward
            metadata["vlm_error"] = str(e)
            metadata["fallback"] = "scalar"
            return RewardOutput(reward=reward, metadata=metadata)

    def _compute_vlm_step_reward(
        self,
        observation: Observation,
        action: Any,
        next_observation: Observation,
        task_description: Optional[str],
        **kwargs
    ) -> float:
        """Compute step reward using VLM feedback."""
        if not self.vlm_api_url:
            return self.step_penalty

        try:
            # Build step evaluation prompt
            prompt = self._build_vlm_step_prompt(
                observation,
                action,
                next_observation,
                task_description
            )

            # Get VLM response
            response = self._call_vlm(prompt, None, images=[observation.image])

            # Parse progress score
            score = self._parse_vlm_step_response(response)

            return score

        except Exception as e:
            logger.error(f"VLM step reward computation failed: {e}")
            return self.step_penalty

    def _compute_multimodal_trajectory_reward(
        self,
        trajectory: RobotTrajectory,
        task_description: Optional[str],
        metadata: Dict,
        **kwargs
    ) -> RewardOutput:
        """
        Compute multi-modal trajectory reward (Phase 2).

        Combines vision, proprioception, force, and audio signals.
        """
        # This is a placeholder for Phase 2 implementation
        # Would integrate with a trained multi-modal PRM

        rewards = {}

        # Vision-based reward
        if self.use_vision:
            rewards["vision"] = self._compute_vision_reward(trajectory)

        # Proprioception-based reward
        if self.use_proprioception:
            rewards["proprioception"] = self._compute_proprioception_reward(trajectory)

        # Force-based reward
        if self.use_force:
            rewards["force"] = self._compute_force_reward(trajectory)

        # Audio-based reward
        if self.use_audio:
            rewards["audio"] = self._compute_audio_reward(trajectory)

        # Combine rewards (simple average for now)
        if rewards:
            combined_reward = np.mean(list(rewards.values()))
        else:
            combined_reward = self.success_reward if trajectory.success else self.failure_reward

        metadata["modalities"] = list(rewards.keys())
        metadata["component_rewards"] = rewards

        return RewardOutput(
            reward=combined_reward,
            metadata=metadata,
        )

    def _build_vlm_trajectory_prompt(
        self,
        trajectory: RobotTrajectory,
        task_description: Optional[str],
    ) -> str:
        """Build prompt for VLM trajectory evaluation."""
        task = task_description or trajectory.task_description

        prompt = (
            f"You are evaluating a robot's execution of the task: \"{task}\"\n\n"
            f"The robot executed {len(trajectory.steps)} steps.\n"
        )

        if trajectory.success is not None:
            prompt += f"Final outcome: {'Success' if trajectory.success else 'Failure'}\n\n"

        prompt += (
            "Rate the robot's performance on a scale from -1 to 1:\n"
            "- 1: Perfect execution, task completed optimally\n"
            "- 0: Neutral, partial completion or suboptimal\n"
            "- -1: Failed completely or unsafe behavior\n\n"
            "Respond with only a number between -1 and 1."
        )

        return prompt

    def _build_vlm_step_prompt(
        self,
        observation: Observation,
        action: Any,
        next_observation: Observation,
        task_description: Optional[str],
    ) -> str:
        """Build prompt for VLM step evaluation."""
        prompt = (
            f"Task: {task_description or 'Execute the given task'}\n\n"
            "Evaluate if this action made progress toward the goal.\n"
            "Rate progress on a scale from -0.1 to 0.1:\n"
            "- 0.1: Clear progress toward the goal\n"
            "- 0: Neutral or unclear progress\n"
            "- -0.1: Regress or incorrect action\n\n"
            "Respond with only a number."
        )

        return prompt

    def _call_vlm(
        self,
        prompt: str,
        trajectory: Optional[RobotTrajectory],
        images: Optional[List[np.ndarray]] = None,
    ) -> str:
        """Call VLM API."""
        # Prepare messages
        content = [{"type": "text", "text": prompt}]

        # Add images if available
        if images:
            for img in images:
                if img is not None:
                    img_b64 = self._encode_image(img)
                    if img_b64:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img_b64},
                        })

        messages = [{"role": "user", "content": content}]

        # Make API call
        response = requests.post(
            f"{self.vlm_api_url}/v1/chat/completions",
            json={
                "model": self.vlm_model,
                "messages": messages,
                "temperature": self.vlm_temperature,
                "max_tokens": self.vlm_max_tokens,
            },
            timeout=30,
        )

        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]

    def _parse_vlm_reward_response(self, response: str) -> Tuple[float, str]:
        """Parse reward value and feedback from VLM response."""
        # Try to extract number from response
        try:
            # Look for a number between -1 and 1
            import re
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                reward = float(numbers[0])
                # Clamp to valid range
                reward = max(-1.0, min(1.0, reward))
                return reward, response
        except Exception as e:
            logger.warning(f"Failed to parse VLM response: {e}")

        # Default to neutral if parsing fails
        return 0.0, response

    def _parse_vlm_step_response(self, response: str) -> float:
        """Parse step progress score from VLM response."""
        try:
            import re
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                # Clamp to valid range
                return max(-0.1, min(0.1, score))
        except Exception as e:
            logger.warning(f"Failed to parse VLM step response: {e}")

        return 0.0

    def _encode_image(self, image: np.ndarray) -> Optional[str]:
        """Encode numpy image to base64."""
        try:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    # Placeholder methods for Phase 2 multi-modal rewards

    def _compute_vision_reward(self, trajectory: RobotTrajectory) -> float:
        """Compute vision-based reward (Phase 2)."""
        # Placeholder: Would use trained visual PRM
        return self.success_reward if trajectory.success else self.failure_reward

    def _compute_proprioception_reward(self, trajectory: RobotTrajectory) -> float:
        """Compute proprioception-based reward (Phase 2)."""
        # Placeholder: Would use proprioceptive features
        return self.success_reward if trajectory.success else self.failure_reward

    def _compute_force_reward(self, trajectory: RobotTrajectory) -> float:
        """Compute force-based reward (Phase 2)."""
        # Placeholder: Would use force/torque features
        return 0.0  # Neutral if not using force

    def _compute_audio_reward(self, trajectory: RobotTrajectory) -> float:
        """Compute audio-based reward (Phase 2)."""
        # Placeholder: Would use audio features
        return 0.0  # Neutral if not using audio
