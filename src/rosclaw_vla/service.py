"""VLA Service - Main inference service for Vision-Language-Action models."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal

import numpy as np
import torch
from PIL import Image

from .action_parser import ActionParser, ActionSequence
from .openvla_adapter import OpenVLAAdapter
from .policies.base import BasePolicy, PolicyOutput
from .policies.openvla import OpenVLAPolicy

logger = logging.getLogger(__name__)


@dataclass
class VLAConfig:
    """Configuration for VLA Service."""

    model_name: str = "openvla/openvla-7b"
    """HuggingFace model identifier."""

    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    """Device to run inference on."""

    dtype: Literal["fp32", "fp16", "bf16", "int8", "int4"] = "bf16"
    """Model precision."""

    max_new_tokens: int = 512
    """Maximum tokens to generate."""

    temperature: float = 0.7
    """Sampling temperature."""

    top_p: float = 0.9
    """Nucleus sampling parameter."""

    do_sample: bool = True
    """Whether to use sampling."""

    action_horizon: int = 16
    """Number of actions to predict."""

    action_dim: int = 7
    """Dimensionality of action space (x, y, z, roll, pitch, yaw, gripper)."""

    warmup_steps: int = 1
    """Number of warmup inference calls."""

    streaming: bool = True
    """Enable streaming mode for real-time control."""

    cache_dir: Path | None = None
    """Directory to cache downloaded models."""

    custom_policy: BasePolicy | None = None
    """Custom policy implementation (optional)."""

    # Safety limits
    max_velocity: float = 0.5
    """Maximum velocity command (m/s)."""

    max_angular_velocity: float = 1.0
    """Maximum angular velocity (rad/s)."""

    workspace_bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.8),
    })
    """Workspace boundaries for safety."""


@dataclass
class VLAOutput:
    """Output from VLA inference."""

    actions: np.ndarray
    """Predicted action sequence [horizon, action_dim]."""

    raw_text: str
    """Raw generated text from model."""

    confidence: float
    """Confidence score (0-1)."""

    latency_ms: float
    """Inference latency in milliseconds."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""


class VLAService:
    """Vision-Language-Action inference service.

    Provides a unified interface for VLA models including:
    - OpenVLA (7B parameter model)
    - Custom policy implementations
    - Streaming inference for real-time control
    - Safety validation and action parsing
    """

    def __init__(self, config: VLAConfig | None = None):
        """Initialize VLA service.

        Args:
            config: VLA configuration. Uses defaults if not provided.
        """
        self.config = config or VLAConfig()
        self._policy: BasePolicy | None = None
        self._action_parser = ActionParser(self.config.action_dim)
        self._initialized = False
        self._warmup_complete = False
        self._inference_count = 0
        self._callback: Callable[[VLAOutput], None] | None = None

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the VLA service and load model."""
        if self._initialized:
            logger.warning("VLA service already initialized")
            return

        logger.info(f"Initializing VLA service with model: {self.config.model_name}")

        try:
            # Initialize policy
            if self.config.custom_policy:
                self._policy = self.config.custom_policy
            elif "openvla" in self.config.model_name.lower():
                self._policy = OpenVLAPolicy(self.config)
            else:
                raise ValueError(f"Unsupported model: {self.config.model_name}")

            # Load model weights
            await self._policy.load()

            self._initialized = True
            logger.info("VLA service initialized successfully")

            # Run warmup
            if self.config.warmup_steps > 0:
                await self._warmup()

        except Exception as e:
            logger.error(f"Failed to initialize VLA service: {e}")
            raise RuntimeError(f"VLA initialization failed: {e}") from e

    async def _warmup(self) -> None:
        """Run warmup inference calls."""
        logger.info(f"Running {self.config.warmup_steps} warmup steps...")

        # Create dummy inputs
        dummy_image = Image.new("RGB", (224, 224), color="gray")
        dummy_instruction = "warmup"

        for i in range(self.config.warmup_steps):
            try:
                _ = await self.predict(dummy_image, dummy_instruction)
                logger.debug(f"Warmup step {i + 1}/{self.config.warmup_steps} complete")
            except Exception as e:
                logger.warning(f"Warmup step {i + 1} failed: {e}")

        self._warmup_complete = True
        logger.info("Warmup complete")

    async def predict(
        self,
        image: Image.Image,
        instruction: str,
        proprioception: np.ndarray | None = None,
    ) -> VLAOutput:
        """Generate actions from image and instruction.

        Args:
            image: Current camera image.
            instruction: Natural language instruction.
            proprioception: Optional robot state (joint positions, etc.).

        Returns:
            VLAOutput containing predicted actions and metadata.
        """
        if not self._initialized:
            raise RuntimeError("VLA service not initialized. Call initialize() first.")

        if not self._policy:
            raise RuntimeError("No policy loaded")

        import time
        start_time = time.perf_counter()

        try:
            # Run inference through policy
            policy_output: PolicyOutput = await self._policy.predict(
                image=image,
                instruction=instruction,
                proprioception=proprioception,
            )

            # Parse and validate actions
            actions = self._action_parser.parse(
                policy_output.raw_output,
                self.config.action_horizon,
            )

            # Apply safety constraints
            actions = self._apply_safety_constraints(actions)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Create output
            output = VLAOutput(
                actions=actions,
                raw_text=policy_output.raw_output,
                confidence=policy_output.confidence,
                latency_ms=latency_ms,
                metadata={
                    "model_name": self.config.model_name,
                    "inference_count": self._inference_count,
                    **policy_output.metadata,
                },
            )

            self._inference_count += 1

            # Trigger callback if set
            if self._callback:
                self._callback(output)

            return output

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"VLA prediction failed: {e}") from e

    async def predict_stream(
        self,
        image: Image.Image,
        instruction: str,
        proprioception: np.ndarray | None = None,
    ) -> AsyncIterator[VLAOutput]:
        """Stream actions as they are generated.

        Yields partial action sequences as they become available,
        enabling real-time robot control.

        Args:
            image: Current camera image.
            instruction: Natural language instruction.
            proprioception: Optional robot state.

        Yields:
            VLAOutput with incremental action updates.
        """
        if not self._initialized:
            raise RuntimeError("VLA service not initialized")

        if not self._policy:
            raise RuntimeError("No policy loaded")

        import time
        start_time = time.perf_counter()

        try:
            async for partial_output in self._policy.predict_stream(
                image=image,
                instruction=instruction,
                proprioception=proprioception,
            ):
                # Parse actions
                actions = self._action_parser.parse(
                    partial_output.raw_output,
                    self.config.action_horizon,
                )

                # Apply safety constraints
                actions = self._apply_safety_constraints(actions)

                latency_ms = (time.perf_counter() - start_time) * 1000

                yield VLAOutput(
                    actions=actions,
                    raw_text=partial_output.raw_output,
                    confidence=partial_output.confidence,
                    latency_ms=latency_ms,
                    metadata={
                        "model_name": self.config.model_name,
                        "inference_count": self._inference_count,
                        "partial": True,
                        **partial_output.metadata,
                    },
                )

            self._inference_count += 1

        except Exception as e:
            logger.error(f"Streaming prediction failed: {e}")
            raise RuntimeError(f"VLA streaming prediction failed: {e}") from e

    def _apply_safety_constraints(self, actions: np.ndarray) -> np.ndarray:
        """Apply safety constraints to actions.

        Args:
            actions: Raw predicted actions.

        Returns:
            Safety-constrained actions.
        """
        # Clone to avoid modifying original
        safe_actions = actions.copy()

        # Clip velocities
        if safe_actions.shape[1] >= 6:  # Has velocity components
            safe_actions[:, 0] = np.clip(
                safe_actions[:, 0],
                -self.config.max_velocity,
                self.config.max_velocity,
            )  # x velocity
            safe_actions[:, 1] = np.clip(
                safe_actions[:, 1],
                -self.config.max_velocity,
                self.config.max_velocity,
            )  # y velocity
            safe_actions[:, 2] = np.clip(
                safe_actions[:, 2],
                -self.config.max_velocity,
                self.config.max_velocity,
            )  # z velocity

        # Clip angular velocities
            safe_actions[:, 3] = np.clip(
                safe_actions[:, 3],
                -self.config.max_angular_velocity,
                self.config.max_angular_velocity,
            )  # roll
            safe_actions[:, 4] = np.clip(
                safe_actions[:, 4],
                -self.config.max_angular_velocity,
                self.config.max_angular_velocity,
            )  # pitch
            safe_actions[:, 5] = np.clip(
                safe_actions[:, 5],
                -self.config.max_angular_velocity,
                self.config.max_angular_velocity,
            )  # yaw

        # Clip gripper to [0, 1]
        if safe_actions.shape[1] >= 7:
            safe_actions[:, 6] = np.clip(safe_actions[:, 6], 0.0, 1.0)

        # Apply workspace bounds if position commands
        # (Assuming first 3 dims are position in some modes)

        return safe_actions

    def set_callback(self, callback: Callable[[VLAOutput], None] | None) -> None:
        """Set callback for inference results.

        Args:
            callback: Function called after each inference.
        """
        self._callback = callback

    async def shutdown(self) -> None:
        """Shutdown the VLA service and free resources."""
        logger.info("Shutting down VLA service...")

        if self._policy:
            await self._policy.unload()
            self._policy = None

        self._initialized = False
        self._warmup_complete = False

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("VLA service shutdown complete")

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics.

        Returns:
            Dictionary of service statistics.
        """
        return {
            "initialized": self._initialized,
            "warmup_complete": self._warmup_complete,
            "inference_count": self._inference_count,
            "model_name": self.config.model_name,
            "device": str(self._policy.device) if self._policy else "none",
        }
