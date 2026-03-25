"""Base policy interface for VLA models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PolicyOutput:
    """Output from policy inference."""

    raw_output: str | np.ndarray
    """Raw model output (text or array)."""

    confidence: float = 1.0
    """Confidence score (0-1)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""


class BasePolicy(ABC):
    """Abstract base class for VLA policies.

    All policy implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: Any):
        """Initialize policy with configuration.

        Args:
            config: Policy configuration object.
        """
        self.config = config
        self._loaded = False

    @abstractmethod
    async def load(self) -> None:
        """Load model weights and initialize."""
        raise NotImplementedError

    @abstractmethod
    async def unload(self) -> None:
        """Unload model and free resources."""
        raise NotImplementedError

    @abstractmethod
    async def predict(
        self,
        image: Image.Image,
        instruction: str,
        proprioception: np.ndarray | None = None,
    ) -> PolicyOutput:
        """Generate actions from observation.

        Args:
            image: Current camera observation.
            instruction: Natural language instruction.
            proprioception: Optional robot proprioceptive state.

        Returns:
            Policy output containing actions and metadata.
        """
        raise NotImplementedError

    async def predict_stream(
        self,
        image: Image.Image,
        instruction: str,
        proprioception: np.ndarray | None = None,
    ) -> AsyncIterator[PolicyOutput]:
        """Stream actions as they are generated.

        Default implementation yields final result only.
        Subclasses can override for true streaming.

        Args:
            image: Current camera observation.
            instruction: Natural language instruction.
            proprioception: Optional robot proprioceptive state.

        Yields:
            Partial policy outputs.
        """
        # Default: no streaming, yield final result
        output = await self.predict(image, instruction, proprioception)
        yield output

    @property
    def is_loaded(self) -> bool:
        """Check if policy is loaded."""
        return self._loaded

    @property
    @abstractmethod
    def device(self) -> str:
        """Get device policy is running on."""
        raise NotImplementedError

    @property
    def supports_streaming(self) -> bool:
        """Check if policy supports streaming inference."""
        return False

    def validate_input(
        self,
        image: Image.Image,
        instruction: str,
    ) -> None:
        """Validate input before inference.

        Args:
            image: Input image.
            instruction: Input instruction.

        Raises:
            ValueError: If inputs are invalid.
        """
        if image is None:
            raise ValueError("Image cannot be None")

        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")

        if not instruction or not instruction.strip():
            raise ValueError("Instruction cannot be empty")

        # Check image size
        if image.size[0] < 10 or image.size[1] < 10:
            raise ValueError(f"Image too small: {image.size}")

        if image.size[0] > 4096 or image.size[1] > 4096:
            raise ValueError(f"Image too large: {image.size}")

    def preprocess_image(
        self,
        image: Image.Image,
        target_size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """Preprocess image for model input.

        Args:
            image: Input image.
            target_size: Optional (width, height) to resize to.

        Returns:
            Preprocessed image.
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if target size specified
        if target_size is not None:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        return image

    def postprocess_actions(
        self,
        actions: np.ndarray,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Postprocess and clip actions to valid range.

        Args:
            actions: Raw predicted actions.
            bounds: Optional action bounds.

        Returns:
            Postprocessed actions.
        """
        if bounds is not None:
            for dim, (min_val, max_val) in bounds.items():
                dim_idx = self._get_dim_index(dim)
                if dim_idx is not None and dim_idx < actions.shape[-1]:
                    actions[..., dim_idx] = np.clip(
                        actions[..., dim_idx], min_val, max_val
                    )

        return actions

    def _get_dim_index(self, dim_name: str) -> int | None:
        """Get action dimension index from name.

        Args:
            dim_name: Dimension name (x, y, z, roll, pitch, yaw, gripper).

        Returns:
            Index or None if not found.
        """
        mapping = {
            "x": 0, "y": 1, "z": 2,
            "roll": 3, "pitch": 4, "yaw": 5,
            "gripper": 6,
        }
        return mapping.get(dim_name.lower())
