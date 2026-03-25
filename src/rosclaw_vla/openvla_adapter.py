"""OpenVLA Adapter - Integration with OpenVLA model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class OpenVLAAdapter:
    """Adapter for OpenVLA vision-language-action model.

    OpenVLA is a 7B parameter open-source VLA model trained on
    Open X-Embodiment dataset with over 1M robot trajectories.

    Reference: https://github.com/openvla/openvla
    """

    # OpenVLA action space (standardized)
    ACTION_DIM = 7  # x, y, z, roll, pitch, yaw, gripper
    ACTION_HORIZON = 16  # Predict 16 future actions

    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        device: str = "auto",
        dtype: str = "bf16",
        cache_dir: str | None = None,
    ):
        """Initialize OpenVLA adapter.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to run on (auto/cuda/cpu/mps).
            dtype: Model precision (fp32/fp16/bf16/int8/int4).
            cache_dir: Directory for model cache.
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.cache_dir = cache_dir

        self._model: Any = None
        self._processor: Any = None
        self._is_loaded = False

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        """Resolve dtype string to torch dtype."""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(dtype, torch.bfloat16)

    async def load(self) -> None:
        """Load model and processor."""
        if self._is_loaded:
            return

        logger.info(f"Loading OpenVLA model: {self.model_name}")

        try:
            # Import here to avoid dependency if not using OpenVLA
            from transformers import AutoModelForVision2Seq, AutoProcessor

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            # Load model with appropriate quantization
            load_kwargs = {
                "cache_dir": self.cache_dir,
                "torch_dtype": self.dtype,
                "trust_remote_code": True,
            }

            # Handle quantization
            if "int8" in self.model_name or "int4" in self.model_name:
                from transformers import BitsAndBytesConfig

                if "int4" in self.model_name:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.dtype,
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
                # Remove torch_dtype for quantized models
                del load_kwargs["torch_dtype"]
            else:
                load_kwargs["device_map"] = self.device

            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

            self._is_loaded = True
            logger.info(f"OpenVLA model loaded on {self.device}")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            raise RuntimeError(
                "transformers library required for OpenVLA. "
                "Install with: pip install transformers torch torchvision"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load OpenVLA model: {e}")
            raise

    async def unload(self) -> None:
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("OpenVLA model unloaded")

    async def predict(
        self,
        image: Image.Image,
        instruction: str,
        proprioception: np.ndarray | None = None,
        unnorm_key: str | None = None,
    ) -> dict[str, Any]:
        """Predict actions from image and instruction.

        Args:
            image: Current camera image.
            instruction: Natural language task description.
            proprioception: Optional robot state for conditioning.
            unnorm_key: Dataset key for action unnormalization.

        Returns:
            Dictionary containing predicted actions and metadata.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Format prompt for OpenVLA
        # OpenVLA expects: "In: What action should the robot take to {instruction}?\nOut:"
        prompt = f"What action should the robot take to {instruction}?"

        # Prepare inputs
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode output
        generated_text = self._processor.decode(outputs[0], skip_special_tokens=True)

        # Extract action tokens
        # OpenVLA outputs actions in format: "[x, y, z, roll, pitch, yaw, gripper]"
        actions = self._parse_actions_from_text(generated_text)

        return {
            "actions": actions,
            "raw_text": generated_text,
            "confidence": 1.0,  # Placeholder - OpenVLA doesn't provide confidence
            "prompt": prompt,
        }

    def _parse_actions_from_text(self, text: str) -> np.ndarray:
        """Parse action array from generated text.

        OpenVLA typically outputs actions in format:
        "The robot should move [0.1, -0.2, 0.3, 0.0, 0.0, 0.1, 0.5]"

        Args:
            text: Generated text from model.

        Returns:
            Numpy array of actions [action_horizon, action_dim].
        """
        import re

        # Try to find array in text
        # Match patterns like [0.1, -0.2, 0.3] or [ 0.1 -0.2 0.3 ]
        patterns = [
            r'\[\s*([-\d.\s,]+)\s*\]',  # [0.1, -0.2, 0.3]
            r'action[s]?\s*[:=]?\s*([-\d.\s,]+)',  # actions: 0.1, -0.2, 0.3
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract numbers
                numbers_str = match.group(1)
                # Split by comma or whitespace
                numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', numbers_str)]

                if len(numbers) >= self.ACTION_DIM:
                    # Reshape to [horizon, action_dim]
                    actions = np.array(numbers[:self.ACTION_DIM * self.ACTION_HORIZON])
                    if actions.size >= self.ACTION_DIM:
                        actions = actions.reshape(-1, self.ACTION_DIM)
                        # Pad or truncate to action_horizon
                        if actions.shape[0] < self.ACTION_HORIZON:
                            padding = np.zeros(
                                (self.ACTION_HORIZON - actions.shape[0], self.ACTION_DIM)
                            )
                            actions = np.vstack([actions, padding])
                        else:
                            actions = actions[:self.ACTION_HORIZON]
                        return actions

        # Fallback: return zeros if parsing fails
        logger.warning(f"Failed to parse actions from text: {text[:200]}...")
        return np.zeros((self.ACTION_HORIZON, self.ACTION_DIM))

    def normalize_action(
        self,
        action: np.ndarray,
        stats: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Normalize action using dataset statistics.

        Args:
            action: Raw action array.
            stats: Dictionary with 'mean' and 'std' keys.

        Returns:
            Normalized action.
        """
        if stats is None:
            return action

        mean = stats.get("mean", np.zeros_like(action))
        std = stats.get("std", np.ones_like(action))

        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)

        return (action - mean) / std

    def unnormalize_action(
        self,
        action: np.ndarray,
        stats: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Unnormalize action using dataset statistics.

        Args:
            action: Normalized action array.
            stats: Dictionary with 'mean' and 'std' keys.

        Returns:
            Unnormalized action.
        """
        if stats is None:
            return action

        mean = stats.get("mean", np.zeros_like(action))
        std = stats.get("std", np.ones_like(action))

        return action * std + mean

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def device(self) -> str:
        """Get device string."""
        return self._device

    @property
    def model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "loaded": self._is_loaded,
            "action_dim": self.ACTION_DIM,
            "action_horizon": self.ACTION_HORIZON,
        }
