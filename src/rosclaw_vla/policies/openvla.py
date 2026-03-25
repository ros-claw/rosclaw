"""OpenVLA Policy - Integration with OpenVLA model."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import numpy as np
import torch
from PIL import Image

from .base import BasePolicy, PolicyOutput

logger = logging.getLogger(__name__)


class OpenVLAPolicy(BasePolicy):
    """OpenVLA policy implementation.

    Wraps the OpenVLA 7B parameter vision-language-action model
    for robot control. Supports both standard and streaming inference.
    """

    # Default image size for OpenVLA
    IMAGE_SIZE = (224, 224)

    def __init__(self, config: Any):
        """Initialize OpenVLA policy.

        Args:
            config: VLAConfig object.
        """
        super().__init__(config)
        self.model_name = config.model_name
        self.device = self._resolve_device(config.device)
        self.dtype = self._resolve_dtype(config.dtype)
        self.cache_dir = config.cache_dir

        self._model: Any = None
        self._processor: Any = None
        self._generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "do_sample": config.do_sample,
        }

    def _resolve_device(self, device: str) -> str:
        """Resolve device string."""
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
        """Load OpenVLA model and processor."""
        if self._loaded:
            return

        logger.info(f"Loading OpenVLA policy: {self.model_name}")

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            # Prepare model load kwargs
            load_kwargs = {
                "cache_dir": self.cache_dir,
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
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = self.dtype
                load_kwargs["device_map"] = self.device

            # Load model
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

            self._loaded = True
            logger.info(f"OpenVLA policy loaded on {self.device}")

        except ImportError as e:
            logger.error(f"Transformers not installed: {e}")
            raise RuntimeError(
                "transformers, torch, and torchvision required for OpenVLA"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load OpenVLA: {e}")
            raise

    async def unload(self) -> None:
        """Unload model and free memory."""
        if not self._loaded:
            return

        logger.info("Unloading OpenVLA policy")

        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._loaded = False

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("OpenVLA policy unloaded")

    async def predict(
        self,
        image: Image.Image,
        instruction: str,
        proprioception: np.ndarray | None = None,
    ) -> PolicyOutput:
        """Generate actions from image and instruction.

        Args:
            image: Current camera image.
            instruction: Natural language instruction.
            proprioception: Optional robot state (unused for OpenVLA).

        Returns:
            Policy output with actions.
        """
        self.validate_input(image, instruction)

        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded")

        # Preprocess image
        image = self.preprocess_image(image, self.IMAGE_SIZE)

        # Format prompt
        prompt = self._format_prompt(instruction)

        # Prepare inputs
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        # Move to device
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **self._generation_kwargs,
            )

        # Decode
        generated_text = self._processor.decode(outputs[0], skip_special_tokens=True)

        # Parse actions
        actions = self._parse_actions(generated_text)

        return PolicyOutput(
            raw_output=generated_text,
            confidence=self._calculate_confidence(generated_text, actions),
            metadata={
                "prompt": prompt,
                "actions": actions,
                "device": self.device,
            },
        )

    async def predict_stream(
        self,
        image: Image.Image,
        instruction: str,
        proprioception: np.ndarray | None = None,
    ) -> AsyncIterator[PolicyOutput]:
        """Stream actions as they are generated.

        Note: OpenVLA doesn't natively support streaming generation,
        so this yields intermediate tokens if possible.

        Args:
            image: Current camera image.
            instruction: Natural language instruction.
            proprioception: Optional robot state.

        Yields:
            Partial policy outputs.
        """
        # OpenVLA doesn't support true streaming, so just yield final result
        output = await self.predict(image, instruction, proprioception)
        yield output

    def _format_prompt(self, instruction: str) -> str:
        """Format instruction for OpenVLA.

        OpenVLA expects prompts in format:
        "What action should the robot take to {instruction}?"

        Args:
            instruction: Natural language instruction.

        Returns:
            Formatted prompt.
        """
        # OpenVLA was trained with specific prompt formats
        # Using the standard format from the paper
        return f"What action should the robot take to {instruction}?"

    def _parse_actions(self, text: str) -> np.ndarray:
        """Parse actions from generated text.

        Args:
            text: Generated text.

        Returns:
            Parsed actions array.
        """
        import re

        # Try to find array patterns
        patterns = [
            r'\[\s*([-\d.\s,]+)\s*\]',
            r'action[s]?\s*[:=]?\s*([-\d.\s,]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                numbers_str = match.group(1)
                numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', numbers_str)]

                if len(numbers) >= 7:
                    # Reshape to [horizon, action_dim]
                    action_dim = 7
                    horizon = len(numbers) // action_dim
                    actions = np.array(numbers[:horizon * action_dim])
                    return actions.reshape(horizon, action_dim)

        # Fallback: extract all numbers
        numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', text)]
        if len(numbers) >= 7:
            return np.array(numbers[:7])

        # Return zeros if parsing fails
        logger.warning(f"Failed to parse actions from: {text[:100]}...")
        return np.zeros((1, 7))

    def _calculate_confidence(
        self,
        text: str,
        actions: np.ndarray,
    ) -> float:
        """Calculate confidence score.

        Args:
            text: Generated text.
            actions: Parsed actions.

        Returns:
            Confidence score (0-1).
        """
        # Simple heuristic: longer text with more numbers = higher confidence
        confidence = 0.5

        # Check if we found actions
        if np.any(actions != 0):
            confidence += 0.3

        # Check for action-related keywords
        keywords = ["action", "move", "gripper", "position"]
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                confidence += 0.05

        return min(confidence, 1.0)

    @property
    def supports_streaming(self) -> bool:
        """OpenVLA doesn't support true streaming."""
        return False

    @property
    def device(self) -> str:
        """Get current device."""
        return str(self._device)
