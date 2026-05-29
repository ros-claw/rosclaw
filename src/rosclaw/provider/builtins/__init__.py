"""Built-in mock providers for out-of-box capability support.

These providers implement mock responses for common capability domains,
useful for testing, development, and demos without requiring real models.
"""

from rosclaw.provider.builtins.vlm import MockVLMProvider
from rosclaw.provider.builtins.skill import MockSkillProvider
from rosclaw.provider.builtins.critic import MockCriticProvider
from rosclaw.provider.builtins.deepseek import DeepSeekProvider

__all__ = ["MockVLMProvider", "MockSkillProvider", "MockCriticProvider", "DeepSeekProvider"]
