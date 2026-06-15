"""
ROSClaw LLM Provider Abstraction Layer

Provides a unified interface for multiple LLM backends:
- DeepSeek (deepseek-v4-pro, etc.)
- OpenAI (GPT-4, GPT-4o, etc.)
- Qwen (qwen-max, qwen-plus, etc.)

All providers implement the same LLMProvider ABC, ensuring
AgentRuntime can switch backends without code changes.

Configuration:
    Set PROVIDER_TYPE env var or pass provider_type to AgentRuntime.
    Provider-specific API keys use standard env var names:
    - DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL
    - OPENAI_API_KEY / OPENAI_BASE_URL
    - DASHSCOPE_API_KEY (for Qwen)
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Base configuration for any LLM provider."""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 30.0
    extra_headers: dict = field(default_factory=dict)


@dataclass
class TaskPlan:
    """Structured task plan from LLM."""
    task_name: str
    steps: list[dict]
    safety_notes: list[str]


@dataclass
class FailureAnalysis:
    """Failure analysis from LLM."""
    root_cause: str
    severity: str
    recovery_strategy: str
    preventive_measures: list[str]


@dataclass
class SkillDescription:
    """Skill description from LLM."""
    skill_name: str
    description: str
    preconditions: list[str]
    parameters: dict[str, str]
    success_criteria: list[str]


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All ROSClaw-supported LLM backends implement this interface.
    The AgentRuntime uses this ABC to remain provider-agnostic.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Any | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""
        ...

    def _get_client(self) -> Any:
        """Lazy initialization of HTTP client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @abstractmethod
    def _create_client(self) -> Any:
        """Create and return the provider-specific client."""
        ...

    def _chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Unified chat completion call.

        Returns raw response content string.
        Subclasses may override for provider-specific quirks.
        """
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content or ""

    def plan_task(self, instruction: str, robot_context: dict) -> dict:
        """
        Generate a task plan from natural language instruction.

        Args:
            instruction: Natural language task description
            robot_context: Current robot state and capabilities

        Returns:
            Task plan dict with steps and parameters
        """
        system_prompt = """You are a robot task planner for ROSClaw.
Given a natural language instruction and robot context,
generate a structured task plan with specific robot commands.

Respond in JSON format:
{
    "task_name": "string",
    "steps": [
        {
            "action": "move_joints|grasp|wait|...",
            "parameters": {},
            "description": "human-readable explanation"
        }
    ],
    "safety_notes": ["list of safety considerations"]
}"""

        user_prompt = f"""Instruction: {instruction}

Robot Context:
{json.dumps(robot_context, indent=2)}

Generate a task plan."""

        try:
            content = self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format={"type": "json_object"},
            )
            return json.loads(content) if content else {"error": "empty response"}
        except Exception as e:
            return {"error": str(e), "task_name": "failed", "steps": []}

    def analyze_failure(self, task_description: str, error_log: str,
                        heuristic_engine=None) -> dict:
        """
        Analyze a task failure and suggest recovery.

        Args:
            task_description: What the robot was trying to do
            error_log: Error messages and state information
            heuristic_engine: Optional HeuristicEngine for fast rule lookup

        Returns:
            Analysis and recovery suggestions
        """
        # 1. Try heuristic first (fast, deterministic, free)
        if heuristic_engine is not None:
            try:
                import asyncio
                heuristic = asyncio.run(heuristic_engine.suggest_recovery(error_log))
                if heuristic:
                    return {
                        "root_cause": "matched_heuristic",
                        "severity": "medium",
                        "recovery_strategy": heuristic["action"],
                        "preventive_measures": [
                            f"Rule {heuristic['rule_id']}: {heuristic['condition']}"
                        ],
                        "source": "heuristic",
                    }
            except Exception:
                pass  # fallback to LLM

        # 2. Fall back to LLM (slow, expensive, but handles novel errors)
        system_prompt = """You are a robot failure analyst for ROSClaw.
Analyze task failures and suggest recovery strategies.

Respond in JSON format:
{
    "root_cause": "string",
    "severity": "low|medium|high|critical",
    "recovery_strategy": "string",
    "preventive_measures": ["list of suggestions"]
}"""

        user_prompt = f"""Task: {task_description}

Error Log:
{error_log}

Analyze this failure."""

        try:
            content = self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            return json.loads(content) if content else {"error": "empty response"}
        except Exception as e:
            return {"error": str(e), "root_cause": "analysis_failed"}

    def generate_skill_description(self, demonstration: dict) -> dict:
        """
        Generate a natural language skill description from demonstration data.

        Args:
            demonstration: Recorded demonstration data

        Returns:
            Skill description with parameters and constraints
        """
        system_prompt = """You are a robot skill synthesizer for ROSClaw.
Convert demonstration data into reusable skill descriptions.

Respond in JSON format:
{
    "skill_name": "string",
    "description": "natural language description",
    "preconditions": ["list of required conditions"],
    "parameters": {"param_name": "description"},
    "success_criteria": ["list of success conditions"]
}"""

        user_prompt = f"""Demonstration Data:
{json.dumps(demonstration, indent=2, default=str)}

Synthesize a skill description."""

        try:
            content = self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.5,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            return json.loads(content) if content else {"error": "empty response"}
        except Exception as e:
            return {"error": str(e), "skill_name": "unknown"}

    def health_check(self) -> dict:
        """
        Verify provider connectivity.

        Returns:
            Dict with "ok" bool and optional "error" string.
        """
        try:
            client = self._get_client()
            # Lightweight call: list models or send a minimal completion
            # Most OpenAI-compatible APIs support a simple completion
            _ = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            return {"ok": True, "provider": self.provider_name}
        except Exception as e:
            return {"ok": False, "provider": self.provider_name, "error": str(e)}


class DeepSeekProvider(LLMProvider):
    """
    DeepSeek API provider.

    Uses OpenAI-compatible client with DeepSeek base URL.
    Models: deepseek-v4-pro, deepseek-chat, deepseek-coder, etc.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-v4-pro"

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        if config is not None and kwargs:
            raise TypeError("Cannot pass both config and keyword arguments")
        cfg = config if config is not None else LLMConfig(**kwargs)
        if not cfg.api_key:
            cfg.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not cfg.base_url:
            cfg.base_url = os.getenv("DEEPSEEK_BASE_URL", self.DEFAULT_BASE_URL)
        if not cfg.model:
            cfg.model = self.DEFAULT_MODEL
        super().__init__(cfg)

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def _create_client(self) -> Any:
        try:
            import openai
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                default_headers=self.config.extra_headers,
            )
        except ImportError as err:
            raise RuntimeError(
                "openai package required for DeepSeek integration. "
                "Install with: pip install openai"
            ) from err


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider.

    Native OpenAI client.
    Models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        if config is not None and kwargs:
            raise TypeError("Cannot pass both config and keyword arguments")
        cfg = config if config is not None else LLMConfig(**kwargs)
        if not cfg.api_key:
            cfg.api_key = os.getenv("OPENAI_API_KEY", "")
        if not cfg.base_url:
            cfg.base_url = os.getenv("OPENAI_BASE_URL", self.DEFAULT_BASE_URL)
        if not cfg.model:
            cfg.model = self.DEFAULT_MODEL
        super().__init__(cfg)

    @property
    def provider_name(self) -> str:
        return "openai"

    def _create_client(self) -> Any:
        try:
            import openai
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                default_headers=self.config.extra_headers,
            )
        except ImportError as err:
            raise RuntimeError(
                "openai package required for OpenAI integration. "
                "Install with: pip install openai"
            ) from err


class QwenProvider(LLMProvider):
    """
    Alibaba Qwen API provider.

    Uses OpenAI-compatible client with DashScope base URL.
    Models: qwen-max, qwen-plus, qwen-turbo, etc.
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen-max"

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        if config is not None and kwargs:
            raise TypeError("Cannot pass both config and keyword arguments")
        cfg = config if config is not None else LLMConfig(**kwargs)
        if not cfg.api_key:
            cfg.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not cfg.base_url:
            cfg.base_url = os.getenv("DASHSCOPE_BASE_URL", self.DEFAULT_BASE_URL)
        if not cfg.model:
            cfg.model = self.DEFAULT_MODEL
        super().__init__(cfg)

    @property
    def provider_name(self) -> str:
        return "qwen"

    def _create_client(self) -> Any:
        try:
            import openai
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                default_headers=self.config.extra_headers,
            )
        except ImportError as err:
            raise RuntimeError(
                "openai package required for Qwen integration. "
                "Install with: pip install openai"
            ) from err


# Provider registry for runtime selection
_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "deepseek": DeepSeekProvider,
    "openai": OpenAIProvider,
    "qwen": QwenProvider,
}


def get_provider(name: str, config: LLMConfig | None = None) -> LLMProvider:
    """
    Factory function to get a provider by name.

    Args:
        name: Provider name (deepseek, openai, qwen)
        config: Optional LLMConfig override

    Returns:
        Configured LLMProvider instance

    Raises:
        ValueError: If provider name is unknown
    """
    name = name.lower().strip()
    if name not in _PROVIDER_REGISTRY:
        available = ", ".join(_PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown LLM provider '{name}'. Available: {available}")
    return _PROVIDER_REGISTRY[name](config)


def list_providers() -> list[str]:
    """List available provider names."""
    return list(_PROVIDER_REGISTRY.keys())


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    """
    Register a custom provider at runtime.

    Args:
        name: Provider identifier
        cls: LLMProvider subclass
    """
    if not issubclass(cls, LLMProvider):
        raise TypeError(f"Provider must subclass LLMProvider, got {cls}")
    _PROVIDER_REGISTRY[name.lower().strip()] = cls
