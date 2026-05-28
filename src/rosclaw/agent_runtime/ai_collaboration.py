"""
ROSClaw AI Collaboration - DeepSeek Integration

Provides integration with DeepSeek API for AI collaboration.
This module enables:
- LLM-based trajectory planning
- Natural language task understanding
- Multi-agent reasoning
- Skill synthesis from demonstrations

Configuration:
    Set DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL environment variables,
    or pass them directly to DeepSeekClient.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek API client."""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    temperature: float = 0.7
    max_tokens: int = 4096


class DeepSeekClient:
    """
    Client for DeepSeek API integration.

    Provides AI collaboration capabilities for ROSClaw:
    - Task planning from natural language
    - Trajectory reasoning
    - Error analysis and recovery
    """

    def __init__(self, config: Optional[DeepSeekConfig] = None):
        self.config = config or DeepSeekConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazy initialization of HTTP client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
            except ImportError:
                raise RuntimeError(
                    "openai package required for DeepSeek integration. "
                    "Install with: pip install openai"
                )
        return self._client

    def plan_task(self, instruction: str, robot_context: dict) -> dict:
        """
        Generate a task plan from natural language instruction.

        Args:
            instruction: Natural language task description
            robot_context: Current robot state and capabilities

        Returns:
            Task plan with steps and parameters
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
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
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
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
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
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content) if content else {"error": "empty response"}
        except Exception as e:
            return {"error": str(e), "skill_name": "unknown"}
