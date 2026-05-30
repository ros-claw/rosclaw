"""DeepSeek LLM Provider for ROSClaw.

Capabilities:
    - llm.task_planning: generate task plans from natural language
    - llm.summary: summarize execution traces or observations
    - llm.chat: general conversational interface

Configuration via environment:
    DEEPSEEK_API_KEY: API key (required)
    DEEPSEEK_BASE_URL: defaults to https://api.deepseek.com
    DEEPSEEK_MODEL: defaults to deepseek-v4-flash
"""

from __future__ import annotations

import os
from typing import Any

from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class DeepSeekProvider(Provider):
    """DeepSeek API provider for LLM capabilities.

    Uses urllib for minimal dependencies (no aiohttp required).
    """

    name = "deepseek"
    version = "1.0.0"
    capabilities = [
        "llm.task_planning",
        "llm.summary",
        "llm.chat",
    ]

    def __init__(self, manifest):
        super().__init__(manifest)
        self._api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self._model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self._healthy = bool(self._api_key)

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        self._ensure_capability_supported(request.capability)

        if not self._api_key:
            return ProviderResponse(
                request_id=request.request_id,
                provider=self.name,
                capability=request.capability,
                result={"error": "DEEPSEEK_API_KEY not set"},
                status="error",
            )

        # Build prompt from capability + inputs
        prompt = self._build_prompt(request)

        try:
            result = await self._call_api(prompt)
            return ProviderResponse(
                request_id=request.request_id,
                provider=self.name,
                capability=request.capability,
                result=result,
                status="ok",
                latency_ms=result.get("_latency_ms", 0),
            )
        except Exception as e:
            return ProviderResponse(
                request_id=request.request_id,
                provider=self.name,
                capability=request.capability,
                result={"error": str(e)},
                status="error",
            )

    def _build_prompt(self, request: ProviderRequest) -> str:
        cap = request.capability
        inputs = request.inputs or {}

        if cap == "llm.task_planning":
            task = inputs.get("task", "")
            robot = inputs.get("robot_id", "unknown")
            return (
                f"You are a robot task planner. The robot is {robot}.\n"
                f"Task: {task}\n"
                f"Generate a step-by-step plan as a JSON list of actions."
            )

        if cap == "llm.summary":
            text = inputs.get("text", "")
            return f"Summarize the following robot execution trace concisely:\n\n{text}"

        # Default chat
        return inputs.get("message", inputs.get("query", "Hello"))

    async def _call_api(self, prompt: str) -> dict[str, Any]:
        import asyncio
        import json
        import re
        import time
        import urllib.request

        start = time.time()

        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.3,
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        # Run blocking HTTP call in executor
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, urllib.request.urlopen, req)
        raw = resp.read()

        # Robust JSON parsing with cleanup for malformed LLM output
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            text = raw.decode("utf-8", errors="ignore")
            text = re.sub(r",(\s*[}\]])", r"\1", text)
            try:
                body = json.loads(text)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"DeepSeek returned invalid JSON: {e}") from e

        latency_ms = round((time.time() - start) * 1000, 2)

        # Defensive response parsing — CRITICAL fix from audit
        if not isinstance(body, dict):
            raise RuntimeError(f"DeepSeek returned non-dict body: {type(body)}")
        if body.get("error"):
            err = body["error"]
            raise RuntimeError(f"DeepSeek API error: {err.get('message', err)}")
        choices = body.get("choices")
        if not choices or not isinstance(choices, list):
            raise RuntimeError(f"DeepSeek missing choices in response keys: {list(body.keys())}")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError(f"DeepSeek first choice is not dict: {type(first_choice)}")
        message = first_choice.get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError(f"DeepSeek message is not dict: {type(message)}")
        content = message.get("content", "")
        if not content:
            raise RuntimeError("DeepSeek returned empty content")

        return {
            "text": content,
            "model": self._model,
            "_latency_ms": latency_ms,
        }
