"""Provider-neutral ModelGateway probe behaviour."""

from __future__ import annotations

from typing import Any

import pytest

from rosclaw.agentd.models.gateway import OpenAICompatGateway
from rosclaw.agentd.models.policy import ModelProfile


class _VllmLikeRuntime:
    """Reject system-only chat requests like vLLM's OpenAI endpoint does."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def health_detail(self) -> dict[str, Any]:
        return {
            "reachable": True,
            "served_models": ["qwen3.6-27b"],
            "expected_model_present": True,
        }

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = payload["inputs"]
        self.requests.append(inputs)
        messages = inputs["messages"]
        if not any(message.get("role") == "user" for message in messages):
            raise AssertionError("No user query found in messages")
        if inputs.get("tools"):
            return {
                "model": "qwen3.6-27b",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_ping",
                            "type": "function",
                            "function": {"name": "ping", "arguments": '{"echo": true}'},
                        }
                    ],
                },
                "tool_calls": [
                    {
                        "id": "call_ping",
                        "type": "function",
                        "function": {"name": "ping", "arguments": '{"echo": true}'},
                    }
                ],
                "finish_reason": "tool_calls",
                "usage": {},
            }
        return {
            "model": "qwen3.6-27b",
            "result": "ok",
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
            "usage": {},
        }


@pytest.mark.asyncio
async def test_probe_supplies_user_message_for_vllm_compatible_servers() -> None:
    gateway = OpenAICompatGateway(
        ModelProfile(
            name="qwen_vision",
            provider="qwen_lan",
            model="qwen3.6-27b",
            base_url="http://127.0.0.1:30521/v1",
            local=True,
        )
    )
    assert gateway._runtime.trust_env is False
    runtime = _VllmLikeRuntime()
    gateway._runtime = runtime  # type: ignore[assignment]

    try:
        result = await gateway.probe()
    finally:
        await gateway.close()

    assert result.chat_ok is True
    assert result.tool_call_ok is True
    assert all(
        any(message.get("role") == "user" for message in request["messages"])
        for request in runtime.requests
    )
