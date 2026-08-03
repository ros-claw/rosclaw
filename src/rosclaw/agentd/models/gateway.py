"""ModelGateway — unified model turns over the provider layer (PR-NA-030).

- ``OpenAICompatGateway`` wraps ``OpenAICompatRuntime`` (no new HTTP stack)
  and adapts its results to ``ModelTurnResultV1`` with the full assistant
  message preserved for tool-loop continuity.
- ``MockModelGateway`` is a scripted, deterministic gateway for tests —
  the whole AgentLoop state machine runs against it without any network.
- Strict tools: JSON Schema objects must carry ``additionalProperties:
  false`` and a full ``required`` list; the gateway refuses to send lax
  tool schemas (fail closed).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rosclaw.agentd.models.policy import ModelProfile
from rosclaw.contracts.agent.model_turn import (
    ModelTurnResultV1,
    ModelUsage,
    ToolCall,
)
from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.provider.core.errors import RuntimeAdapterError
from rosclaw.provider.runtimes.openai_compat_runtime import OpenAICompatRuntime

#: Standard OpenAI chat message fields; anything else is internal bookkeeping.
_WIRE_MESSAGE_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


def _sanitize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Strip internal journal keys before a message crosses the provider wire."""
    if set(message) <= _WIRE_MESSAGE_KEYS:
        return message
    return {k: v for k, v in message.items() if k in _WIRE_MESSAGE_KEYS}


class ModelGatewayError(Exception):
    """Model invocation failed in a classified, diagnosable way."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


@dataclass(frozen=True)
class StrictTool:
    """A tool definition with a strict JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]

    def validate(self) -> None:
        if self.parameters.get("type") != "object":
            raise ValidationError(f"tool {self.name!r} schema must be an object")
        if self.parameters.get("additionalProperties") is not False:
            raise ValidationError(f"tool {self.name!r} schema must set additionalProperties: false")
        properties = self.parameters.get("properties") or {}
        required = set(self.parameters.get("required") or [])
        if required != set(properties):
            raise ValidationError(f"tool {self.name!r} required list must match properties exactly")

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ModelTurnRequest:
    system_prompt: str
    messages: list[dict[str, Any]]
    tools: list[StrictTool] = field(default_factory=list)
    tool_choice: str | None = None
    max_output_tokens: int = 16_000
    mission_id: str | None = None
    context_id: str | None = None
    context_revision: int | None = None


@dataclass(frozen=True)
class ModelProbeResult:
    reachable: bool
    models_visible: tuple[str, ...] = ()
    expected_model_present: bool | None = None
    chat_ok: bool | None = None
    tool_call_ok: bool | None = None
    error: str | None = None


class ModelGateway(Protocol):
    profile: ModelProfile

    async def complete(self, request: ModelTurnRequest) -> ModelTurnResultV1: ...

    async def complete_stream(
        self, request: ModelTurnRequest, on_text_delta=None
    ) -> ModelTurnResultV1: ...

    async def probe(self) -> ModelProbeResult: ...

    async def close(self) -> None: ...


def _resolve_api_key(api_key_ref: str) -> str:
    """Resolve ``env:VAR`` references. Never logs or returns the raw ref."""
    import os

    if api_key_ref.startswith("env:"):
        value = os.environ.get(api_key_ref[4:], "")
        if not value:
            raise ModelGatewayError(
                "missing_credential",
                f"environment variable {api_key_ref[4:]} is not set",
            )
        return value
    if api_key_ref:
        raise ModelGatewayError(
            "unsupported_credential_ref",
            "only env: credential references are supported in P0",
        )
    return ""


def key_fingerprint(api_key: str) -> str:
    """SHA-256 prefix for local doctor diagnostics. Never the key itself."""
    import hashlib

    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:8]


class OpenAICompatGateway:
    """Kimi K3 and any OpenAI-compatible endpoint via the provider runtime."""

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile
        api_key = _resolve_api_key(profile.api_key_ref)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._runtime = OpenAICompatRuntime(
            name=f"agentd.{profile.name}",
            endpoint=profile.base_url,
            model=profile.model,
            timeout_sec=profile.timeout_sec,
            retries=profile.retry_attempts,
            headers=headers,
        )
        self._started = False

    async def _ensure_started(self) -> None:
        if not self._started:
            await self._runtime.start()
            self._started = True

    async def close(self) -> None:
        if self._started:
            await self._runtime.stop()
            self._started = False

    def _build_inputs(self, request: ModelTurnRequest) -> dict[str, Any]:
        for tool in request.tools:
            tool.validate()
        messages = [{"role": "system", "content": request.system_prompt}]
        # Internal journal keys (entry_id/seq/atomic_group/source/ref …) are
        # local bookkeeping; providers only accept standard message fields.
        messages.extend(_sanitize_message(m) for m in request.messages)
        inputs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": request.max_output_tokens,
        }
        if request.tools:
            inputs["tools"] = [t.to_openai() for t in request.tools]
            inputs["tool_choice"] = request.tool_choice or "auto"
        if self.profile.vendor_parameters:
            inputs["vendor_parameters"] = dict(self.profile.vendor_parameters)
        return inputs

    def _usage_from_raw(self, usage_raw: dict[str, Any]) -> ModelUsage:
        prompt = int(usage_raw.get("prompt_tokens") or 0)
        completion = int(usage_raw.get("completion_tokens") or 0)
        details = usage_raw.get("completion_tokens_details") or {}
        reasoning = int(details.get("reasoning_tokens") or 0)
        from rosclaw.agentd.usage import estimate_cost_microunits

        return ModelUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            reasoning_tokens=reasoning,
            total_tokens=int(usage_raw.get("total_tokens") or prompt + completion),
            cost_microunits=estimate_cost_microunits(
                prompt_tokens=prompt,
                completion_tokens=completion,
                price_input_per_mtok=self.profile.price_input_per_mtok_microunits,
                price_output_per_mtok=self.profile.price_output_per_mtok_microunits,
            ),
        )

    def _checked_tool_calls(self, raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        tool_calls = []
        for call in raw_calls:
            function = call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            # Malformed arguments must surface as an error, not a silent "{}".
            try:
                json.loads(arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ModelGatewayError(
                    "malformed_tool_arguments",
                    f"tool call {function.get('name')!r} arguments are not JSON",
                ) from exc
            tool_calls.append(
                ToolCall(
                    call_id=call.get("id") or new_id("call"),
                    name=function.get("name") or "",
                    arguments_json=arguments,
                )
            )
        return tool_calls

    async def complete(self, request: ModelTurnRequest) -> ModelTurnResultV1:
        await self._ensure_started()
        inputs = self._build_inputs(request)
        started = time.monotonic()
        try:
            raw = await self._runtime.invoke({"inputs": inputs})
        except RuntimeAdapterError as exc:
            raise ModelGatewayError(exc.kind or "invoke_failed", str(exc)) from exc
        return ModelTurnResultV1(
            turn_id=new_id("turn"),
            mission_id=request.mission_id,
            provider=self.profile.provider,
            model=raw.get("model") or self.profile.model,
            profile=self.profile.name,
            content=raw.get("result") or "",
            tool_calls=self._checked_tool_calls(raw.get("tool_calls") or []),
            assistant_message=raw.get("message") or {},
            finish_reason=raw.get("finish_reason"),
            usage=self._usage_from_raw(raw.get("usage") or {}),
            provider_request_id=raw.get("request_id"),
            latency_ms=int((time.monotonic() - started) * 1000),
            context_id=request.context_id,
            context_revision=request.context_revision,
        )

    async def complete_stream(
        self,
        request: ModelTurnRequest,
        on_text_delta=None,
    ) -> ModelTurnResultV1:
        """Streaming complete. Aggregates SSE deltas (picoclaw pattern):
        text concatenated (reported via ``on_text_delta``), tool_calls
        assembled by index with arguments string-append, usage taken from
        the last non-empty chunk (include_usage is set by the runtime).
        """
        await self._ensure_started()
        inputs = self._build_inputs(request)
        started = time.monotonic()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        usage_raw: dict[str, Any] = {}
        finish_reason: str | None = None
        model_seen = ""
        request_id = ""
        try:
            async for chunk in self._runtime.invoke_stream({"inputs": inputs}):
                request_id = request_id or chunk.get("id") or ""
                model_seen = chunk.get("model") or model_seen
                if chunk.get("usage"):
                    usage_raw = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        content_parts.append(piece)
                        if on_text_delta is not None:
                            on_text_delta(piece)
                    reasoning_piece = delta.get("reasoning_content")
                    if reasoning_piece:
                        reasoning_parts.append(reasoning_piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    for call_delta in delta.get("tool_calls") or []:
                        idx = int(call_delta.get("index") or 0)
                        acc = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if call_delta.get("id"):
                            acc["id"] = call_delta["id"]
                        function = call_delta.get("function") or {}
                        if function.get("name"):
                            acc["name"] = function["name"]
                        if function.get("arguments"):
                            acc["arguments"] += function["arguments"]
        except RuntimeAdapterError as exc:
            raise ModelGatewayError(exc.kind or "stream_failed", str(exc)) from exc
        raw_calls = [
            {
                "id": acc["id"],
                "type": "function",
                "function": {"name": acc["name"], "arguments": acc["arguments"] or "{}"},
            }
            for _, acc in sorted(tool_acc.items())
        ]
        tool_calls = self._checked_tool_calls(raw_calls)
        content = "".join(content_parts)
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if reasoning_parts:
            assistant_message["reasoning_content"] = "".join(reasoning_parts)
        if raw_calls:
            assistant_message["tool_calls"] = raw_calls
        return ModelTurnResultV1(
            turn_id=new_id("turn"),
            mission_id=request.mission_id,
            provider=self.profile.provider,
            model=model_seen or self.profile.model,
            profile=self.profile.name,
            content=content,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            finish_reason=finish_reason,
            usage=self._usage_from_raw(usage_raw),
            provider_request_id=request_id or None,
            latency_ms=int((time.monotonic() - started) * 1000),
            context_id=request.context_id,
            context_revision=request.context_revision,
        )

    async def probe(self) -> ModelProbeResult:
        """K0 probe: models listing + short chat + one strict tool call."""
        await self._ensure_started()
        health = await self._runtime.health_detail()
        if not health.get("reachable"):
            return ModelProbeResult(reachable=False, error=health.get("error"))
        visible = tuple(health.get("served_models") or ())
        expected = health.get("expected_model_present")
        probe = ModelProbeResult(
            reachable=True, models_visible=visible, expected_model_present=expected
        )
        try:
            turn = await self.complete(
                ModelTurnRequest(
                    system_prompt="Reply with exactly: ok", messages=[], max_output_tokens=8
                )
            )
            probe = ModelProbeResult(
                reachable=True,
                models_visible=visible,
                expected_model_present=expected,
                chat_ok=bool(turn.content or turn.assistant_message),
            )
        except ModelGatewayError as exc:
            return ModelProbeResult(
                reachable=True,
                models_visible=visible,
                expected_model_present=expected,
                chat_ok=False,
                error=f"chat probe: {exc}",
            )
        ping_tool = StrictTool(
            name="ping",
            description="Connectivity check. Call with echo=true.",
            parameters={
                "type": "object",
                "properties": {"echo": {"type": "boolean"}},
                "required": ["echo"],
                "additionalProperties": False,
            },
        )
        try:
            turn = await self.complete(
                ModelTurnRequest(
                    system_prompt=(
                        "You must call the ping tool with echo=true. Do not answer in text."
                    ),
                    messages=[{"role": "user", "content": "ping"}],
                    tools=[ping_tool],
                    tool_choice="required",
                    max_output_tokens=256,
                )
            )
            ok = any(c.name == "ping" for c in turn.tool_calls)
            probe = ModelProbeResult(
                reachable=True,
                models_visible=visible,
                expected_model_present=expected,
                chat_ok=True,
                tool_call_ok=ok,
                error=None if ok else "model did not emit the required tool call",
            )
        except ModelGatewayError as exc:
            probe = ModelProbeResult(
                reachable=True,
                models_visible=visible,
                expected_model_present=expected,
                chat_ok=True,
                tool_call_ok=False,
                error=f"tool probe: {exc}",
            )
        return probe


class MockModelGateway:
    """Scripted deterministic gateway for tests (no network)."""

    def __init__(
        self,
        profile: ModelProfile,
        script: list[ModelTurnResultV1 | Callable[[ModelTurnRequest], ModelTurnResultV1]],
    ) -> None:
        self.profile = profile
        self._script = list(script)
        self.requests: list[ModelTurnRequest] = []

    async def complete(self, request: ModelTurnRequest) -> ModelTurnResultV1:
        self.requests.append(request)
        if not self._script:
            raise ModelGatewayError("script_exhausted", "mock gateway has no more turns")
        item = self._script.pop(0)
        result = item(request) if callable(item) else item
        # Mirror the real gateway: price the turn from the profile's rates.
        from rosclaw.agentd.usage import estimate_cost_microunits

        usage = result.usage.model_copy(
            update={
                "cost_microunits": estimate_cost_microunits(
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                    price_input_per_mtok=self.profile.price_input_per_mtok_microunits,
                    price_output_per_mtok=self.profile.price_output_per_mtok_microunits,
                )
            }
        )
        return result.model_copy(
            update={
                "mission_id": request.mission_id,
                "context_id": request.context_id,
                "context_revision": request.context_revision,
                "usage": usage,
            }
        )

    async def complete_stream(
        self, request: ModelTurnRequest, on_text_delta=None
    ) -> ModelTurnResultV1:
        result = await self.complete(request)
        if on_text_delta is not None and result.content:
            # Deterministic chunking: two halves, like a real token stream.
            mid = max(1, len(result.content) // 2)
            on_text_delta(result.content[:mid])
            on_text_delta(result.content[mid:])
        return result

    async def probe(self) -> ModelProbeResult:
        return ModelProbeResult(
            reachable=True,
            models_visible=(self.profile.model,),
            expected_model_present=True,
            chat_ok=True,
            tool_call_ok=True,
        )

    async def close(self) -> None:
        return None
