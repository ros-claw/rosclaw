"""OpenAI-compatible HTTP runtime adapter.

Supports any server exposing the OpenAI API surface (vLLM, SGLang, TGI,
llama.cpp, LM Studio, ...). Selected with ``runtime.backend:
openai_compatible`` in provider.yaml; ``runtime.endpoint`` is the API base
URL (e.g. ``http://127.0.0.1:8000/v1``).

Manifest env hints (all optional):

- ``api_kind``: ``chat_completions`` (default) or ``embeddings``
- ``model``: served model name sent in requests; ``model_fallback`` is used
  for health matching when the server reports a path-like model id
- ``health_endpoint``: override for the health probe (default
  ``{endpoint}/models``)
- ``timeout_sec``, ``retries``, ``headers`` (same semantics as HTTPRuntime)

Chat requests additionally support multimodal input via
``inputs["content_parts"]`` (OpenAI content items such as ``image_url`` /
``video_url``) and pass-through of ``mm_processor_kwargs`` /
``min_pixels`` / ``max_pixels``.

Failures raise ``RuntimeAdapterError`` with a structured ``kind``
(``timeout`` / ``unavailable`` / ``http_error`` / ``invalid_response``)
so router fallback policies can match on failure categories instead of
parsing message strings.
"""

from __future__ import annotations

import asyncio
import ipaddress
from contextlib import suppress
from typing import Any
from urllib.parse import urlparse

from rosclaw.provider.core.errors import RuntimeAdapterError
from rosclaw.provider.runtimes.base import RuntimeAdapter

_API_CHAT = "chat_completions"
_API_EMBEDDINGS = "embeddings"


class OpenAICompatRuntime(RuntimeAdapter):
    """Runtime adapter for OpenAI-compatible inference servers."""

    def __init__(
        self,
        name: str,
        endpoint: str,
        api_kind: str = _API_CHAT,
        model: str = "",
        model_fallback: str = "",
        health_endpoint: str = "",
        timeout_sec: float = 30.0,
        retries: int = 1,
        headers: dict[str, str] | None = None,
        stream_idle_timeout_sec: float = 60.0,
        trust_env: bool | None = None,
    ):
        super().__init__(name, config={"endpoint": endpoint, "timeout": timeout_sec})
        if api_kind not in (_API_CHAT, _API_EMBEDDINGS):
            raise RuntimeAdapterError(
                f"Unsupported api_kind for openai_compatible backend: {api_kind!r}",
                provider=name,
            )
        self.endpoint = endpoint.rstrip("/")
        self.api_kind = api_kind
        self.model = model
        self.model_fallback = model_fallback
        self.health_endpoint = health_endpoint or f"{self.endpoint}/models"
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.headers = headers or {}
        self.stream_idle_timeout_sec = stream_idle_timeout_sec
        self.trust_env = trust_env
        self._session = None

    async def start(self) -> None:
        try:
            import aiohttp
        except ImportError as err:
            raise RuntimeError(
                "aiohttp is required for OpenAICompatRuntime. pip install aiohttp"
            ) from err
        # Respect the operator's standard HTTP(S)_PROXY / NO_PROXY settings for
        # cloud providers. Always keep loopback runtimes local, even when a host
        # forgot to include localhost in NO_PROXY.
        host = urlparse(self.endpoint).hostname or ""
        loopback = host.lower() == "localhost"
        if host:
            with suppress(ValueError):
                loopback = loopback or ipaddress.ip_address(host).is_loopback
        trust_env = not loopback if self.trust_env is None else self.trust_env
        self._session = aiohttp.ClientSession(headers=self.headers, trust_env=trust_env)
        self._started = True

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._started = False

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------
    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_started()
        if self.api_kind == _API_EMBEDDINGS:
            path, body = self._build_embeddings_request(payload)
        else:
            path, body = self._build_chat_request(payload)
        raw = await self._post(path, body)
        if self.api_kind == _API_EMBEDDINGS:
            return self._parse_embeddings_response(raw)
        return self._parse_chat_response(raw)

    # ------------------------------------------------------------------
    # Streaming (PR-NA-030b; SSE per picoclaw parseStreamResponse)
    # ------------------------------------------------------------------
    async def invoke_stream(self, payload: dict[str, Any]):
        """Yield raw SSE chunk dicts of a streaming chat completion.

        Requests ``stream_options.include_usage`` so the final chunks carry
        token usage (hermes pattern). Yields parsed JSON chunk objects;
        the caller aggregates text/tool_calls/usage. The literal ``[DONE]``
        sentinel terminates the stream. Raises RuntimeAdapterError on
        classified failures; retries happen only *before* the stream opens
        (mid-stream retries would duplicate side effects).
        """
        self.ensure_started()
        if self.api_kind != _API_CHAT:
            raise RuntimeAdapterError(
                "streaming is only supported for chat_completions", provider=self.name
            )
        path, body = self._build_chat_request(payload)
        body = dict(body)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        async for chunk in self._post_stream(path, body):
            yield chunk

    _SSE_MAX_BUFFER = 10 * 1024 * 1024  # picoclaw's 10MB cap

    async def _post_stream(self, path: str, body: dict[str, Any]):
        import aiohttp

        client_error = getattr(aiohttp, "ClientError", OSError)
        if not isinstance(client_error, type) or not issubclass(client_error, BaseException):
            client_error = OSError

        if self._session is None:
            raise RuntimeAdapterError("Session not initialized", provider=self.name)
        url = f"{self.endpoint}{path}"
        last_error: RuntimeAdapterError | None = None
        for attempt in range(self.retries + 1):
            emitted = False
            try:
                async with self._session.post(
                    url,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
                ) as resp:
                    if resp.status >= 400:
                        raise await self._http_error(resp, url)
                    async for chunk in self._iter_sse(resp):
                        emitted = True
                        yield chunk
                    return
            except RuntimeAdapterError as e:
                last_error = e
                if emitted or not self._is_retryable(e) or attempt >= self.retries:
                    raise
            except TimeoutError:
                last_error = RuntimeAdapterError(
                    f"Timeout calling {url} after {self.timeout_sec}s",
                    provider=self.name,
                    kind=RuntimeAdapterError.KIND_TIMEOUT,
                )
                if attempt >= self.retries:
                    raise last_error from None
            except (client_error, OSError) as e:
                last_error = RuntimeAdapterError(
                    f"Cannot reach {url}: {e}",
                    provider=self.name,
                    kind=RuntimeAdapterError.KIND_UNAVAILABLE,
                )
                if emitted or attempt >= self.retries:
                    raise last_error from None
            await self._sleep_before_retry(attempt, last_error)
        raise last_error or RuntimeAdapterError(
            "OpenAI-compatible stream failed", provider=self.name
        )

    async def _iter_sse(self, resp):
        """SSE framing: blank-line separated events, ``data:`` payload lines,
        ``:`` comment heartbeats, ``[DONE]`` terminator, idle watchdog."""
        import json as _json

        data_lines: list[str] = []
        buffered = 0
        async for raw_line in self._read_lines_with_idle_timeout(resp):
            line = raw_line.rstrip("\r\n")
            if line.startswith(":"):
                continue  # heartbeat comment
            if line == "":
                if not data_lines:
                    continue
                data = "\n".join(data_lines)
                data_lines = []
                if data.strip() == "[DONE]":
                    return
                try:
                    yield _json.loads(data)
                except _json.JSONDecodeError as exc:
                    raise RuntimeAdapterError(
                        f"Malformed SSE data chunk: {data[:120]!r}",
                        provider=self.name,
                        kind=RuntimeAdapterError.KIND_INVALID_RESPONSE,
                    ) from exc
                continue
            if line.startswith("data:"):
                piece = line[5:].lstrip()
                buffered += len(piece)
                if buffered > self._SSE_MAX_BUFFER:
                    raise RuntimeAdapterError(
                        "SSE buffer exceeded 10MB",
                        provider=self.name,
                        kind=RuntimeAdapterError.KIND_INVALID_RESPONSE,
                    )
                data_lines.append(piece)

    async def _read_lines_with_idle_timeout(self, resp):
        idle = self.stream_idle_timeout_sec
        while True:
            try:
                if idle:
                    line = await asyncio.wait_for(resp.content.readline(), timeout=idle)
                else:
                    line = await resp.content.readline()
            except TimeoutError as exc:
                raise RuntimeAdapterError(
                    f"SSE stream idle for {idle}s",
                    provider=self.name,
                    kind=RuntimeAdapterError.KIND_TIMEOUT,
                ) from exc
            if not line:
                return
            yield line.decode("utf-8", errors="replace")

    def _build_chat_request(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        inputs = payload.get("inputs") or {}
        constraints = payload.get("constraints") or {}
        messages = inputs.get("messages")
        if not messages:
            messages = []
            system = inputs.get("system")
            if system:
                messages.append({"role": "system", "content": system})
            prompt = inputs.get("prompt") or inputs.get("text") or ""
            # Multimodal: content_parts holds OpenAI content items
            # ({"type": "image_url", ...} / {"type": "video_url", ...});
            # the text prompt is appended as a final text part.
            parts = inputs.get("content_parts")
            if parts:
                user_content: Any = list(parts)
                if prompt:
                    user_content.append({"type": "text", "text": prompt})
            else:
                user_content = prompt
            messages.append({"role": "user", "content": user_content})
        body: dict[str, Any] = {"messages": messages}
        if self.model:
            body["model"] = self.model
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "seed",
            "repetition_penalty",
            "mm_processor_kwargs",
            "min_pixels",
            "max_pixels",
            # Tool use (PR-NA-030): strict JSON-schema tool definitions,
            # choice and parallel-call control pass straight through.
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            # Structured output / vendor parameters (e.g. Kimi reasoning_effort).
            "response_format",
            "reasoning_effort",
        ):
            if key in inputs:
                body[key] = inputs[key]
            elif key in constraints:
                body[key] = constraints[key]
        # Vendor-specific extra parameters merged verbatim (capability map is
        # owned by the caller's model profile, not hardcoded here).
        extra = inputs.get("vendor_parameters") or constraints.get("vendor_parameters")
        if isinstance(extra, dict):
            body.update(extra)
        return "/chat/completions", body

    def _build_embeddings_request(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        inputs = payload.get("inputs") or {}
        text_input = inputs.get("input") or inputs.get("texts") or inputs.get("text") or ""
        body: dict[str, Any] = {"input": text_input}
        if self.model:
            body["model"] = self.model
        if "dimensions" in inputs:
            body["dimensions"] = inputs["dimensions"]
        return "/embeddings", body

    @staticmethod
    def _parse_chat_response(raw: dict[str, Any]) -> dict[str, Any]:
        choices = raw.get("choices") or []
        if not choices:
            # An empty choices array is an invalid upstream response, never a
            # fabricated empty success (fail closed: invalid_response).
            raise RuntimeAdapterError("OpenAI-compatible server returned an empty choices array")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        finish_reason = choices[0].get("finish_reason")
        # Preserve the complete assistant message (including tool_calls and
        # provider reasoning fields) for protocol continuity in tool loops,
        # plus a normalized tool_call view. ``id``/request id is kept for
        # diagnosis and must be redacted before public traces.
        return {
            "result": content,
            "model": raw.get("model", ""),
            "finish_reason": finish_reason,
            "usage": raw.get("usage") or {},
            "message": message,
            "tool_calls": message.get("tool_calls") or [],
            "request_id": raw.get("id"),
        }

    @staticmethod
    def _parse_embeddings_response(raw: dict[str, Any]) -> dict[str, Any]:
        data = raw.get("data") or []
        vectors = [item.get("embedding") for item in data]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            raise RuntimeAdapterError("OpenAI-compatible server returned no embedding vectors")
        dimension = len(vectors[0])
        result: Any = vectors[0] if len(vectors) == 1 else vectors
        return {
            "result": result,
            "dimension": dimension,
            "count": len(vectors),
            "model": raw.get("model", ""),
            "usage": raw.get("usage") or {},
        }

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeAdapterError("Session not initialized", provider=self.name)

        # Some servers (vLLM without --served-model-name) only accept a
        # path-like model id; retry with model_fallback on model-not-found.
        model_chain = [m for m in (self.model, self.model_fallback) if m]
        if not self.model_fallback:
            model_chain = model_chain[:1]

        last_error: Exception | None = None
        for mi, model in enumerate(model_chain or [""]):
            if model:
                body = {**body, "model": model}
            try:
                return await self._post_with_retries(path, body)
            except RuntimeAdapterError as e:
                last_error = e
                is_model_not_found = getattr(e, "status", None) == 404
                if mi + 1 < len(model_chain) and is_model_not_found:
                    continue
                raise

        raise RuntimeAdapterError(
            f"OpenAI-compatible invoke failed: {last_error}",
            provider=self.name,
        )

    # ------------------------------------------------------------------
    # Error classification (picoclaw error_classifier pattern):
    # type layer → HTTP status → message keywords. Retry only what is
    # retryable; honor Retry-After (capped); exponential backoff + jitter.
    # ------------------------------------------------------------------
    RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
    MAX_RETRY_AFTER_SEC = 30.0

    async def _http_error(self, resp, url: str) -> RuntimeAdapterError:
        status = resp.status
        try:
            preview = await resp.text()
        except Exception:  # noqa: BLE001
            preview = ""
        retry_after = 0.0
        raw_ra = resp.headers.get("Retry-After") if resp.headers else None
        if raw_ra:
            try:
                retry_after = min(float(raw_ra), self.MAX_RETRY_AFTER_SEC)
            except ValueError:
                retry_after = 0.0
        if status == 429:
            kind = "rate_limited"
        elif status == 408:
            kind = RuntimeAdapterError.KIND_TIMEOUT
        elif status in (401, 403):
            kind = "auth_error"
        elif status >= 500:
            kind = RuntimeAdapterError.KIND_HTTP_ERROR
        else:
            kind = RuntimeAdapterError.KIND_INVALID_RESPONSE
        error = RuntimeAdapterError(
            f"HTTP {status} from {url}: {preview[:200]}",
            provider=self.name,
            kind=kind,
        )
        error.status = status  # type: ignore[attr-defined]
        error.retry_after = retry_after  # type: ignore[attr-defined]
        return error

    def _is_retryable(self, error: RuntimeAdapterError) -> bool:
        status = getattr(error, "status", None)
        if status is not None:
            return status in self.RETRYABLE_STATUS
        return error.kind in (
            RuntimeAdapterError.KIND_TIMEOUT,
            RuntimeAdapterError.KIND_UNAVAILABLE,
            "rate_limited",
        )

    async def _sleep_before_retry(self, attempt: int, error: RuntimeAdapterError | None) -> None:
        import random

        retry_after = getattr(error, "retry_after", 0.0) if error else 0.0
        backoff = min(0.5 * (2**attempt), self.MAX_RETRY_AFTER_SEC)
        jitter = backoff * 0.25 * random.random()
        await asyncio.sleep(max(retry_after, backoff + jitter))

    async def _post_with_retries(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        client_error = getattr(aiohttp, "ClientError", OSError)
        if not isinstance(client_error, type) or not issubclass(client_error, BaseException):
            client_error = OSError

        url = f"{self.endpoint}{path}"
        last_error: RuntimeAdapterError | None = None
        for attempt in range(self.retries + 1):
            try:
                async with self._session.post(
                    url,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
                ) as resp:
                    if resp.status >= 400:
                        raise await self._http_error(resp, url)
                    try:
                        return await resp.json()
                    except Exception:  # noqa: BLE001 - classified below
                        try:
                            preview = (await resp.text())[:200]
                        except Exception:  # noqa: BLE001
                            preview = ""
                        raise RuntimeAdapterError(
                            f"Non-JSON response from {url}: {preview[:120]}",
                            provider=self.name,
                            kind=RuntimeAdapterError.KIND_INVALID_RESPONSE,
                        ) from None
            except RuntimeAdapterError as e:
                last_error = e
                if not self._is_retryable(e) or attempt >= self.retries:
                    raise
            except TimeoutError:
                last_error = RuntimeAdapterError(
                    f"Timeout calling {url} after {self.timeout_sec}s",
                    provider=self.name,
                    kind=RuntimeAdapterError.KIND_TIMEOUT,
                )
                if attempt >= self.retries:
                    raise last_error from None
            except (client_error, OSError) as e:
                # Connector failures usually derive from OSError, while
                # disconnects and malformed HTTP responses are ClientError.
                last_error = RuntimeAdapterError(
                    f"Cannot reach {url}: {e}",
                    provider=self.name,
                    kind=RuntimeAdapterError.KIND_UNAVAILABLE,
                )
                if attempt >= self.retries:
                    raise last_error from None
            except Exception as e:  # noqa: BLE001 - last-resort, unclassified
                raise RuntimeAdapterError(
                    f"OpenAI-compatible invoke failed: {e}",
                    provider=self.name,
                ) from e
            await self._sleep_before_retry(attempt, last_error)

        raise last_error or RuntimeAdapterError(
            "OpenAI-compatible invoke failed",
            provider=self.name,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    async def health_detail(self) -> dict[str, Any]:
        """Probe ``GET {endpoint}/models`` and check the expected model id.

        Returns a dict with ``reachable`` and, when the manifest declared a
        model, ``expected_model_present``. Never raises; errors are reported
        in the returned dict.
        """
        if self._session is None:
            return {"reachable": False, "error": "runtime not started"}

        import aiohttp

        try:
            async with self._session.get(
                self.health_endpoint,
                timeout=aiohttp.ClientTimeout(total=min(self.timeout_sec, 30.0)),
            ) as resp:
                if resp.status >= 400:
                    return {"reachable": False, "error": f"HTTP {resp.status}"}
                body = await resp.json()
        except Exception as e:  # noqa: BLE001 - health must not raise
            # str(TimeoutError()) is "" — always emit a diagnosable class.
            return {"reachable": False, "error": str(e) or type(e).__name__}

        served = [m.get("id", "") for m in (body.get("data") or [])]
        detail: dict[str, Any] = {"reachable": True, "served_models": served}
        expected = self.model or self.model_fallback
        if expected:
            candidates = {self.model, self.model_fallback} - {""}
            detail["expected_model"] = expected
            # Exact match is authoritative; substring matching is reported
            # separately as a hint (it can false-positive, e.g. "qwen3"
            # matching served id "qwen3.5").
            detail["expected_model_present"] = any(s in candidates for s in served)
            if not detail["expected_model_present"]:
                detail["expected_model_present_fuzzy"] = any(
                    any(c and (c in s or s in c) for c in candidates) for s in served
                )
        return detail
