"""Streaming + classified-retry tests for OpenAICompatRuntime (PR-NA-030b).

Mock aiohttp server: SSE chunk framing, tool_calls delta sequence, usage in
final chunk, [DONE], malformed SSE, 429 with Retry-After, 5xx retry, 400
no-retry, idle stream timeout.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web

from rosclaw.provider.core.errors import RuntimeAdapterError
from rosclaw.provider.runtimes.openai_compat_runtime import OpenAICompatRuntime

pytestmark = pytest.mark.asyncio


def _sse_chunk(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _delta(content: str | None = None, tool_calls=None, finish=None, usage=None):
    chunk: dict = {
        "id": "chatcmpl-x",
        "model": "mock",
        "choices": [
            {
                "index": 0,
                "delta": ({} if content is None and not tool_calls else {
                    **({"content": content} if content is not None else {}),
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                }),
                "finish_reason": finish,
            }
        ],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


async def _start_server(handler) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    app.router.add_get("/v1/models", lambda r: web.json_response({"data": [{"id": "mock"}]}))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1"


async def _runtime(url: str, **kwargs) -> OpenAICompatRuntime:
    rt = OpenAICompatRuntime("test", url, model="mock", **kwargs)
    await rt.start()
    return rt


class TestStreaming:
    async def test_text_stream_and_usage(self) -> None:
        captured: dict = {}

        async def handler(request: web.Request) -> web.StreamResponse:
            captured["body"] = await request.json()
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await resp.prepare(request)
            await resp.write(_sse_chunk(_delta(content="你")).encode())
            await resp.write(b": heartbeat\n\n")
            await resp.write(_sse_chunk(_delta(content="好")).encode())
            await resp.write(
                _sse_chunk(
                    _delta(
                        finish="stop",
                        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                    )
                ).encode()
            )
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp

        runner, url = await _start_server(handler)
        try:
            rt = await _runtime(url)
            chunks = [c async for c in rt.invoke_stream({"inputs": {"prompt": "hi"}})]
            await rt.stop()
        finally:
            await runner.cleanup()
        assert captured["body"]["stream"] is True
        assert captured["body"]["stream_options"] == {"include_usage": True}
        texts = [
            c["choices"][0]["delta"].get("content")
            for c in chunks
            if c.get("choices") and c["choices"][0]["delta"].get("content")
        ]
        assert texts == ["你", "好"]
        usages = [c["usage"] for c in chunks if c.get("usage")]
        assert usages[-1]["total_tokens"] == 7

    async def test_tool_call_deltas_streamed(self) -> None:
        async def handler(request: web.Request) -> web.StreamResponse:
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await resp.prepare(request)
            await resp.write(
                _sse_chunk(
                    _delta(
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "ping", "arguments": "{\"ec"},
                            }
                        ]
                    )
                ).encode()
            )
            await resp.write(
                _sse_chunk(
                    _delta(
                        tool_calls=[
                            {"index": 0, "function": {"arguments": "ho\":true}"}}
                        ],
                        finish="tool_calls",
                    )
                ).encode()
            )
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp

        runner, url = await _start_server(handler)
        try:
            rt = await _runtime(url)
            chunks = [c async for c in rt.invoke_stream({"inputs": {"prompt": "ping"}})]
            await rt.stop()
        finally:
            await runner.cleanup()
        deltas = [
            c["choices"][0]["delta"]["tool_calls"]
            for c in chunks
            if c.get("choices") and c["choices"][0]["delta"].get("tool_calls")
        ]
        assert len(deltas) == 2
        assert deltas[0][0]["function"]["name"] == "ping"

    async def test_malformed_sse_fails_closed(self) -> None:
        async def handler(request: web.Request) -> web.StreamResponse:
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await resp.prepare(request)
            await resp.write(b"data: {not json\n\n")
            await resp.write_eof()
            return resp

        runner, url = await _start_server(handler)
        try:
            rt = await _runtime(url, retries=0)
            with pytest.raises(RuntimeAdapterError, match="Malformed SSE"):
                async for _ in rt.invoke_stream({"inputs": {"prompt": "x"}}):
                    pass
            await rt.stop()
        finally:
            await runner.cleanup()

    async def test_idle_stream_timeout(self) -> None:
        async def handler(request: web.Request) -> web.StreamResponse:
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await resp.prepare(request)
            await resp.write(_sse_chunk(_delta(content="partial")).encode())
            await asyncio.sleep(2.0)  # stall; watchdog is 0.2s
            await resp.write_eof()
            return resp

        runner, url = await _start_server(handler)
        try:
            rt = await _runtime(url, retries=0, stream_idle_timeout_sec=0.2)
            with pytest.raises(RuntimeAdapterError, match="idle"):
                async for _ in rt.invoke_stream({"inputs": {"prompt": "x"}}):
                    pass
            await rt.stop()
        finally:
            await runner.cleanup()


class TestClassifiedRetry:
    async def test_429_retry_with_retry_after(self) -> None:
        attempts = 0

        async def handler(request: web.Request) -> web.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return web.json_response(
                    {"error": "rate limited"}, status=429, headers={"Retry-After": "0"}
                )
            return web.json_response(
                {
                    "id": "x",
                    "model": "mock",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {},
                }
            )

        runner, url = await _start_server(handler)
        try:
            rt = await _runtime(url, retries=2)
            result = await rt.invoke({"inputs": {"prompt": "hi"}})
            await rt.stop()
        finally:
            await runner.cleanup()
        assert result["result"] == "ok"
        assert attempts == 2

    async def test_400_not_retried(self) -> None:
        attempts = 0

        async def handler(request: web.Request) -> web.Response:
            nonlocal attempts
            attempts += 1
            return web.json_response({"error": "bad request"}, status=400)

        runner, url = await _start_server(handler)
        try:
            rt = await _runtime(url, retries=3)
            with pytest.raises(RuntimeAdapterError, match="HTTP 400"):
                await rt.invoke({"inputs": {"prompt": "hi"}})
            await rt.stop()
        finally:
            await runner.cleanup()
        assert attempts == 1  # no blind retry on client errors

    async def test_503_retried(self) -> None:
        attempts = 0

        async def handler(request: web.Request) -> web.Response:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                return web.json_response({"error": "unavailable"}, status=503)
            return web.json_response(
                {
                    "id": "x",
                    "model": "mock",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {},
                }
            )

        runner, url = await _start_server(handler)
        try:
            rt = await _runtime(url, retries=2)
            result = await rt.invoke({"inputs": {"prompt": "hi"}})
            await rt.stop()
        finally:
            await runner.cleanup()
        assert result["result"] == "ok"
        assert attempts == 2

    async def test_429_kind_is_rate_limited(self) -> None:
        async def handler(request: web.Request) -> web.Response:
            return web.json_response({"error": "slow down"}, status=429)

        runner, url = await _start_server(handler)
        try:
            rt = await _runtime(url, retries=0)
            with pytest.raises(RuntimeAdapterError) as exc_info:
                await rt.invoke({"inputs": {"prompt": "hi"}})
            await rt.stop()
        finally:
            await runner.cleanup()
        assert exc_info.value.kind == "rate_limited"
