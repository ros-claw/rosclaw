"""Connection-drop recovery for OpenAI-compatible streaming."""

import json

import pytest
from aiohttp import web

from rosclaw.provider.runtimes.openai_compat_runtime import OpenAICompatRuntime


@pytest.mark.asyncio
async def test_disconnect_before_first_event_is_retried():
    attempts = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            request.transport.close()
            return web.Response()
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        chunk = {
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "recovered"},
                    "finish_reason": "stop",
                }
            ],
        }
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    runtime = OpenAICompatRuntime(
        "test",
        f"http://127.0.0.1:{port}/v1",
        model="mock",
        retries=1,
    )
    await runtime.start()
    try:
        chunks = [chunk async for chunk in runtime.invoke_stream({"inputs": {"prompt": "hi"}})]
    finally:
        await runtime.stop()
        await runner.cleanup()

    assert attempts == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "recovered"
