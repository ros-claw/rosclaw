"""Utilities for bridging sync and async code safely.

ROSClaw exposes many synchronous entry points (EventBus handlers, CLI
commands, sync analysis helpers) that need to invoke async coroutines.
Using ``asyncio.run`` directly breaks when the caller is already inside
an event loop. The helpers below handle both cases.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

# Shared executor for running coroutines when the current thread already has
# an event loop. asyncio.run cannot be called on such threads, so we offload
# to a worker thread that owns its own fresh event loop.
_SYNC_RUN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="rosclaw-sync-run-",
)


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine synchronously, safe from any calling context.

    Unlike ``asyncio.run``, this works when called from inside an already-
    running event loop by executing the coroutine in a background thread
    that owns its own event loop.

    Warning: when called from inside an event loop this blocks the calling
    thread (and therefore that loop) until the coroutine finishes. Use
    ``fire_and_forget`` for event handlers and other fire-and-forget work.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop in current thread — safe to use asyncio.run.
        return asyncio.run(coro)

    # Inside an event loop. asyncio.run would raise RuntimeError here.
    # Schedule the coroutine on a background thread with its own loop.
    context = contextvars.copy_context()
    return _SYNC_RUN_EXECUTOR.submit(context.run, asyncio.run, coro).result()


def fire_and_forget(coro: Coroutine[Any, Any, Any]) -> None:
    """Schedule a coroutine without waiting for its result.

    Uses ``loop.create_task`` when already inside an event loop, otherwise
    runs the coroutine in a short-lived daemon thread. This is suitable for
    EventBus handlers and other callbacks that must not block the caller.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — run in a daemon thread so the caller is not blocked.
        context = contextvars.copy_context()
        threading.Thread(target=lambda: context.run(asyncio.run, coro), daemon=True).start()
    else:
        # Already inside an event loop — schedule as a task.
        loop.create_task(coro)
