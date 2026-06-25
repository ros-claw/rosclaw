"""CLI dispatch hook for command telemetry."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rosclaw.firstboot.workspace import resolve_home

from .telemetry_client import TelemetryClient


@contextmanager
def telemetry_command_hook(args: Any) -> Generator[None, None, None]:
    """Wrap CLI command dispatch and record a privacy-safe command_completed event.

    Telemetry failures are swallowed so they can never block a user command.
    """
    start = time.monotonic()
    error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            home = resolve_home(getattr(args, "workspace", None))
            command = getattr(args, "command", None)
            if home.exists() and command not in ("firstboot", "doctor"):
                duration_ms = int((time.monotonic() - start) * 1000)
                status = "failure" if error is not None else "success"
                client = TelemetryClient(home)
                client.record_command(args, status, duration_ms, error=error)
                # Fire-and-forget daily heartbeat if it is due.
                _run_heartbeat_async(client)
        except Exception:
            pass


def _run_heartbeat_async(client: TelemetryClient) -> None:
    """Trigger a daily heartbeat without blocking the CLI."""
    def _heartbeat() -> None:
        with __import__("contextlib").suppress(Exception):
            client.heartbeat_if_due()

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
