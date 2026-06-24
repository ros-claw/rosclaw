"""CLI dispatch hook for command telemetry."""

from __future__ import annotations

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
            if home.exists():
                duration_ms = int((time.monotonic() - start) * 1000)
                status = "failure" if error is not None else "success"
                TelemetryClient(home).record_command(args, status, duration_ms, error=error)
        except Exception:
            pass
