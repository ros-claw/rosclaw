"""Schema for the ``sandbox_run`` MCP tool."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class SandboxRunResponse(TypedDict):
    """Envelope payload returned by ``sandbox_run``."""

    physics_state: dict[str, Any]
    mode: str
    note: NotRequired[str]
