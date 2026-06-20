"""Schema for the ``emergency_stop`` MCP tool."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class EmergencyStopResponse(TypedDict):
    """Envelope payload returned by ``emergency_stop``."""

    stopped: bool
    reason: str
    mode: str
    note: NotRequired[str]
