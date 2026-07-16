"""Schema for the ``emergency_stop`` MCP tool."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class EmergencyStopResponse(TypedDict):
    """Envelope payload returned by ``emergency_stop``."""

    stopped: bool
    reason: str
    mode: str
    request_id: NotRequired[str | None]
    targets: NotRequired[list[str]]
    request_dispatched: NotRequired[bool]
    driver_acknowledged: NotRequired[bool]
    physical_stop_observed: NotRequired[bool]
    final_status: NotRequired[str]
    note: NotRequired[str]
