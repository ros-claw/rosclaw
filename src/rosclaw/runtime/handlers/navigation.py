"""Runtime handlers for navigation skills (stub)."""

from __future__ import annotations

from typing import Any

from rosclaw.runtime.plugin import runtime_handler


@runtime_handler("navigate_to")
def _handle_navigate_to(params: dict[str, Any]) -> dict[str, Any]:
    """Runtime handler for navigation skills."""
    return {
        "status": "success",
        "skill": "navigate_to",
        "target": params.get("target", "unknown"),
        "source": "runtime_handler",
    }
