"""Runtime handlers for manipulation skills (stub)."""

from __future__ import annotations

from typing import Any

from rosclaw.runtime.plugin import runtime_handler


@runtime_handler("pick")
def _handle_pick(params: dict[str, Any]) -> dict[str, Any]:
    """Runtime handler for pick manipulation skills."""
    return {
        "status": "success",
        "skill": "pick",
        "object": params.get("object", "unknown"),
        "source": "runtime_handler",
    }


@runtime_handler("place")
def _handle_place(params: dict[str, Any]) -> dict[str, Any]:
    """Runtime handler for place manipulation skills."""
    return {
        "status": "success",
        "skill": "place",
        "location": params.get("location", "unknown"),
        "source": "runtime_handler",
    }
