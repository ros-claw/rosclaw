"""Runtime handlers for navigation skills (stub)."""

from __future__ import annotations

from typing import Any

from rosclaw.runtime.plugin import runtime_handler


@runtime_handler("navigate_to")
def _handle_navigate_to(params: dict[str, Any]) -> dict[str, Any]:
    """Fail closed: this stub has no navigation backend or arrival evidence."""
    return {
        "status": "blocked",
        "reason": "NAVIGATION_BACKEND_UNAVAILABLE",
        "message": (
            "No navigation backend is implemented in this handler; no command was dispatched. "
            "Physical navigation requires rosclawd admission and a verified daemon-side "
            "executor; this stub does not provide or configure that route."
        ),
        "skill": "navigate_to",
        "target": params.get("target", "unknown"),
        "source": "runtime_handler",
    }
