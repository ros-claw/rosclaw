"""Runtime handlers for provider/reasoner skills."""

from __future__ import annotations

from typing import Any

from rosclaw.runtime.plugin import runtime_handler


@runtime_handler("provider_invoke")
def _handle_provider_invoke(params: dict[str, Any]) -> dict[str, Any]:
    """Runtime handler for direct provider invocation skills."""
    return {
        "status": "success",
        "skill": "provider_invoke",
        "provider": params.get("provider", "cosmos"),
        "capability": params.get("capability", "vlm.risk_assessment"),
        "source": "runtime_handler",
    }
