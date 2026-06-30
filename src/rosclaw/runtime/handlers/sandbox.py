"""Runtime handlers for sandbox/simulation skills."""

from __future__ import annotations

from typing import Any

from rosclaw.runtime.plugin import runtime_handler


@runtime_handler("sandbox_run")
def _handle_sandbox_run(params: dict[str, Any]) -> dict[str, Any]:
    """Runtime handler for MuJoCo sandbox execution skills."""
    return {
        "status": "success",
        "skill": "sandbox_run",
        "simulation": params.get("simulation", "default"),
        "source": "runtime_handler",
    }
