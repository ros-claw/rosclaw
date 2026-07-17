"""Thin adapter for emergency-stop coordination through the Runtime/EventBus."""

from __future__ import annotations

from typing import Any


class SafetyClient:
    """Emergency client that delegates to Runtime's acknowledged stop path."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def emergency_stop(self, reason: str) -> dict[str, Any]:
        """Trigger an emergency stop and return its evidence receipt."""
        request = getattr(self._runtime, "request_emergency_stop", None)
        if not callable(request):
            raise RuntimeError("Runtime does not implement acknowledged emergency stop")
        receipt = request(
            reason,
            source="mcp.emergency_stop",
        )
        if isinstance(receipt, dict):
            return receipt
        if hasattr(receipt, "to_dict"):
            return receipt.to_dict()
        raise TypeError("Runtime returned an invalid EmergencyStopReceipt")
