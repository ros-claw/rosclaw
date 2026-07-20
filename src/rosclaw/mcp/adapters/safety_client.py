"""Thin emergency adapter that can only call the rosclawd client."""

from __future__ import annotations

from typing import Any


class SafetyClient:
    """Emergency client that never owns a Runtime or hardware driver."""

    def __init__(self, daemon: Any) -> None:
        self._daemon = daemon

    def emergency_stop(self, reason: str) -> dict[str, Any]:
        """Trigger an emergency stop and return its evidence receipt."""
        request = getattr(self._daemon, "emergency_stop", None)
        if not callable(request):
            raise RuntimeError("rosclawd client does not implement acknowledged emergency stop")
        receipt = request(
            reason,
            source="mcp.emergency_stop",
        )
        if isinstance(receipt, dict):
            return receipt
        if hasattr(receipt, "to_dict"):
            return receipt.to_dict()
        raise TypeError("rosclawd returned an invalid EmergencyStopReceipt")
