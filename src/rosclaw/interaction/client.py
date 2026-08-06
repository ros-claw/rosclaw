"""Small facade over a ROSClaw-compatible guarded-action backend."""

from __future__ import annotations

from typing import Any


class InteractionClient:
    """Adapt RuntimeClient or a robot MCP gateway to the interaction coordinator."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def prepare_action(self, **kwargs: Any) -> Any:
        return self._backend.prepare_operator_action(**kwargs)

    async def confirm_action(self, prepared: Any, **kwargs: Any) -> dict[str, Any]:
        result = await self._backend.confirm_operator_action(prepared, **kwargs)
        if not isinstance(result, dict):
            raise RuntimeError("ROSClaw returned a non-object interaction result")
        return result

    async def defer_action(self, prepared: Any, **kwargs: Any) -> dict[str, Any]:
        """Create a pending proposal without granting the Agent decision authority."""

        defer = getattr(self._backend, "defer_operator_action", None)
        if not callable(defer):
            raise RuntimeError("ROSClaw backend has no Operator Broker proposal adapter")
        result = await defer(prepared, **kwargs)
        if not isinstance(result, dict):
            raise RuntimeError("ROSClaw returned a non-object pending proposal result")
        return result

    async def cancel_action(self, action_id: str) -> None:
        cancel = getattr(self._backend, "cancel_action", None)
        if callable(cancel):
            await cancel(action_id)
