"""Runtime hooks for body lifecycle events.

Default hooks only emit events and log; they do not start hardware. Runtime /
sense modules can subscribe to these events to reload state safely.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rosclaw.body.schema import EffectiveBody

logger = logging.getLogger("rosclaw.body.hooks")


class BodyHookEvent:
    """Canonical body lifecycle event topics."""

    BODY_ACTIVE_SWITCHED = "BODY_ACTIVE_SWITCHED"
    BODY_EFFECTIVE_CHANGED = "BODY_EFFECTIVE_CHANGED"
    BODY_PROVIDER_HEALTH_CHANGED = "BODY_PROVIDER_HEALTH_CHANGED"
    BODY_SKILL_COMPATIBILITY_CHANGED = "BODY_SKILL_COMPATIBILITY_CHANGED"


@dataclass
class BodyHookContext:
    """Context passed to body lifecycle hooks."""

    workspace: Any
    body_instance_id: str
    old_body: EffectiveBody | None = None
    new_body: EffectiveBody | None = None
    reason: str = ""
    strict: bool = False


HookCallback = Callable[[str, BodyHookContext], None]


class BodySwitchHooks:
    """Registry and dispatcher for body switch / change hooks.

    Default behavior: dispatch events, log, never fail the underlying operation
    unless ``strict=True`` is requested. Runtime and sense modules can register
    subscribers via ``subscribe()``.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[HookCallback]] = {}

    def subscribe(self, event_type: str, callback: HookCallback) -> None:
        """Register a callback for a body hook event."""
        self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: HookCallback) -> None:
        """Remove a previously registered callback."""
        callbacks = self._subscribers.get(event_type, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def dispatch(self, event_type: str, context: BodyHookContext) -> dict[str, Any]:
        """Dispatch an event to all subscribers.

        Non-strict mode: subscriber failures are logged but do not propagate.
        Strict mode: the first failure raises, causing the caller to fail.
        """
        results: dict[str, Any] = {"event_type": event_type, "failures": [], "success": True}
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(event_type, context)
            except Exception as exc:  # noqa: BLE001
                msg = f"Body hook {event_type} failed: {exc}"
                logger.warning(msg)
                results["failures"].append(msg)
                if context.strict:
                    results["success"] = False
                    raise
        results["success"] = len(results["failures"]) == 0
        return results

    def on_active_body_switched(
        self,
        workspace: Any,
        body_instance_id: str,
        old_body: EffectiveBody | None = None,
        new_body: EffectiveBody | None = None,
        strict: bool = False,
        reason: str = "",
    ) -> dict[str, Any]:
        context = BodyHookContext(
            workspace=workspace,
            body_instance_id=body_instance_id,
            old_body=old_body,
            new_body=new_body,
            reason=reason,
            strict=strict,
        )
        return self.dispatch(BodyHookEvent.BODY_ACTIVE_SWITCHED, context)

    def on_body_effective_changed(
        self,
        workspace: Any,
        body_instance_id: str,
        old_body: EffectiveBody | None,
        new_body: EffectiveBody,
        strict: bool = False,
        reason: str = "",
    ) -> dict[str, Any]:
        context = BodyHookContext(
            workspace=workspace,
            body_instance_id=body_instance_id,
            old_body=old_body,
            new_body=new_body,
            reason=reason,
            strict=strict,
        )
        return self.dispatch(BodyHookEvent.BODY_EFFECTIVE_CHANGED, context)

    def on_provider_health_changed(
        self,
        workspace: Any,
        body_instance_id: str,
        strict: bool = False,
        reason: str = "",
    ) -> dict[str, Any]:
        context = BodyHookContext(
            workspace=workspace,
            body_instance_id=body_instance_id,
            reason=reason,
            strict=strict,
        )
        return self.dispatch(BodyHookEvent.BODY_PROVIDER_HEALTH_CHANGED, context)

    def on_skill_compatibility_changed(
        self,
        workspace: Any,
        body_instance_id: str,
        strict: bool = False,
        reason: str = "",
    ) -> dict[str, Any]:
        context = BodyHookContext(
            workspace=workspace,
            body_instance_id=body_instance_id,
            reason=reason,
            strict=strict,
        )
        return self.dispatch(BodyHookEvent.BODY_SKILL_COMPATIBILITY_CHANGED, context)


# Global hook dispatcher instance. Tests and runtime can replace or subscribe.
_default_hooks = BodySwitchHooks()


def get_default_hooks() -> BodySwitchHooks:
    return _default_hooks
