"""Base classes for Runtime Kernel producers and consumers."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent

logger = logging.getLogger("rosclaw.runtime.component")


class RuntimeComponent(LifecycleMixin):
    """Base class for all Runtime Kernel components.

    Components are lifecycle-aware and hold a reference to the RuntimeBus.
    """

    def __init__(self, name: str, runtime_bus: RuntimeBus) -> None:
        super().__init__()
        self.name = name
        self.bus = runtime_bus

    def publish_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        robot: str | None = None,
        body_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        """Helper to publish a RuntimeEvent with this component as source."""
        event = RuntimeEvent(
            source=self.name,
            robot=robot,
            body_id=body_id,
            type=event_type,
            payload=payload,
            metadata=metadata or {},
        )
        self.bus.publish(event)
        return event


class RuntimeProducer(RuntimeComponent):
    """A component that produces runtime events.

    Producers typically wrap a driver, sensor, or external MCP server.
    """

    @abstractmethod
    def _do_start(self) -> None:
        """Begin producing events."""

    @abstractmethod
    def _do_stop(self) -> None:
        """Stop producing events."""


class RuntimeConsumer(RuntimeComponent):
    """A component that consumes runtime events.

    Subclasses declare which event types/prefixes they subscribe to; the base
    class handles subscription during start and cleanup during stop.
    """

    def __init__(self, name: str, runtime_bus: RuntimeBus) -> None:
        super().__init__(name, runtime_bus)
        self._subscriptions: list[tuple[str, bool, Any]] = []

    def subscribe(self, event_type: str, callback: Any) -> None:
        """Register a subscription that will be cleaned up on stop."""
        self._subscriptions.append((event_type, False, callback))
        self.bus.subscribe(event_type, callback)

    def subscribe_prefix(self, prefix: str, callback: Any) -> None:
        """Register a prefix subscription that will be cleaned up on stop."""
        self._subscriptions.append((prefix, True, callback))
        self.bus.subscribe_prefix(prefix, callback)

    @abstractmethod
    def on_event(self, event: RuntimeEvent) -> None:
        """Default event handler for wildcard subscriptions."""

    def _do_start(self) -> None:
        """Subclasses can override but should call super()."""
        if not self._subscriptions:
            # Default: subscribe to everything and route to on_event.
            self.subscribe_prefix("*", self.on_event)

    def _do_stop(self) -> None:
        for event_type, is_prefix, callback in self._subscriptions:
            try:
                self.bus.unsubscribe(event_type, callback, is_prefix=is_prefix)
            except Exception as exc:
                logger.debug("Unsubscribe failed for %s: %s", self.name, exc)
        self._subscriptions.clear()
