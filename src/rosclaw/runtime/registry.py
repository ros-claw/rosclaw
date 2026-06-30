"""Runtime component registry and lifecycle orchestration."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from rosclaw.runtime.component import RuntimeComponent

logger = logging.getLogger("rosclaw.runtime.registry")


class RuntimeComponentRegistry:
    """Registry of runtime producers/consumers with ordered lifecycle."""

    def __init__(self) -> None:
        self._components: dict[str, RuntimeComponent] = {}
        self._order: list[str] = []

    def register(self, name: str, component: RuntimeComponent) -> RuntimeComponent:
        """Register a component. If name already exists, replace it."""
        if name in self._components:
            self._order.remove(name)
        self._components[name] = component
        self._order.append(name)
        return component

    def get(self, name: str) -> RuntimeComponent | None:
        return self._components.get(name)

    def __iter__(self) -> Iterator[tuple[str, RuntimeComponent]]:
        for name in self._order:
            yield name, self._components[name]

    def names(self) -> list[str]:
        return list(self._order)

    def initialize_all(self) -> None:
        for name, component in self:
            try:
                component.initialize()
            except Exception as exc:
                logger.error("Failed to initialize %s: %s", name, exc)
                raise

    def start_all(self) -> None:
        for name, component in self:
            try:
                component.start()
            except Exception as exc:
                logger.error("Failed to start %s: %s", name, exc)
                raise

    def stop_all(self) -> None:
        # Stop in reverse order for clean dependency teardown.
        for name in reversed(self._order):
            component = self._components[name]
            try:
                component.stop()
            except Exception as exc:
                logger.error("Failed to stop %s: %s", name, exc)

    def stats(self) -> dict[str, Any]:
        return {
            name: {
                "state": component.state.name,
                "error": component.error_message,
            }
            for name, component in self
        }
