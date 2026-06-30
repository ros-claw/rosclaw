"""Runtime Kernel service lifecycle manager."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rosclaw.core.event_bus import get_global_event_bus
from rosclaw.core.event_sink import JsonlEventSink
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.firstboot.workspace import resolve_home
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.registry import RuntimeComponentRegistry

logger = logging.getLogger("rosclaw.runtime.service")


class RuntimeKernelService(LifecycleMixin):
    """Top-level service that owns the RuntimeBus, registry, and event sink.

    Usage:
        service = RuntimeKernelService()
        service.initialize()
        service.start()
        # ... runtime is active ...
        service.stop()
    """

    def __init__(self, home: str | Path | None = None) -> None:
        super().__init__()
        self._home = resolve_home(str(home) if home else None)
        self._event_bus = get_global_event_bus()
        self._sink = JsonlEventSink(home=self._home)
        self._runtime_bus = RuntimeBus(event_bus=self._event_bus, event_sink=self._sink)
        self._registry = RuntimeComponentRegistry()

    @property
    def bus(self) -> RuntimeBus:
        return self._runtime_bus

    @property
    def registry(self) -> RuntimeComponentRegistry:
        return self._registry

    def register(self, name: str, component: Any) -> Any:
        """Register a runtime component and return it."""
        self._registry.register(name, component)
        return component

    def _do_initialize(self) -> None:
        self._sink.attach(self._event_bus)
        logger.info("RuntimeKernel initialized at %s", self._home)

    def _do_start(self) -> None:
        self._registry.initialize_all()
        self._registry.start_all()
        logger.info("RuntimeKernel started with components: %s", self._registry.names())

    def _do_stop(self) -> None:
        self._registry.stop_all()
        self._sink.close()
        logger.info("RuntimeKernel stopped")

    def stats(self) -> dict[str, Any]:
        return {
            "components": self._registry.stats(),
            "bus": self._runtime_bus.stats(),
        }
