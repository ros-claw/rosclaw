"""Runtime skill plugin registry.

Skills register handlers with the ``@runtime_handler(skill_name)`` decorator.
The ``SkillExecutor`` queries this registry before falling back to the legacy
``SkillEntry.handler`` path.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("rosclaw.runtime.plugin")

HandlerCallable = Callable[[dict[str, Any]], Any]


class RuntimeSkillPlugin:
    """Global registry mapping skill names to runtime handler callables."""

    def __init__(self) -> None:
        self._handlers: dict[str, HandlerCallable] = {}

    def register(self, skill_name: str, handler: HandlerCallable) -> HandlerCallable:
        """Register a handler for ``skill_name``.

        Returns the handler so this method can be used as a decorator.
        """
        self._handlers[skill_name] = handler
        logger.debug("Registered runtime handler for skill: %s", skill_name)
        return handler

    def get_handler(self, skill_name: str) -> HandlerCallable | None:
        """Return the registered handler for ``skill_name`` or None."""
        return self._handlers.get(skill_name)

    def list_handlers(self) -> list[str]:
        """Return all skill names with a registered runtime handler."""
        return list(self._handlers.keys())

    def discover_handlers(self, entry_point_group: str = "rosclaw.runtime_handlers") -> None:
        """Load handlers from package entry points.

        Each entry point should be a module that imports and therefore
        registers its decorated handlers at import time.
        """
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):
                group = eps.select(group=entry_point_group)
            else:
                group = eps.get(entry_point_group, [])
        except Exception as exc:
            logger.warning("Failed to discover runtime handlers: %s", exc)
            return

        for ep in group:
            try:
                ep.load()
                logger.debug("Loaded runtime handler module: %s", ep.value)
            except Exception as exc:
                logger.warning("Failed to load runtime handler module %s: %s", ep.value, exc)


# Global plugin instance used by decorators and the executor.
_plugin = RuntimeSkillPlugin()


def get_runtime_plugin() -> RuntimeSkillPlugin:
    """Return the global runtime skill plugin registry."""
    return _plugin


def runtime_handler(skill_name: str) -> Callable[[HandlerCallable], HandlerCallable]:
    """Decorator to register a function as the runtime handler for a skill.

    Example:
        @runtime_handler("realsense_capture_rgbd")
        def handle_realsense(params):
            return {"status": "success", "frames": []}
    """

    def decorator(func: HandlerCallable) -> HandlerCallable:
        return get_runtime_plugin().register(skill_name, func)

    return decorator
