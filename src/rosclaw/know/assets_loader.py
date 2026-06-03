"""AssetsLoader — keep ``data/knowledge_assets/`` fresh.

Two responsibilities:

1. **Boot-time load**: at runtime start, copy any v1.5 assets that
   were dropped into the assets dir into the running
   ``KnowledgeInterface`` (in-memory).  This is the path that picks
   up a freshly-published asset bundle on the next process restart.

2. **On-demand reload**: when ``rosclaw.knowledge.assets_refreshed`` is
   published (by :class:`KnowledgeBatchEngine`, by CI, or by a CLI
   command), invalidate the task-pack cache and reload bridge_index.

This module deliberately does NOT install a filesystem watcher.  The
runtime stays event-driven; if you want to reload after dropping a
new YAML, publish the event manually:

    runtime.event_bus.publish(Event(
        topic="rosclaw.knowledge.assets_refreshed",
        payload={"source": "manual"},
        source="cli",
    ))
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rosclaw.core.event_bus import Event
from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.know.assets_loader")


class AssetsLoader(LifecycleMixin):
    """Bridges asset publication events to in-memory cache invalidation."""

    def __init__(
        self,
        runtime: Any,
        assets_path: str | Path = "data/knowledge_assets",
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.assets_path = Path(assets_path)
        self._reload_count = 0

    def _do_initialize(self) -> None:
        self.assets_path.mkdir(parents=True, exist_ok=True)
        logger.info("[AssetsLoader] watching %s", self.assets_path)
        # First-time load (idempotent — KnowledgeInterface already
        # does this in its own _do_initialize).
        self._reload(reason="boot")

    def _do_start(self) -> None:
        bus = getattr(self.runtime, "event_bus", None)
        if bus is None:
            return
        bus.subscribe("rosclaw.knowledge.assets_refreshed", self._on_refresh)

    def _on_refresh(self, event: Event) -> None:
        try:
            self._reload(reason=f"event:{event.source}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AssetsLoader] reload failed: %s", exc)

    def _reload(self, *, reason: str) -> None:
        knowledge = getattr(self.runtime, "_knowledge", None)
        bridge_path = self.assets_path / "bridge_index.json"
        if knowledge is not None and bridge_path.exists():
            try:
                knowledge._load_bridge_index(bridge_path)
                self._reload_count += 1
                logger.info(
                    "[AssetsLoader] reloaded bridge_index (#%d, reason=%s)",
                    self._reload_count, reason,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[AssetsLoader] bridge_index reload failed: %s", exc
                )

        # Invalidate the task-pack adapter cache if it's loaded.
        try:
            from rosclaw.know.task_pack_adapter import reload_assets
            reload_assets()
        except ImportError:
            pass

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "assets_path": str(self.assets_path),
            "reload_count": self._reload_count,
        }
