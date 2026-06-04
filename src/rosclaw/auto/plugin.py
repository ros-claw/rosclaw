"""AutoPlugin — Runtime plugin entry point for rosclaw-auto."""
import logging
from typing import Any

from rosclaw.core.lifecycle import LifecycleMixin

from .config import AutoConfig
from .engine.auto_engine import AutoEngine
from .events.subscribers import AutoSubscriber
from .events.publishers import AutoPublisher

logger = logging.getLogger("rosclaw.auto.plugin")


class AutoPlugin(LifecycleMixin):
    """Runtime plugin for rosclaw-auto.

    Integrates with rosclaw-runtime lifecycle:
    - setup: load config, connect store, register event handlers
    - start: subscribe to event bus
    - stop: unsubscribe and flush
    - health: report status
    """

    name = "rosclaw-auto"
    version = "1.0.0"

    def __init__(self, config: dict | None = None, event_bus: Any | None = None,
                 seekdb_client: Any | None = None, skill_registry: Any | None = None):
        super().__init__()
        self.config = AutoConfig.from_dict(config or {})
        self.engine: AutoEngine | None = None
        self.subscriber: AutoSubscriber | None = None
        self.publisher: AutoPublisher | None = None
        self._event_bus = event_bus
        self._seekdb_client = seekdb_client
        self._skill_registry = skill_registry

    def _do_initialize(self) -> None:
        """Initialize engine and connect to runtime context."""
        self.engine = AutoEngine(
            config=self.config,
            event_bus=self._event_bus,
            seekdb_client=self._seekdb_client,
            skill_registry=self._skill_registry,
        )
        self.publisher = AutoPublisher(event_bus=self._event_bus)
        self.subscriber = AutoSubscriber(engine=self.engine, event_bus=self._event_bus)
        logger.info("AutoPlugin: initialized")

    def _do_start(self) -> None:
        if self.subscriber:
            self.subscriber.subscribe_all()
        logger.info("AutoPlugin: started")

    def _do_stop(self) -> None:
        if self.subscriber:
            self.subscriber.unsubscribe_all()
        logger.info("AutoPlugin: stopped")

    def health(self) -> dict:
        return {
            "plugin": self.name,
            "version": self.version,
            "engine_ready": self.engine is not None,
            "subscriber_ready": self.subscriber is not None,
            "publisher_ready": self.publisher is not None,
            "status": "healthy" if self.engine else "not_ready",
        }

    # Backward-compatible aliases for tests and external callers
    def setup(self, ctx: Any | None = None) -> None:
        """Alias for initialize() for backward compatibility."""
        self.initialize()

    def bind_event_bus(self, event_bus: Any) -> None:
        """Bind to runtime event bus (backward-compatible alias)."""
        self._event_bus = event_bus
        if self.publisher:
            self.publisher._bus = event_bus
        if self.subscriber:
            self.subscriber._bus = event_bus
