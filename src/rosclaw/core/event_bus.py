"""
ROSClaw EventBus - Central Nervous System

All modules communicate exclusively through the EventBus using
publish/subscribe patterns. No direct module-to-module calls.

This implements the "Event Bus" principle from the ROSClaw architecture:
- Agent Runtime -> ROSClaw Runtime -> Physical World
- All communication flows through the bus
"""

import asyncio
import fnmatch
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("rosclaw.core.event_bus")

try:
    from rosclaw.core.event_topics import normalize_topic
except ImportError:

    def normalize_topic(topic: str) -> str:
        return topic


class EventPriority(Enum):
    """Priority levels for event processing."""

    CRITICAL = 0  # Emergency stop, safety violations
    HIGH = 1  # Control commands, trajectory execution
    NORMAL = 2  # Standard operational events
    LOW = 3  # Telemetry, logging
    BACKGROUND = 4  # Data export, analytics


@dataclass
class Event:
    """
    Standard event envelope for all ROSClaw communication.

    Attributes:
        topic: Event topic/channel (e.g., "robot.joint_states")
        payload: Event data (dict, numpy array, etc.)
        source: Module that published the event
        timestamp: Unix timestamp when event was created
        priority: Processing priority
        event_id: Unique identifier
        trace_id: End-to-end causal trace identifier
        span_id: Operation that emitted this event
        parent_span_id: Parent operation, when known
        metadata: Additional context
    """

    topic: str
    payload: Any
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""  # correlation ID for distributed tracing across the pipeline
    span_id: str = ""  # active operation that emitted this event
    parent_span_id: str = ""  # parent operation, when known
    metadata: dict = field(default_factory=dict)

    def derive(self, **overrides) -> "Event":
        """Create a new Event inheriting this event's trace_id.

        Args:
            **overrides: Fields to override (topic, payload, source, etc.)

        Returns:
            New Event with inherited trace_id.
        """
        return Event(
            topic=overrides.get("topic", self.topic),
            payload=overrides.get("payload", self.payload),
            source=overrides.get("source", self.source),
            timestamp=overrides.get("timestamp", time.time()),
            priority=overrides.get("priority", self.priority),
            event_id=overrides.get("event_id", str(uuid.uuid4())[:8]),
            trace_id=overrides.get("trace_id", self.trace_id),
            span_id=overrides.get("span_id", self.span_id),
            parent_span_id=overrides.get("parent_span_id", self.parent_span_id),
            metadata=overrides.get("metadata", self.metadata.copy()),
        )


# Global singleton EventBus instance for cross-process sharing
# (within the same Python process). CLI commands and Runtime share
# the same bus so published events are visible via list/tail.
_global_event_bus: Optional["EventBus"] = None
_global_bus_lock = threading.Lock()


def get_global_event_bus() -> "EventBus":
    """Return the singleton global EventBus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        with _global_bus_lock:
            if _global_event_bus is None:
                _global_event_bus = EventBus()
    return _global_event_bus


class EventBus:
    """
    Central publish/subscribe event bus for ROSClaw.

    All modules communicate through this bus. No direct calls.
    Supports both sync and async subscribers.

    Example:
        bus = EventBus()

        # Subscribe to joint state updates
        bus.subscribe("robot.joint_states", on_joint_state)

        # Publish an event
        bus.publish(Event(
            topic="robot.joint_states",
            payload={"positions": [0.1, 0.2, ...]},
            source="mcp_driver"
        ))
    """

    def __init__(self, normalize_topics: bool = True):
        self._subscribers: dict[str, list[Callable]] = {}
        self._async_subscribers: dict[str, list[Callable]] = {}
        self._max_history = 10000
        self._event_history: deque[Event] = deque(maxlen=self._max_history)
        self._lock = threading.Lock()  # protects _subscribers, _async_subscribers, _event_history
        self._async_lock = asyncio.Lock()
        self._running = False
        self._event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._normalize_topics = normalize_topics

    def _norm(self, topic: str) -> str:
        """Normalize topic if normalization is enabled."""
        if self._normalize_topics:
            return normalize_topic(topic)
        return topic

    def subscribe(self, topic: str, callback: Callable[[Event], None]) -> None:
        """Subscribe to a topic with a sync callback.

        Topic is normalized to the v1.0 standard namespace automatically.
        """
        if not callable(callback):
            raise TypeError(f"Handler must be callable, got {type(callback).__name__}")
        if not isinstance(topic, str):
            raise TypeError(f"Topic must be str, got {type(topic).__name__}")
        if not topic:
            raise ValueError("Topic cannot be empty")
        topic = self._norm(topic)
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(callback)

    def subscribe_async(self, topic: str, callback: Callable[[Event], Coroutine]) -> None:
        """Subscribe to a topic with an async callback.

        Topic is normalized to the v1.0 standard namespace automatically.
        """
        if not callable(callback):
            raise TypeError(f"Handler must be callable, got {type(callback).__name__}")
        if not isinstance(topic, str):
            raise TypeError(f"Topic must be str, got {type(topic).__name__}")
        if not topic:
            raise ValueError("Topic cannot be empty")
        topic = self._norm(topic)
        with self._lock:
            if topic not in self._async_subscribers:
                self._async_subscribers[topic] = []
            self._async_subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Unsubscribe a callback from a topic.

        Topic is normalized before lookup.
        """
        topic = self._norm(topic)
        with self._lock:
            if topic in self._subscribers and callback in self._subscribers[topic]:
                self._subscribers[topic].remove(callback)
            if topic in self._async_subscribers and callback in self._async_subscribers[topic]:
                self._async_subscribers[topic].remove(callback)

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if a topic matches a subscription pattern.

        Supports exact match, MQTT-style ``#`` (match all), and
        glob-style ``*`` wildcards.
        """
        if pattern == topic:
            return True
        if pattern == "#":
            return True
        if "*" in pattern or "?" in pattern:
            return fnmatch.fnmatch(topic, pattern)
        return False

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Sync subscribers are called immediately.
        Async subscribers are scheduled on the event loop.
        Topic matching uses the v1.0 standard namespace internally,
        but the event.topic is preserved as-is for backward compatibility.
        """
        # Use normalized topic for subscriber matching, but preserve original
        # event.topic so existing tests and consumers are not broken.
        norm_topic = self._norm(event.topic)

        # Auto-inject causal trace/span identity. ``contextvars`` propagates
        # through sync and async call paths; legacy request IDs remain the
        # fallback for publishers that have not adopted structured spans yet.
        trace_context = None
        try:
            from rosclaw.observability.context import current_trace_context

            trace_context = current_trace_context()
        except ImportError:
            pass
        if not event.trace_id:
            payload = event.payload if isinstance(event.payload, dict) else {}
            metadata = event.metadata if isinstance(event.metadata, dict) else {}
            event.trace_id = (
                (trace_context.trace_id if trace_context else "")
                or payload.get("trace_id")
                or payload.get("request_id")
                or payload.get("correlation_id")  # noqa: W503
                or payload.get("episode_id")  # noqa: W503
                or metadata.get("trace_id")  # noqa: W503
                or metadata.get("request_id")  # noqa: W503
                or metadata.get("correlation_id")  # noqa: W503
                or metadata.get("episode_id")  # noqa: W503
                or f"trace_{uuid.uuid4().hex[:12]}"  # noqa: W503
            )
        if trace_context is not None and event.trace_id == trace_context.trace_id:
            if not event.span_id:
                event.span_id = trace_context.span_id
            if not event.parent_span_id:
                event.parent_span_id = trace_context.parent_span_id or ""

        # Store in history
        with self._lock:
            self._event_history.append(event)
            while len(self._event_history) > self._max_history:
                self._event_history.popleft()

            # Snapshot subscribers under lock to avoid races during iteration
            subscribers_snapshot = {k: list(v) for k, v in self._subscribers.items()}
            async_subscribers_snapshot = {k: list(v) for k, v in self._async_subscribers.items()}

        # Call sync subscribers (exact + wildcard match)
        for topic_pattern, callbacks in subscribers_snapshot.items():
            if not self._topic_matches(topic_pattern, norm_topic):
                continue
            for callback in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"Error in sync subscriber for {event.topic}: {e}")

        # Schedule async subscribers (exact + wildcard match)
        for topic_pattern, callbacks in async_subscribers_snapshot.items():
            if not self._topic_matches(topic_pattern, norm_topic):
                continue
            for callback in callbacks:
                try:
                    asyncio.create_task(self._run_async_callback(callback, event))
                except Exception as e:
                    logger.warning(f"Error scheduling async subscriber for {event.topic}: {e}")

    async def _run_async_callback(
        self, callback: Callable[[Event], Coroutine], event: Event
    ) -> None:
        """Run an async callback with error handling."""
        try:
            await callback(event)
        except Exception as e:
            logger.warning(f"Error in async subscriber for {event.topic}: {e}")

    async def publish_async(self, event: Event) -> None:
        """Async version of publish for use in async contexts."""
        self.publish(event)

    def get_history(self, topic: str | None = None, limit: int = 100) -> list[Event]:
        """Get recent event history, optionally filtered by topic."""
        if limit <= 0:
            return []
        with self._lock:
            events = list(self._event_history)
        if topic:
            events = [e for e in events if e.topic == topic]
        return events[-limit:]

    def clear_history(self, topic: str | None = None) -> None:
        """Clear event history. If topic is given, clear only events for that topic."""
        with self._lock:
            if topic:
                kept = [e for e in self._event_history if e.topic != topic]
                self._event_history.clear()
                self._event_history.extend(kept)
            else:
                self._event_history.clear()

    @property
    def topics(self) -> list[str]:
        """List all topics with subscribers."""
        with self._lock:
            return list(set(list(self._subscribers.keys()) + list(self._async_subscribers.keys())))

    def subscriber_count(self, topic: str) -> int:
        """Count subscribers for a topic."""
        with self._lock:
            sync = len(self._subscribers.get(topic, []))
            async_ = len(self._async_subscribers.get(topic, []))
        return sync + async_

    async def await_event(
        self,
        topic: str,
        timeout: float = 30.0,
        filter_fn: Callable[[Event], bool] | None = None,
    ) -> Event | None:
        """
        Wait for an event on a topic with optional filtering.

        Creates a temporary one-shot subscription, waits for the
        first matching event, then unsubscribes.

        Args:
            topic: Event topic to wait for
            timeout: Maximum seconds to wait
            filter_fn: Optional predicate to filter events

        Returns:
            Matching Event, or None if timeout
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def handler(event: Event) -> None:
            if not future.done() and (filter_fn is None or filter_fn(event)):
                future.set_result(event)

        self.subscribe(topic, handler)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            return None
        finally:
            self.unsubscribe(topic, handler)

    def get_stats(self) -> dict:
        """Return event bus statistics."""
        with self._lock:
            topics = list(
                set(list(self._subscribers.keys()) + list(self._async_subscribers.keys()))
            )
            history_size = len(self._event_history)
            total_subscribers = sum(
                len(self._subscribers.get(t, [])) + len(self._async_subscribers.get(t, []))
                for t in topics
            )
        return {
            "topics": topics,
            "total_subscribers": total_subscribers,
            "history_size": history_size,
            "max_history": self._max_history,
        }
