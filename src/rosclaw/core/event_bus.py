"""
ROSClaw EventBus - Central Nervous System

All modules communicate exclusively through the EventBus using
publish/subscribe patterns. No direct module-to-module calls.

This implements the "Event Bus" principle from the ROSClaw architecture:
- Agent Runtime -> ROSClaw Runtime -> Physical World
- All communication flows through the bus
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


class EventPriority(Enum):
    """Priority levels for event processing."""
    CRITICAL = 0   # Emergency stop, safety violations
    HIGH = 1       # Control commands, trajectory execution
    NORMAL = 2     # Standard operational events
    LOW = 3        # Telemetry, logging
    BACKGROUND = 4 # Data export, analytics


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
        metadata: Additional context
    """
    topic: str
    payload: Any
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""  # correlation ID for distributed tracing across the pipeline
    metadata: dict = field(default_factory=dict)


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

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._async_subscribers: dict[str, list[Callable]] = {}
        self._event_history: list[Event] = []
        self._max_history = 10000
        self._lock = asyncio.Lock()
        self._running = False
        self._event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

    def subscribe(self, topic: str, callback: Callable[[Event], None]) -> None:
        """Subscribe to a topic with a sync callback."""
        if not callable(callback):
            raise TypeError(f"Handler must be callable, got {type(callback).__name__}")
        if not isinstance(topic, str):
            raise TypeError(f"Topic must be str, got {type(topic).__name__}")
        if not topic:
            raise ValueError("Topic cannot be empty")
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def subscribe_async(
        self, topic: str, callback: Callable[[Event], Coroutine]
    ) -> None:
        """Subscribe to a topic with an async callback."""
        if not callable(callback):
            raise TypeError(f"Handler must be callable, got {type(callback).__name__}")
        if not isinstance(topic, str):
            raise TypeError(f"Topic must be str, got {type(topic).__name__}")
        if not topic:
            raise ValueError("Topic cannot be empty")
        if topic not in self._async_subscribers:
            self._async_subscribers[topic] = []
        self._async_subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Unsubscribe a callback from a topic."""
        if topic in self._subscribers and callback in self._subscribers[topic]:
            self._subscribers[topic].remove(callback)
        if topic in self._async_subscribers and callback in self._async_subscribers[topic]:
            self._async_subscribers[topic].remove(callback)

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Sync subscribers are called immediately.
        Async subscribers are scheduled on the event loop.
        """
        # CRITICAL FIX: auto-inject trace_id for distributed tracing if missing
        if not event.trace_id:
            event.trace_id = f"trace_{uuid.uuid4().hex[:12]}"

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Call sync subscribers
        for callback in self._subscribers.get(event.topic, []):
            try:
                callback(event)
            except Exception as e:
                print(f"[EventBus] Error in sync subscriber for {event.topic}: {e}")

        # Schedule async subscribers
        for callback in self._async_subscribers.get(event.topic, []):
            try:
                asyncio.create_task(self._run_async_callback(callback, event))
            except Exception as e:
                print(f"[EventBus] Error scheduling async subscriber for {event.topic}: {e}")

    async def _run_async_callback(
        self, callback: Callable[[Event], Coroutine], event: Event
    ) -> None:
        """Run an async callback with error handling."""
        try:
            await callback(event)
        except Exception as e:
            print(f"[EventBus] Error in async subscriber for {event.topic}: {e}")

    async def publish_async(self, event: Event) -> None:
        """Async version of publish for use in async contexts."""
        self.publish(event)

    def get_history(self, topic: Optional[str] = None, limit: int = 100) -> list[Event]:
        """Get recent event history, optionally filtered by topic."""
        events = self._event_history
        if topic:
            events = [e for e in events if e.topic == topic]
        return events[-limit:]

    def clear_history(self, topic: Optional[str] = None) -> None:
        """Clear event history. If topic is given, clear only events for that topic."""
        if topic:
            self._event_history = [e for e in self._event_history if e.topic != topic]
        else:
            self._event_history.clear()

    @property
    def topics(self) -> list[str]:
        """List all topics with subscribers."""
        return list(set(list(self._subscribers.keys()) + list(self._async_subscribers.keys())))

    def subscriber_count(self, topic: str) -> int:
        """Count subscribers for a topic."""
        sync = len(self._subscribers.get(topic, []))
        async_ = len(self._async_subscribers.get(topic, []))
        return sync + async_

    async def await_event(
        self,
        topic: str,
        timeout: float = 30.0,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> Optional[Event]:
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
            if not future.done():
                if filter_fn is None or filter_fn(event):
                    future.set_result(event)

        self.subscribe(topic, handler)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.unsubscribe(topic, handler)

    def get_stats(self) -> dict:
        """Return event bus statistics."""
        return {
            "topics": self.topics,
            "total_subscribers": sum(
                len(self._subscribers.get(t, [])) + len(self._async_subscribers.get(t, []))
                for t in self.topics
            ),
            "history_size": len(self._event_history),
            "max_history": self._max_history,
        }
