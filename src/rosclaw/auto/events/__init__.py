"""rosclaw.auto.events — Event Bus integration layer."""
from .publishers import AutoPublisher
from .schemas import BenchmarkCompletedEvent, EventEnvelope, PraxisFailedEvent
from .subscribers import AutoSubscriber

__all__ = [
    "EventEnvelope",
    "PraxisFailedEvent",
    "BenchmarkCompletedEvent",
    "AutoSubscriber",
    "AutoPublisher",
]
