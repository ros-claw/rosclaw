"""rosclaw.auto.events — Event Bus integration layer."""
from .schemas import EventEnvelope, PraxisFailedEvent, BenchmarkCompletedEvent
from .subscribers import AutoSubscriber
from .publishers import AutoPublisher

__all__ = [
    "EventEnvelope",
    "PraxisFailedEvent",
    "BenchmarkCompletedEvent",
    "AutoSubscriber",
    "AutoPublisher",
]
