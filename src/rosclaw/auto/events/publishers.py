"""Auto event publishers — emit rosclaw-auto events to Event Bus."""
import logging
from typing import Any

try:
    from rosclaw.core.event_bus import Event, EventPriority
except ImportError:
    Event = dict
    EventPriority = None

from .schemas import (
    AutoProposalCreatedEvent,
    ChampionPromotedEvent,
    DeadEndRegisteredEvent,
    EventEnvelope,
)

logger = logging.getLogger("rosclaw.auto.events.publishers")


class AutoPublisher:
    """Publish rosclaw-auto events to the Event Bus."""

    def __init__(self, event_bus: Any | None = None):
        self._bus = event_bus

    def _publish(self, event: EventEnvelope) -> None:
        if self._bus is None:
            logger.debug("No event bus; event dropped locally: %s", event.event_type)
            return
        try:
            # Convert EventEnvelope to core Event
            if Event is not dict:
                core_event = Event(
                    topic=event.event_type,
                    payload=event.to_dict(),
                    source=event.source or "rosclaw-auto",
                    priority=EventPriority.NORMAL if EventPriority else None,
                    trace_id=event.trace_id,
                )
                self._bus.publish(core_event)
            else:
                self._bus.publish(event.to_dict())
            logger.debug("Published %s", event.event_type)
        except Exception as exc:
            logger.warning("Event publish failed: %s", exc)

    def proposal_created(self, proposal_id: str, task_id: str,
                         target_skill_id: str, hypothesis_statement: str) -> None:
        event = AutoProposalCreatedEvent(
            event_id=f"evt_prop_{proposal_id}",
            proposal_id=proposal_id,
            task_id=task_id,
            target_skill_id=target_skill_id,
            hypothesis_statement=hypothesis_statement,
        )
        self._publish(event)

    def champion_promoted(self, champion_id: str, skill_id: str,
                          task_id: str, level: str, metrics: dict) -> None:
        event = ChampionPromotedEvent(
            event_id=f"evt_champ_{champion_id}",
            champion_id=champion_id,
            skill_id=skill_id,
            task_id=task_id,
            level=level,
            metrics=metrics,
        )
        self._publish(event)

    def deadend_registered(self, deadend_id: str, task_id: str,
                           direction: str, rejection_reason: str) -> None:
        event = DeadEndRegisteredEvent(
            event_id=f"evt_de_{deadend_id}",
            deadend_id=deadend_id,
            task_id=task_id,
            direction=direction,
            rejection_reason=rejection_reason,
        )
        self._publish(event)
