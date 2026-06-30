"""Generic contact-event types shared across bodies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# Canonical event labels used by the generic event detector.
# Bodies may map their local labels to these.
CONTACT_EVENT_LABELS = [
    "no_contact",
    "weak_contact",
    "desired_contact",
    "over_contact",
    "emergency_contact",
    "early_contact",
    "late_contact",
    "self_collision",
    "position_stall",
    "motion_blocked",
    "hardware_protection",
    "temperature_limited",
    "force_sensor_drift",
    "unknown",
]

# Safety-priority order for selecting a mutually-exclusive primary event.
PRIMARY_EVENT_PRIORITY = [
    "hardware_protection",
    "motion_blocked",
    "emergency_contact",
    "over_contact",
    "position_stall",
    "temperature_limited",
    "early_contact",
    "self_collision",
    "desired_contact",
    "late_contact",
    "weak_contact",
    "force_sensor_drift",
    "no_contact",
    "unknown",
]


@dataclass
class ContactEvent:
    """A single inferred contact event."""

    event_type: str = "unknown"
    confidence: float = 1.0
    dofs: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContactEvent":
        return cls(**data)


def select_primary_event(tags: List[str]) -> str:
    """Pick the highest-priority tag from a list of labels."""
    tag_set = set(tags)
    for event in PRIMARY_EVENT_PRIORITY:
        if event in tag_set:
            return event
    return "unknown"


def event_distribution(events: List[ContactEvent]) -> Dict[str, int]:
    """Count events by type."""
    dist: Dict[str, int] = {}
    for ev in events:
        dist[ev.event_type] = dist.get(ev.event_type, 0) + 1
    return dist


def tag_distribution(events: List[ContactEvent], secondary: Optional[List[List[str]]] = None) -> Dict[str, int]:
    """Count all tags, optionally including secondary tag lists."""
    dist: Dict[str, int] = {}
    for ev in events:
        dist[ev.event_type] = dist.get(ev.event_type, 0) + 1
    if secondary:
        for tags in secondary:
            for tag in tags:
                dist[tag] = dist.get(tag, 0) + 1
    return dist
