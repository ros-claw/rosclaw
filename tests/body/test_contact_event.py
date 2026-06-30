"""Tests for generic contact-event types."""
from __future__ import annotations

from rosclaw.body.contact_event import (
    ContactEvent,
    event_distribution,
    select_primary_event,
    tag_distribution,
)


def test_select_primary_event_safety_priority():
    assert select_primary_event(["desired_contact", "over_contact"]) == "over_contact"
    assert select_primary_event(["hardware_protection", "over_contact"]) == "hardware_protection"
    assert select_primary_event([]) == "unknown"


def test_event_distribution():
    events = [
        ContactEvent(event_type="desired_contact"),
        ContactEvent(event_type="desired_contact"),
        ContactEvent(event_type="no_contact"),
    ]
    assert event_distribution(events) == {"desired_contact": 2, "no_contact": 1}


def test_tag_distribution_includes_secondary():
    events = [ContactEvent(event_type="desired_contact")]
    secondary = [["stable", "low_temp"]]
    dist = tag_distribution(events, secondary=secondary)
    assert dist["desired_contact"] == 1
    assert dist["stable"] == 1
    assert dist["low_temp"] == 1


def test_contact_event_roundtrip():
    ev = ContactEvent(event_type="over_contact", dofs=["thumb"], confidence=0.9)
    restored = ContactEvent.from_dict(ev.to_dict())
    assert restored.event_type == "over_contact"
    assert restored.dofs == ["thumb"]
    assert restored.confidence == 0.9
