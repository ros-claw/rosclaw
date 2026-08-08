from __future__ import annotations

import pytest

from rosclaw.core.event_bus import EventBus
from rosclaw.knowledge.event_adapter import KnowledgeEventAdapter, safe_event_payload


def test_events_publish_only_redacted_ids():
    bus = EventBus()
    adapter = KnowledgeEventAdapter(bus)
    adapter.publish(
        "know.reference_pack.created",
        {"reference_pack_id": "pack_1", "index_version": "idx_1", "document": "drop me"},
    )
    event = bus.get_history("know.reference_pack.created")[0]
    assert dict(event.payload) == {"reference_pack_id": "pack_1", "index_version": "idx_1"}


def test_sensitive_event_fields_fail_closed():
    with pytest.raises(ValueError, match="forbidden"):
        safe_event_payload({"reference_pack_id": "pack", "api_key": "secret"})
