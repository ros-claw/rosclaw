"""Tests for the Runtime Replay engine (Milestone 9)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rosclaw.core.event_bus import EventBus
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent
from rosclaw.runtime.replay import RuntimeReplay


def _make_bus() -> RuntimeBus:
    return RuntimeBus(event_bus=EventBus())


def _event(
    event_id: str,
    event_type: str,
    payload: dict,
    trace_id: str = "trace_001",
    timestamp: datetime | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        id=event_id,
        timestamp=timestamp or datetime.now(tz=timezone.utc),
        source="test",
        robot="test_bot",
        body_id="test_body",
        type=event_type,
        payload=payload,
        metadata={"trace_id": trace_id},
    )


def test_replay_event_by_id() -> None:
    bus = _make_bus()
    replay = RuntimeReplay(bus)
    event = _event("ev_1", "skill.invoke", {"skill_id": "pick"})
    bus.publish(event)

    found = replay.replay_event("ev_1")
    assert found is not None
    assert found.id == "ev_1"


def test_replay_episode_by_trace_id() -> None:
    bus = _make_bus()
    replay = RuntimeReplay(bus)
    bus.publish(_event("ev_1", "skill.invoke", {"skill_id": "pick"}, trace_id="ep_1"))
    bus.publish(_event("ev_2", "skill.complete", {"skill_id": "pick"}, trace_id="ep_1"))
    bus.publish(_event("ev_3", "skill.invoke", {"skill_id": "place"}, trace_id="ep_2"))

    events = replay.replay_episode("ep_1")
    assert len(events) == 2
    assert {e.id for e in events} == {"ev_1", "ev_2"}


def test_replay_skill_filters_by_skill_id() -> None:
    bus = _make_bus()
    replay = RuntimeReplay(bus)
    bus.publish(_event("ev_1", "skill.invoke", {"skill_id": "pick"}, trace_id="ep_1"))
    bus.publish(_event("ev_2", "skill.complete", {"skill_id": "pick"}, trace_id="ep_1"))
    bus.publish(_event("ev_3", "skill.invoke", {"skill_id": "place"}, trace_id="ep_1"))

    events = replay.replay_skill("pick", episode_id="ep_1")
    assert len(events) == 2
    assert all(e.payload.get("skill_id") == "pick" for e in events)


def test_replay_provider_filters_by_request_id() -> None:
    bus = _make_bus()
    replay = RuntimeReplay(bus)
    bus.publish(_event("ev_1", "provider.request", {"request_id": "req_1"}, trace_id="ep_1"))
    bus.publish(_event("ev_2", "provider.result", {"request_id": "req_1"}, trace_id="ep_1"))
    bus.publish(_event("ev_3", "provider.result", {"request_id": "req_2"}, trace_id="ep_1"))

    events = replay.replay_provider(request_id="req_1", episode_id="ep_1")
    assert len(events) == 2
    assert all(e.payload.get("request_id") == "req_1" for e in events)


def test_replay_time_range() -> None:
    bus = _make_bus()
    replay = RuntimeReplay(bus)
    now = datetime.now(tz=timezone.utc)
    bus.publish(_event("ev_1", "camera.frame", {}, timestamp=now - timedelta(seconds=10)))
    bus.publish(_event("ev_2", "camera.frame", {}, timestamp=now))
    bus.publish(_event("ev_3", "camera.frame", {}, timestamp=now + timedelta(seconds=10)))

    events = replay.replay_time_range(now - timedelta(seconds=5), now + timedelta(seconds=5), event_type="camera.frame")
    assert len(events) == 1
    assert events[0].id == "ev_2"


def test_replay_event_not_found_returns_none() -> None:
    bus = _make_bus()
    replay = RuntimeReplay(bus)
    assert replay.replay_event("missing") is None
