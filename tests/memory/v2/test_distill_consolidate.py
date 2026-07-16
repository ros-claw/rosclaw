"""Tests for the Memory 2.0 distillation pipeline and consolidation (PR-MEM-1)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore
from rosclaw.memory.v2.consolidate import MemoryConsolidator
from rosclaw.memory.v2.distill import (
    SessionContext,
    build_candidates,
    distill_events,
    distill_session_dir,
)
from rosclaw.memory.v2.gate import MemoryWriteGate
from rosclaw.memory.v2.models import MemoryStatus
from rosclaw.memory.v2.repository import MemoryRepository


@pytest.fixture
def repo() -> MemoryRepository:
    client = InMemoryKnowledgeStore()
    client.connect()
    return MemoryRepository(client)


@pytest.fixture
def gate(repo: MemoryRepository) -> MemoryWriteGate:
    return MemoryWriteGate(repo)


def _context() -> SessionContext:
    return SessionContext(
        practice_id="prac_test",
        session_id="sess_test",
        episode_id="ep_test",
        robot_id="rh56_rps_robot",
        body_id="rh56_right",
        task_id="rh56_rps",
        skill_id="rh56_rps",
    )


def _events() -> list[dict]:
    base = {"practice_id": "prac_test", "session_id": "sess_test", "episode_id": "ep_test"}
    events = [
        {**base, "event_id": "evt_start", "event_type": "runtime.start", "payload": {}},
        {
            **base,
            "event_id": "evt_g1",
            "event_type": "rps.gesture.executed",
            "payload": {
                "hand": "right",
                "gesture_name": "scissors",
                "command_success": True,
                "verified": True,
                "telemetry_summary": {"temperature_max": 36},
            },
        },
        {
            **base,
            "event_id": "evt_g2",
            "event_type": "rps.gesture.executed",
            "payload": {
                "hand": "right",
                "gesture_name": "scissors",
                "command_success": True,
                "verified": False,
                "failure_reason": "joint_not_reached",
                "telemetry_summary": {"current_peak": 500, "temperature_max": 41},
            },
        },
        {
            **base,
            "event_id": "evt_g3",
            "event_type": "rps.gesture.executed",
            "payload": {
                "hand": "right",
                "gesture_name": "scissors",
                "command_success": True,
                "verified": True,
                "telemetry_summary": {"temperature_max": 38},
            },
        },
        {
            **base,
            "event_id": "evt_h1",
            "event_type": "health_check",
            "payload": {
                "runtime_s": 60.0,
                "right": {"summary": {"temperature_max": 36}, "error": {}},
                "left": {"summary": {"temperature_max": 35}, "error": {}},
            },
        },
        {
            **base,
            "event_id": "evt_h2",
            "event_type": "health_check",
            "payload": {
                "runtime_s": 3600.0,
                "right": {"summary": {"temperature_max": 42}, "error": {}},
                "left": {"summary": {"temperature_max": 35}, "error": {}},
            },
        },
        {
            **base,
            "event_id": "evt_stop",
            "event_type": "practice.session_stopped",
            "payload": {"outcome": "SUCCESS"},
        },
    ]
    return events


def test_build_candidates_covers_types(repo: MemoryRepository, gate: MemoryWriteGate) -> None:
    candidates = build_candidates(_context(), _events())
    types = {c.memory_type for c in candidates}
    assert "episodic" in types
    assert "failure" in types  # the unverified scissors
    assert "body" in types  # thermal drift 36→42
    assert "skill" in types  # scissors 2/3


def test_distill_idempotent_on_rerun(repo: MemoryRepository, gate: MemoryWriteGate) -> None:
    events = _events()
    first = distill_events(_context(), events, gate=gate, repository=repo)
    count_after_first = repo.count()
    assert first.candidates > 0
    second = distill_events(_context(), events, gate=gate, repository=repo)
    # Rerun: no new stores (dedup via content hash → IGNORE or UPDATE-same).
    assert repo.count() == count_after_first
    assert not second.stored or set(second.stored) <= set(first.stored)


def test_distilled_memories_all_traceable(repo: MemoryRepository, gate: MemoryWriteGate) -> None:
    distill_events(_context(), _events(), gate=gate, repository=repo)
    items = repo.query({"robot_id": "rh56_rps_robot"}, limit=100)
    assert items
    for item in items:
        trace = repo.trace(item.memory_id)
        assert trace["traceable"], f"memory {item.memory_id} has no evidence"


def test_distill_real_7x24_session(repo: MemoryRepository, gate: MemoryWriteGate) -> None:
    """Distill the final 7x24 session from disk (real data, not a mock)."""
    session_dir = Path(
        "/home/nvidia/.rosclaw/practice/runs/rh56_rps/sessions/prac_20260716T174229Z_600fb1"
    )
    if not session_dir.exists():
        pytest.skip("7x24 session data not available")
    result = distill_session_dir(session_dir, gate=gate, repository=repo)
    assert result.candidates > 0
    assert result.stored
    items = repo.query({"robot_id": "rh56_rps_robot"}, limit=100)
    types = {item.memory_type for item in items}
    assert "episodic" in types
    assert "skill" in types
    for item in items:
        assert repo.trace(item.memory_id)["traceable"]


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def test_consolidator_expires_ttl(repo: MemoryRepository) -> None:
    from rosclaw.memory.v2.models import MemoryItem

    old = MemoryItem(
        memory_type="episodic",
        robot_id="r1",
        title="old",
        document="old doc",
        evidence_refs=["e1"],
        expires_at=time.time() - 1,
    )
    repo.store(old)
    result = MemoryConsolidator(repo).consolidate()
    assert result.expired == 1
    assert repo.get(old.memory_id).status == MemoryStatus.EXPIRED.value  # type: ignore[union-attr]


def test_consolidator_supersedes_duplicates(repo: MemoryRepository) -> None:
    from rosclaw.memory.v2.models import MemoryItem

    older = MemoryItem(
        memory_type="body",
        robot_id="r1",
        title="same",
        document="same doc",
        evidence_refs=["e1"],
        event_time=100.0,
    )
    newer = MemoryItem(
        memory_type="body",
        robot_id="r1",
        title="same",
        document="same doc",
        evidence_refs=["e2"],
        event_time=200.0,
    )
    repo.store(older)
    # store() dedups by content_hash, so write the second directly.
    repo._client.insert("memory_items", newer.to_record())
    result = MemoryConsolidator(repo).consolidate()
    assert result.superseded == 1
    assert repo.get(older.memory_id).status == MemoryStatus.SUPERSEDED.value  # type: ignore[union-attr]
    assert repo.get(newer.memory_id).status == MemoryStatus.ACTIVE.value  # type: ignore[union-attr]


def test_consolidator_decay_skips_pinned(repo: MemoryRepository) -> None:
    from rosclaw.memory.v2.models import MemoryItem

    ancient = MemoryItem(
        memory_type="body",
        robot_id="r1",
        title="ancient",
        document="doc",
        evidence_refs=["e1"],
        event_time=time.time() - 400 * 86400,
        importance=0.9,
    )
    repo.store(ancient)
    repo.pin(ancient.memory_id)
    result = MemoryConsolidator(repo, decay_half_life_days=30).consolidate()
    assert result.decayed == 0
    assert result.pinned_kept == 1
    assert repo.get(ancient.memory_id).importance == 0.9  # type: ignore[union-attr]


def test_consolidator_decays_unpinned(repo: MemoryRepository) -> None:
    from rosclaw.memory.v2.models import MemoryItem

    ancient = MemoryItem(
        memory_type="body",
        robot_id="r1",
        title="ancient",
        document="doc",
        evidence_refs=["e1"],
        event_time=time.time() - 400 * 86400,
        importance=0.9,
    )
    repo.store(ancient)
    result = MemoryConsolidator(repo, decay_half_life_days=30).consolidate()
    assert result.decayed == 1
    assert repo.get(ancient.memory_id).importance < 0.9  # type: ignore[union-attr]
