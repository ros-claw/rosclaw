"""Tests for the Memory 2.0 write gate (PR-MEM-1 §5.5, §5.8)."""

from __future__ import annotations

import pytest

from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore
from rosclaw.memory.v2.gate import MemoryWriteGate
from rosclaw.memory.v2.models import GateDecision, MemoryItem
from rosclaw.memory.v2.repository import MemoryRepository


@pytest.fixture
def repo() -> MemoryRepository:
    client = InMemoryKnowledgeStore()
    client.connect()
    return MemoryRepository(client)


@pytest.fixture
def gate(repo: MemoryRepository) -> MemoryWriteGate:
    return MemoryWriteGate(repo)


def _candidate(**overrides) -> MemoryItem:
    base = {
        "memory_type": "episodic",
        "robot_id": "rh56_rps_robot",
        "title": "something happened",
        "document": "a complete description",
        "evidence_refs": ["evt_1"],
    }
    base.update(overrides)
    return MemoryItem(**base)


def test_heartbeat_and_healthcheck_are_ignored(gate: MemoryWriteGate) -> None:
    for noise in ("heartbeat", "health_check", "imu_event", "frame_event", "rps.telemetry"):
        candidate = _candidate(metadata={"source_event_type": noise})
        assert gate.evaluate(candidate).decision == GateDecision.IGNORE.value, noise


def test_empty_candidate_ignored(gate: MemoryWriteGate) -> None:
    assert gate.evaluate(_candidate(title="", document="")).decision == GateDecision.IGNORE.value


@pytest.mark.parametrize(
    "secret",
    [
        "api_key: ghp_abcdefghij0123456789",
        "token=sk-abcdefghij0123456789abc",
        "password: hunter2",
        "ak_0123456789abcdefghijAB",
    ],
)
def test_secrets_are_rejected(gate: MemoryWriteGate, secret: str) -> None:
    candidate = _candidate(document=f"failed with config {secret}")
    assert gate.evaluate(candidate).decision == GateDecision.IGNORE.value


def test_failure_memory_without_evidence_is_quarantined(gate: MemoryWriteGate) -> None:
    candidate = _candidate(memory_type="failure", evidence_refs=[])
    assert gate.evaluate(candidate).decision == GateDecision.QUARANTINE.value


def test_safety_content_without_evidence_is_quarantined(gate: MemoryWriteGate) -> None:
    candidate = _candidate(document="finger overcurrent 过流 detected", evidence_refs=[])
    assert gate.evaluate(candidate).decision == GateDecision.QUARANTINE.value


def test_llm_inference_without_evidence_ignored(gate: MemoryWriteGate) -> None:
    candidate = _candidate(memory_type="body", evidence_refs=[])
    assert gate.evaluate(candidate).decision == GateDecision.IGNORE.value


def test_full_cot_stripped_by_default(gate: MemoryWriteGate) -> None:
    candidate = _candidate(metadata={"cot_trace": ["think", "think"], "note": "keep"})
    decision = gate.evaluate(candidate)
    assert decision.decision == GateDecision.STORE.value
    assert "cot_trace" in decision.redacted_fields
    assert "cot_trace" not in candidate.metadata
    assert candidate.metadata["note"] == "keep"


def test_full_cot_enabled_goes_to_quarantine(repo: MemoryRepository) -> None:
    gate = MemoryWriteGate(repo, store_full_cot=True)
    candidate = _candidate(metadata={"cot_trace": ["x"]})
    assert gate.evaluate(candidate).decision == GateDecision.QUARANTINE.value


def test_exact_duplicate_with_new_evidence_is_update(
    gate: MemoryWriteGate, repo: MemoryRepository
) -> None:
    original = _candidate(evidence_refs=["evt_1"])
    repo.store(original)
    newer = _candidate(evidence_refs=["evt_1", "evt_2"])  # same content, new evidence
    decision = gate.evaluate(newer)
    assert decision.decision == GateDecision.UPDATE.value
    assert decision.target_memory_id == original.memory_id


def test_exact_duplicate_without_new_evidence_is_ignore(
    gate: MemoryWriteGate, repo: MemoryRepository
) -> None:
    original = _candidate(event_time=200.0)
    repo.store(original)
    # Even a "newer" candidate is ignored when it brings no new evidence:
    # re-distilling the same session must be a pure no-op.
    rerun = _candidate(event_time=300.0)
    decision = gate.evaluate(rerun)
    assert decision.decision == GateDecision.IGNORE.value


def test_near_duplicate_is_merge(gate: MemoryWriteGate, repo: MemoryRepository) -> None:
    existing = _candidate(title="rh56 scissors gesture failed at high temperature")
    repo.store(existing)
    candidate = _candidate(title="rh56 scissors gesture failed at high temperature!")
    decision = gate.evaluate(candidate)
    assert decision.decision == GateDecision.MERGE.value
    assert decision.target_memory_id == existing.memory_id


def test_novel_candidate_is_store(gate: MemoryWriteGate) -> None:
    decision = gate.evaluate(_candidate(title="a brand new observation"))
    assert decision.decision == GateDecision.STORE.value
