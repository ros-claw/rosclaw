"""PR-DF-05: Memory canonical write path — critic judgments become
evidence-backed MemoryItems through the WriteGate; experience_graph stays
as the compatibility projection (Phase A)."""

import json

from rosclaw.core.event_bus import Event
from rosclaw.core.runtime import Runtime, RuntimeConfig


def _runtime(store):
    cfg = RuntimeConfig(
        robot_id="sim_test",
        seekdb_backend="memory",
        enable_memory=True,
        enable_practice=False,
        enable_how=False,
        enable_auto=False,
        enable_knowledge=False,
        enable_skill_manager=False,
        enable_provider=False,
        enable_sense=False,
        enable_firewall=False,
        enable_event_persistence=False,
        enable_tracing=False,
    )
    rt = Runtime(cfg)
    from unittest.mock import patch

    with patch.object(Runtime, "_create_seekdb_client", return_value=store):
        rt.initialize()
    return rt


def _judgment(bus, episode_id="ep_1", status="SUCCESS", reward=0.9, reason=""):
    bus.publish(
        Event(
            topic="rosclaw.critic.judgment",
            payload={
                "episode_id": episode_id,
                "status": status,
                "reward": reward,
                "reason": reason,
                "context": {"outcome": {"skill_name": "pick_cup"}, "instruction": "pick the cup"},
            },
        )
    )


def test_critic_judgment_writes_memory_item():
    from rosclaw.memory.seekdb_client import InMemoryStructuredStore

    store = InMemoryStructuredStore()
    store.connect()
    rt = _runtime(store)
    assert rt._memory_gate is not None and rt._memory_repository is not None

    _judgment(rt.event_bus, episode_id="ep_ok", status="SUCCESS", reward=0.9)
    items = store.query("memory_items", {})
    assert len(items) == 1
    item = items[0]
    assert item["memory_type"] == "episodic"
    assert item["episode_id"] == "ep_ok"
    assert item["reward"] == 0.9
    # evidence-backed (§12): ACTIVE memory must carry evidence rows
    evd = store.query("memory_evidence", {"memory_id": item["id"]})
    assert evd, "memory_items row without evidence"
    # compat projection also ran
    assert store.count("experience_graph", {}) >= 1


def test_critic_failure_becomes_failure_memory_with_evidence():
    from rosclaw.memory.seekdb_client import InMemoryStructuredStore

    store = InMemoryStructuredStore()
    store.connect()
    rt = _runtime(store)
    _judgment(rt.event_bus, episode_id="ep_bad", status="FAILURE", reward=0.1, reason="slip")
    items = store.query("memory_items", {"memory_type": "failure"})
    assert len(items) == 1
    assert items[0]["failure_type"] == "slip"
    refs = items[0]["evidence_refs"]
    if isinstance(refs, str):
        refs = json.loads(refs)
    assert refs == ["critic_result:ep_bad"]


def test_critic_write_is_idempotent():
    """E2E-7 flavor: replaying the same judgment must not duplicate memory."""
    from rosclaw.memory.seekdb_client import InMemoryStructuredStore

    store = InMemoryStructuredStore()
    store.connect()
    rt = _runtime(store)
    _judgment(rt.event_bus, episode_id="ep_same", status="SUCCESS", reward=0.5)
    _judgment(rt.event_bus, episode_id="ep_same", status="SUCCESS", reward=0.5)
    assert store.count("memory_items", {}) == 1


def test_memory_items_absent_when_store_missing():
    """Data-plane failure → legacy projection only, loop unbroken (ADR-0009 §4)."""
    rt = _runtime(None)
    assert rt._memory_gate is None
    _judgment(rt.event_bus)  # must not raise
