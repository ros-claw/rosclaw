"""PR-DF-23 (phase-II §9): lineage graph payload + viewer for the dashboard."""

from __future__ import annotations

import json
import time

from rosclaw.dashboard.lineage_view import (
    LINEAGE_PAGE_HTML,
    _node_detail,
    build_lineage_payload,
)
from rosclaw.memory.seekdb_client import InMemoryStructuredStore
from rosclaw.storage.lineage import LineageRepository


def _store():
    s = InMemoryStructuredStore()
    s.connect()
    return s


def _seed_graph(store):
    repo = LineageRepository(store)
    repo.link("champion", "champ_1", "promoted_from", "evaluation", "eval_1")
    repo.link("evaluation", "eval_1", "evaluated_from", "experiment", "exp_1")
    repo.link("experiment", "exp_1", "derived_from", "patch", "patch_1")
    repo.link("patch", "patch_1", "patched_from", "proposal", "prop_1")
    repo.link("proposal", "prop_1", "proposed_from", "memory_insight", "ins_1")
    repo.link("memory_insight", "ins_1", "derived_from", "memory", "mem_1")
    repo.link("memory", "mem_1", "derived_from", "episode", "ep_1")
    repo.link("episode", "ep_1", "supported_by", "receipt", "rcpt_1")
    store.insert(
        "memory_items",
        {
            "id": "mem_1",
            "title": "force overshoot recovery",
            "outcome": "SUCCESS",
            "body_id": "rh56_right",
            "skill_id": "slide",
            "quality_score": 0.9,
            "evidence_refs": json.dumps(["episode:ep_1"]),
            "artifact_refs": json.dumps(["file:///tmp/s.mcap"]),
        },
    )
    store.insert(
        "episodes",
        {"id": "ep_1", "outcome": "SUCCESS", "task_id": "slide", "robot_id": "rh56",
         "artifact_uri": "file:///tmp/events.jsonl"},
    )
    store.insert(
        "execution_receipts",
        {
            "id": "rcpt_1", "action_id": "act_1", "execution_mode": "REAL",
            "final_state": "SUCCEEDED", "evidence_level": "task_verified",
            "evidence_domain": "hardware", "body_id": "rh56_right",
            "capability_id": "rh56.single_step",
        },
    )
    store.insert(
        "evolution_records",
        {
            "id": "champions:champ_1", "namespace": "champions", "key": "champ_1",
            "data": json.dumps({
                "id": "champ_1", "level": "sandbox", "evaluation_id": "eval_1",
                "metrics": {"success_rate": 0.9},
                "validation_summary": {"promotion_verified": True},
            }),
            "updated_at": time.time(),
        },
    )
    store.insert(
        "evolution_records",
        {
            "id": "deadends:de_1", "namespace": "deadends", "key": "de_1",
            "data": json.dumps({
                "id": "de_1", "direction": "raise force",
                "rejection_reason": "collisions above 400",
            }),
            "updated_at": time.time(),
        },
    )


def test_payload_has_graph_and_details():
    store = _store()
    _seed_graph(store)
    payload = build_lineage_payload(store, "champion", "champ_1")
    node_ids = {n["id"] for n in payload["nodes"]}
    assert node_ids == {"champ_1", "eval_1", "exp_1", "patch_1", "prop_1", "ins_1", "mem_1", "ep_1", "rcpt_1"}

    memory = payload["details"]["memory:mem_1"]
    assert memory["body_id"] == "rh56_right"
    assert memory["evidence_refs"] == ["episode:ep_1"]
    assert memory["artifact_refs"] == ["file:///tmp/s.mcap"]

    receipt = payload["details"]["receipt:rcpt_1"]
    assert receipt["execution_mode"] == "REAL"
    assert receipt["evidence_domain"] == "hardware"

    champion = payload["details"]["champion:champ_1"]
    assert champion["why_promoted"]["promotion_verified"] is True
    assert champion["why_promoted"]["evaluation_id"] == "eval_1"
    assert champion["score"] == {"success_rate": 0.9}

    episode = payload["details"]["episode:ep_1"]
    assert episode["artifact_uri"] == "file:///tmp/events.jsonl"


def test_dead_end_why_rejected():
    store = _store()
    _seed_graph(store)
    detail = _node_detail(store, "dead_end", "de_1")
    assert detail["why_rejected"] == "collisions above 400"
    assert detail["direction"] == "raise force"


def test_missing_entity_is_empty_graph_not_error():
    store = _store()
    payload = build_lineage_payload(store, "champion", "champ_nope")
    assert payload["nodes"] == [{"type": "champion", "id": "champ_nope", "depth": 0}]
    assert payload["edges"] == []


def test_page_html_renders_viewer():
    assert "/api/lineage/" in LINEAGE_PAGE_HTML
    assert "Trace" in LINEAGE_PAGE_HTML
    assert "<svg" in LINEAGE_PAGE_HTML
