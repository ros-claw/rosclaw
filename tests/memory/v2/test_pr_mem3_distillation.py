"""PR-MEM-3 unit tests (数据库优化v3 §13):

* RPS implicit failure (round.resolved invalid) -> Failure Memory
* episode quality PARTIAL_SUCCESS with verified-rate distribution
* body memory is an observation (causal_status), never a thermal limit
* empty body cognition -> no candidate (gate IGNORE path)
* bilingual document builder ZH/EN/CANONICAL/ALIASES
* exact-entity extraction incl. ZH aliases, thumb vs thumb_rot,
  bare left/right ambiguity
* metadata flattening round-trip (new top-level fields)
"""

from __future__ import annotations

import json

from rosclaw.memory.v2.adapters import adapter_for
from rosclaw.memory.v2.adapters.rh56_rps import Rh56RpsAdapter
from rosclaw.memory.v2.distill import (
    SessionContext,
    build_candidates,
    extract_failure_memories,
)
from rosclaw.memory.v2.document import (
    MultilingualMemoryDocumentBuilder,
    extract_exact_terms,
)
from rosclaw.memory.v2.models import MemoryItem


def _ctx() -> SessionContext:
    return SessionContext(
        practice_id="prac_test",
        session_id="sess_test",
        episode_id="ep_test",
        robot_id="rh56_rps_robot",
        body_id="rh56_rps_robot",
        task_id="rh56_rps",
        skill_id="rh56_rps",
    )


def _round_event(
    round_id: str, result: str, reason: str | None = None, choice: str = "rock"
) -> dict:
    return {
        "event_type": "rps.stress.round.resolved",
        "event_id": f"evt_{round_id}",
        "timestamp_ns": 1_000_000_000,
        "payload": {
            "round": {
                "round_id": round_id,
                "result": result,
                "robot_choice": choice,
                "robot_gesture_failure_reason": reason,
            }
        },
    }


def _gesture_event(
    round_id: str, hand: str, gesture: str, verified: bool, reason: str | None = None
) -> dict:
    return {
        "event_type": "rps.gesture.executed",
        "event_id": f"evt_g_{round_id}_{hand}_{gesture}",
        "timestamp_ns": 1_000_000_000,
        "payload": {
            "round_id": round_id,
            "hand": hand,
            "gesture_name": gesture,
            "command_success": True,
            "verified": verified,
            "failure_reason": reason,
        },
    }


def _health(side: str, temp: float, runtime_s: float) -> dict:
    return {
        "event_type": "health_check",
        "event_id": f"evt_h_{side}_{runtime_s}",
        "timestamp_ns": int(runtime_s * 1e9),
        "payload": {side: {"summary": {"temperature_max": temp}}, "runtime_s": runtime_s},
    }


# ---------------------------------------------------------------------------
# RPS implicit failure
# ---------------------------------------------------------------------------


def test_rps_invalid_becomes_failure_memory():
    events = [
        _round_event("stress_000001", "draw"),
        _round_event("stress_000002", "invalid", "joint_not_reached", choice="scissors"),
        _gesture_event("stress_000002", "right", "scissors", False, "joint_not_reached"),
    ]
    adapter = Rh56RpsAdapter()
    failures = adapter.extract_failures(_ctx(), events)
    assert len(failures) == 1
    item = failures[0]
    assert item.memory_type == "failure"
    assert item.failure_type == "joint_not_reached"
    assert item.gesture_name == "scissors"
    assert item.body_id == "rh56_right_01"
    assert item.outcome == "failure"
    assert item.metadata["round_id"] == "stress_000002"
    assert item.metadata["round_index"] == 2
    assert "evt_stress_000002" in item.evidence_refs
    assert "evt_g_stress_000002_right_scissors" in item.evidence_refs
    # bilingual document
    assert "[ZH]" in item.document and "[EN]" in item.document and "[CANONICAL]" in item.document
    assert "剪刀" in item.document
    # joint attribution is honest: session never recorded the joint
    assert item.joint_name is None
    assert item.metadata["joint_attribution"] == "not_recorded_in_session"


def test_rps_gesture_failure_in_valid_round_also_captured():
    events = [
        _round_event("stress_000001", "draw"),
        _gesture_event("stress_000001", "left", "left_ready", False, "joint_not_reached"),
    ]
    failures = Rh56RpsAdapter().extract_failures(_ctx(), events)
    assert len(failures) == 1
    assert failures[0].body_id == "rh56_left_01"
    assert failures[0].gesture_name == "left_ready"


def test_missing_gesture_status_is_not_invented_as_failure():
    event = {
        "event_type": "rps.gesture.executed",
        "event_id": "evt_unknown",
        "timestamp_ns": 1_000_000_000,
        "payload": {"round_id": "stress_000001", "hand": "left"},
    }

    assert Rh56RpsAdapter().extract_failures(_ctx(), [event]) == []
    assert extract_failure_memories(_ctx(), [event]) == []


def test_rps_adapter_selected_by_event_shape():
    adapter = adapter_for(_ctx(), [_round_event("stress_000001", "draw")])
    assert isinstance(adapter, Rh56RpsAdapter)


def test_rps_adapter_event_sniffing_is_not_limited_to_first_200_events():
    context = SessionContext(practice_id="old", task_id=None)
    events = [{"event_type": "unrelated"} for _ in range(250)]
    events.append(_round_event("stress_000001", "draw"))

    assert isinstance(adapter_for(context, events), Rh56RpsAdapter)


# ---------------------------------------------------------------------------
# Episode quality
# ---------------------------------------------------------------------------


def test_episode_quality_partial_success():
    events = [_round_event(f"stress_{i:06d}", "draw") for i in range(1, 95)]
    events += [
        _round_event(f"stress_{i:06d}", "invalid", "joint_not_reached") for i in range(95, 101)
    ]
    quality = Rh56RpsAdapter().build_episode_quality(_ctx(), events)
    assert quality["outcome"] == "partial_success"
    q = quality["quality"]
    assert q["total_rounds"] == 100
    assert q["verified_rounds"] == 94
    assert q["invalid_rounds"] == 6
    assert q["verified_rate"] == 0.94
    assert q["failure_distribution"] == {"joint_not_reached": 6}
    assert q["first_degradation_round"] == 95


def test_episode_quality_success_and_failure_bands():
    ok = [_round_event(f"stress_{i:06d}", "draw") for i in range(1, 100)]
    ok.append(_round_event("stress_000100", "invalid", "unverified"))
    assert Rh56RpsAdapter().build_episode_quality(_ctx(), ok)["outcome"] == "success"
    bad = [_round_event(f"stress_{i:06d}", "invalid", "unverified") for i in range(1, 101)]
    assert Rh56RpsAdapter().build_episode_quality(_ctx(), bad)["outcome"] == "failure"


def test_build_candidates_episode_carries_quality():
    events = [_round_event(f"stress_{i:06d}", "draw") for i in range(1, 10)]
    events.append(_round_event("stress_000010", "invalid", "unverified"))
    candidates = build_candidates(_ctx(), events)
    episode = next(c for c in candidates if c.memory_type == "episodic")
    assert episode.outcome == "partial_success"
    assert episode.metadata["quality"]["total_rounds"] == 10


# ---------------------------------------------------------------------------
# Body observation, never a limit
# ---------------------------------------------------------------------------


def test_body_temperature_is_observation_not_limit():
    events = [
        _health("right", 38.0, 0.0),
        _health("right", 48.0, 600.0),
        _gesture_event("stress_000033", "right", "rock", False, "joint_not_reached"),
        _round_event("stress_000033", "invalid", "joint_not_reached"),
    ]
    body = Rh56RpsAdapter().extract_body_patterns(_ctx(), events)
    assert len(body) == 1
    meta = body[0].metadata
    assert meta["observed_temperature_min"] == 38.0
    assert meta["observed_temperature_max"] == 48.0
    assert meta["causal_status"] == "observed_correlation"
    assert "thermal_limit" not in json.dumps(meta)
    assert "NOT a thermal limit" in body[0].document


def test_unattributed_invalid_round_does_not_claim_a_hand_correlation():
    events = [
        _health("left", 38.0, 0.0),
        _health("right", 39.0, 0.0),
        _health("left", 48.0, 600.0),
        _health("right", 49.0, 600.0),
        _round_event("stress_000033", "invalid", "joint_not_reached"),
    ]

    body = Rh56RpsAdapter().extract_body_patterns(_ctx(), events)

    assert len(body) == 2
    assert all(item.metadata["first_failure_temperature"] is None for item in body)
    assert all(item.metadata["causal_status"] == "insufficient_data" for item in body)


def test_empty_body_memory_ignored():
    # fewer than 2 health checks -> no body candidates at all
    assert Rh56RpsAdapter().extract_body_patterns(_ctx(), []) == []
    assert Rh56RpsAdapter().extract_body_patterns(_ctx(), [_health("left", 40.0, 0.0)]) == []


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def test_bilingual_document_builder():
    doc = MultilingualMemoryDocumentBuilder().build_failure(
        hand="right",
        joint="middle",
        gesture="scissors",
        failure_type="joint_not_reached",
        round_index=37,
        temperature_c=46.0,
    )
    assert "右手" in doc.zh and "中指" in doc.zh
    assert "right hand" in doc.en and "middle" in doc.en
    assert "joint=middle" in doc.canonical and "failure=joint_not_reached" in doc.canonical
    assert "剪刀" in doc.aliases and "scissors" in doc.aliases
    assert "middle" in doc.exact_terms
    for section in ("[ZH]", "[EN]", "[CANONICAL]", "[ALIASES]"):
        assert section in doc.combined


def test_unknown_hand_is_not_rendered_as_right_hand():
    doc = MultilingualMemoryDocumentBuilder().build_failure(
        hand="unknown",
        joint=None,
        gesture="ready",
        failure_type="unverified",
    )

    assert "未知手别" in doc.zh
    assert "右手" not in doc.zh


def test_prefixed_gesture_is_canonicalized_only_in_derived_document():
    doc = MultilingualMemoryDocumentBuilder().build_failure(
        hand="left",
        joint=None,
        gesture="left_scissors",
        failure_type="joint_not_reached",
    )

    assert "剪刀" in doc.zh
    assert "gesture=scissors" in doc.canonical
    assert "left_scissors" not in doc.combined
    assert "scissors" in doc.exact_terms


def test_exact_joint_extraction():
    assert extract_exact_terms("中指未到位")["joints"] == ["middle"]
    assert extract_exact_terms("拇指根关节")["joints"] == ["thumb_rot"]
    assert extract_exact_terms("RH56 middle joint_not_reached") == {
        "joints": ["middle"],
        "devices": ["RH56"],
        "failure_types": ["joint_not_reached"],
    }
    # thumb_rot never degrades into a bare thumb hit
    assert extract_exact_terms("thumb_rot")["joints"] == ["thumb_rot"]
    assert extract_exact_terms("thumb")["joints"] == ["thumb"]
    # bare left/right are NOT hand constraints; explicit forms are
    assert extract_exact_terms("you left the right copyright") == {}
    assert extract_exact_terms("左手")["hands"] == ["left"]
    assert extract_exact_terms("right hand rock")["hands"] == ["right"]
    # error codes verbatim
    assert extract_exact_terms("tty EIO -110")["error_codes"] == ["EIO", "-110"]


# ---------------------------------------------------------------------------
# Metadata flattening round-trip
# ---------------------------------------------------------------------------


def test_memory_item_flattened_fields_round_trip():
    item = MemoryItem(
        memory_type="failure",
        robot_id="rh56_rps_robot",
        title="t",
        document="d",
        task_name="RH56 Rock-Paper-Scissors",
        failure_type="joint_not_reached",
        joint_name="middle",
        gesture_name="scissors",
        embedding_profile_id="qwen3_06b_1024_v1",
        evidence_refs=["evt_x"],
    )
    record = item.to_record()
    for field_name, value in (
        ("task_name", "RH56 Rock-Paper-Scissors"),
        ("failure_type", "joint_not_reached"),
        ("joint_name", "middle"),
        ("gesture_name", "scissors"),
        ("embedding_profile_id", "qwen3_06b_1024_v1"),
    ):
        assert record[field_name] == value and isinstance(record[field_name], str)
    back = MemoryItem.from_record(record)
    assert back.failure_type == "joint_not_reached"
    assert back.joint_name == "middle"
    assert back.gesture_name == "scissors"
    assert back.task_name == "RH56 Rock-Paper-Scissors"
    assert back.embedding_profile_id == "qwen3_06b_1024_v1"
