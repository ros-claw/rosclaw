"""Synthetic state ordering tests; no observer/actuator transport is installed."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from rosclaw.connectors.ros.verification.brush_timeline import BrushStateEvent, BrushStateTimeline

NOW = datetime(2026, 10, 8, tzinfo=UTC)


def setup(max_buffer=2048):
    return BrushStateTimeline(
        run_id="run",
        body_snapshot_hash="body",
        attachment_hash="brush",
        producer_id="owned_actuator",
        max_buffer=max_buffer,
    )


def event(sequence, sim, enabled=False, kind="WATERMARK"):
    return BrushStateEvent(
        "run",
        "body",
        "brush",
        "owned_actuator",
        sequence,
        sim,
        kind,
        enabled,
        NOW.isoformat(),
        True,
    )


def append(timeline, e, mono=10):
    timeline.append(e, artifact_hash=e.artifact_hash(), now=NOW, received_monotonic=mono)


def test_pending_pose_cannot_use_last_enabled_flag_and_late_disable_prevents_false_credit():
    timeline = setup()
    append(timeline, event(0, 0))
    append(timeline, event(1, 1, True, "TRANSITION"))
    append(timeline, event(2, 1.1, True))
    assert timeline.state_at(1.05, now_monotonic=10.01)["enabled"] is True
    result = timeline.state_at(2, now_monotonic=10.02)
    assert result["status"] == "PENDING" and result["enabled"] is None
    append(timeline, event(3, 1.5, False, "TRANSITION"), mono=10.03)
    append(timeline, event(4, 2.1, False), mono=10.04)
    assert timeline.state_at(2, now_monotonic=10.05)["enabled"] is False


def test_watermark_is_exclusive_until_same_sim_tick_transitions_are_known():
    timeline = setup()
    append(timeline, event(0, 0))
    append(timeline, event(1, 1, True, "TRANSITION"))
    append(timeline, event(2, 2, True))
    assert timeline.state_at(2, now_monotonic=10.01)["status"] == "PENDING"
    append(timeline, event(3, 2, False, "TRANSITION"))
    append(timeline, event(4, 2.1, False))
    assert timeline.state_at(2, now_monotonic=10.01)["enabled"] is False


def test_enable_boundary_receives_no_credit_and_only_later_known_poses_enable():
    timeline = setup()
    append(timeline, event(0, 0))
    append(timeline, event(1, 1, True, "TRANSITION"))
    append(timeline, event(2, 2, True))
    assert timeline.state_at(1, now_monotonic=10.01)["enabled"] is False
    assert timeline.state_at(1.1, now_monotonic=10.02)["enabled"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "other"},
        {"body_snapshot_hash": "other"},
        {"attachment_hash": "other"},
        {"producer_id": "agent"},
        {"evidence_domain": "REAL"},
        {"complete": False},
        {"sim_time_sec": -0.1},
        {"sim_time_sec": float("nan")},
        {"sequence": True},
        {"sequence": 3},
        {"enabled": 1},
        {"captured_at": (NOW - timedelta(seconds=1)).isoformat()},
        {"captured_at": (NOW + timedelta(seconds=1)).isoformat()},
        {"captured_at": "2026-10-08T00:00:00"},
    ],
)
def test_source_order_or_capture_faults_latch_and_cannot_credit_future_poses(changes):
    timeline = setup()
    append(timeline, event(0, 0))
    bad = replace(event(1, 1), **changes)
    try:
        hashed = bad.artifact_hash()
    except ValueError:
        hashed = "invalid"
    with pytest.raises(ValueError):
        timeline.append(bad, artifact_hash=hashed, now=NOW, received_monotonic=10)
    with pytest.raises(ValueError, match="latched"):
        timeline.state_at(1, now_monotonic=10.01)


def test_changed_state_without_transition_and_retroactive_transition_fail_closed():
    timeline = setup()
    append(timeline, event(0, 1))
    with pytest.raises(ValueError, match="without"):
        append(timeline, event(1, 2, True))
    timeline = setup()
    append(timeline, event(0, 1))
    append(timeline, event(1, 2))
    with pytest.raises(ValueError, match="retroactively"):
        append(timeline, event(2, 1.5, True, "TRANSITION"))


def test_missing_initial_brush_off_or_stale_source_cannot_credit():
    timeline = setup()
    with pytest.raises(ValueError, match="initial"):
        append(timeline, event(0, 1, True))
    timeline = setup()
    append(timeline, event(0, 1))
    with pytest.raises(ValueError, match="stale"):
        timeline.state_at(1.5, now_monotonic=11)


def test_bounded_unconsumed_history_and_pose_reuse_latch_failure():
    timeline = setup(max_buffer=2)
    append(timeline, event(0, 0))
    append(timeline, event(1, 1, True, "TRANSITION"))
    with pytest.raises(ValueError, match="buffer"):
        append(timeline, event(2, 2, False, "TRANSITION"))
    timeline = setup()
    append(timeline, event(0, 0))
    append(timeline, event(1, 2))
    timeline.state_at(1, now_monotonic=10.01)
    with pytest.raises(ValueError, match="reused"):
        timeline.state_at(1, now_monotonic=10.02)


def test_pending_state_has_no_pair_credit_or_complete_timeline():
    timeline = setup()
    append(timeline, event(0, 0))
    assert timeline.state_at(1, now_monotonic=10.01)["status"] == "PENDING"
    assert timeline.result()["paired_pose_count"] == 0
    assert timeline.result()["complete"] is False
    append(timeline, event(1, 2))
    assert timeline.state_at(1, now_monotonic=10.02)["status"] == "PAIRED"
    assert timeline.result()["paired_pose_count"] == 1
    assert timeline.result()["complete"] is True


def test_pair_artifact_binds_actual_watermark_and_successive_pose_chain():
    timeline = setup()
    append(timeline, event(0, 0))
    watermark = event(1, 3)
    append(timeline, watermark)
    first = timeline.state_at(1, now_monotonic=10.01)
    second = timeline.state_at(2, now_monotonic=10.02)
    assert first["watermark_event_hash"] == watermark.artifact_hash()
    assert second["pair_chain_hash"] != first["pair_chain_hash"]
    assert timeline.result()["pair_chain_hash"] == second["pair_chain_hash"]
    assert timeline.result()["paired_pose_count"] == 2


def test_corrupted_event_digest_latches_before_any_brush_pair():
    timeline = setup()
    with pytest.raises(ValueError, match="integrity"):
        timeline.append(event(0, 0), artifact_hash="tampered", now=NOW, received_monotonic=10)
    assert timeline.result()["complete"] is False
    assert timeline.result()["paired_pose_count"] == 0
