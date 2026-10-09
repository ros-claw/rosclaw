"""Synthetic observation admission negatives, never real SDK/physics evidence."""

import importlib
from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("collection_pause_scenario")


def inputs():
    binding = {
        "run_id": "synthetic",
        "body_snapshot_hash": "body",
        "attachment_hash": "attachment",
        "producer_id": "producer",
    }
    sample = {
        "time_sec": 10,
        "lease_remaining_sec": 1,
        "physics_collision_count": 0,
        "captured_at": datetime.fromtimestamp(100, UTC).isoformat(),
        "evidence_domain": "GAZEBO_PHYSICS",
        "observation_complete": True,
        "brush_source_fault": None,
        "physics_source_fault": None,
        "brush_source_binding": binding,
        "cleaning_enabled": True,
    }
    snapshot = {
        "robot_sim_time_sec": 10,
        "sampled_monotonic_sec": 200,
        "source_fault": None,
        "live_source_constraint_satisfied": True,
        "robot_collision_count": 0,
        "probe_completed_cache_cycles": 1,
    }
    return sample, snapshot, binding


def test_healthy_measured_brush_on_is_only_eligible_trigger(module):
    sample, snapshot, binding = inputs()
    assert module.eligible_cleaning_sample(
        sample, snapshot, binding, unix_time=100.1, monotonic_time=200.1
    )
    sample["cleaning_enabled"] = False
    assert not module.eligible_cleaning_sample(
        sample, snapshot, binding, unix_time=100.1, monotonic_time=200.1
    )


@pytest.mark.parametrize(
    "fault",
    [
        "contact",
        "clock",
        "stale",
        "binding",
        "missing_brush",
        "source_fault",
        "unknown",
        "nan",
        "future",
    ],
)
def test_unknown_or_faulted_source_cannot_trigger_registered_collection_pause(module, fault):
    sample, snapshot, binding = inputs()
    if fault == "contact":
        sample["physics_collision_count"] = 1
    elif fault == "clock":
        snapshot["robot_sim_time_sec"] = 9
    elif fault == "stale":
        snapshot["sampled_monotonic_sec"] = 199
    elif fault == "binding":
        sample["brush_source_binding"] = {}
    elif fault == "missing_brush":
        sample.pop("brush_source_fault")
    elif fault == "source_fault":
        snapshot["source_fault"] = "original fault"
    elif fault == "unknown":
        snapshot["live_source_constraint_satisfied"] = None
    elif fault == "nan":
        sample["time_sec"] = float("nan")
    else:
        sample["captured_at"] = datetime.fromtimestamp(101, UTC).isoformat()
    with pytest.raises(ValueError):
        module.eligible_cleaning_sample(
            sample, snapshot, binding, unix_time=100.1, monotonic_time=200.1
        )
