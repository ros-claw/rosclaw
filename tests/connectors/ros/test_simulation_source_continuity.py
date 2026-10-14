"""Pure continuity contracts; no motion or initial source admission is proved."""

import hashlib
import importlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tests.connectors.ros.test_sim_source_binding import fixture as source_fixture


def setup_guard(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[3] / "integrations/ros_probe/acceptance")
    )
    cls = importlib.import_module("simulation_source_continuity").SimulationSourceContinuityGuard
    parts = source_fixture(tmp_path)
    sample = deepcopy(parts[4][0])
    sample["observer_run_label"] = "a" * 32
    raw = encoded(sample)
    kwargs = {
        "expected_sha256": hashlib.sha256(raw).hexdigest(),
        "run_id": "a" * 32,
        "world_id": sample["worlds"][0]["name"],
        "robot_entity": sample["models"][0]["name"],
        "expected_world_entity": 1,
        "expected_robot_entity": 2,
        "max_age_ms": 500,
        "now_monotonic_ns": sample["observer_monotonic_ns"] + 1_000_000,
    }
    return cls.from_pinned_sample(raw, **kwargs), sample, cls, kwargs


def encoded(sample):
    return json.dumps(sample, allow_nan=False).encode()


def advance(sample):
    sample = deepcopy(sample)
    for key in ("sim_time_ns", "observer_monotonic_ns"):
        sample[key] += 100_000_000
    sample["iterations"] += 10
    sample["sample"] += 1
    return sample


def test_navigation_pose_changes_and_identical_fresh_reads_are_continuous(tmp_path, monkeypatch):
    guard, sample, _, _ = setup_guard(tmp_path, monkeypatch)
    result = guard.observe(
        encoded(sample), now_monotonic_ns=sample["observer_monotonic_ns"] + 2_000_000
    )
    assert result["continuous"] and result["execution_authority"] == "NONE"
    sample = advance(sample)
    sample["models"][0]["world_pose"] = [0.2, 0.1, 0, 0, 0, 0.25]
    result = guard.observe(
        encoded(sample), now_monotonic_ns=sample["observer_monotonic_ns"] + 1_000_000
    )
    assert result["continuous"] and result["L1"] == "NOT_EVALUATED" and result["L2"] == "NOT_RUN"
    with pytest.raises(FrozenInstanceError):
        guard.max_age_ns = 9_000_000_000


@pytest.mark.parametrize(
    "fault",
    [
        "stale",
        "future",
        "run",
        "reset",
        "paused",
        "incomplete",
        "world_entity",
        "robot_entity",
        "collision_entity",
        "geometry",
        "local_pose",
        "new_model",
        "duplicate_member",
        "orphan_member",
        "clock_freeze",
        "counter_regression",
        "same_counter_changed",
        "malformed",
    ],
)
def test_loss_is_terminal_even_if_a_later_sample_looks_healthy(tmp_path, monkeypatch, fault):
    guard, reference, _, _ = setup_guard(tmp_path, monkeypatch)
    sample = advance(reference)
    now = sample["observer_monotonic_ns"] + 1_000_000
    if fault == "stale":
        now += 500_000_000
    elif fault == "future":
        now = sample["observer_monotonic_ns"] - 1
    elif fault == "run":
        sample["observer_run_label"] = "b" * 32
    elif fault == "reset":
        sample["reset_epoch"] = 1
    elif fault == "paused":
        sample["paused"] = True
    elif fault == "incomplete":
        sample["inventory_complete"] = False
    elif fault == "world_entity":
        sample["worlds"][0]["entity"] = 9
    elif fault == "robot_entity":
        sample["models"][0]["entity"] = 9
    elif fault == "collision_entity":
        sample["collisions"][0]["entity"] = 99
    elif fault == "geometry":
        sample["collisions"][0]["geometry_sdf"] = (
            "<geometry><box><size>.2 .1 .1</size></box></geometry>"
        )
    elif fault == "local_pose":
        sample["collisions"][0]["pose"][0] = 0.1
    elif fault == "new_model":
        sample["models"].append(
            {"name": "unexpected", "entity": 9, "static": False, "world_pose": [0] * 6}
        )
    elif fault == "duplicate_member":
        sample["collisions"].append(deepcopy(sample["collisions"][0]))
    elif fault == "orphan_member":
        sample["collisions"][0]["ancestry"][1]["entity"] = 999
    elif fault == "clock_freeze":
        sample["sim_time_ns"] = reference["sim_time_ns"]
    elif fault == "counter_regression":
        sample["iterations"] = 0
    elif fault == "same_counter_changed":
        sample["sample"] = reference["sample"]
    raw = b'{"truncated":' if fault == "malformed" else encoded(sample)
    result = guard.observe(raw, now_monotonic_ns=now)
    assert result["terminal"] and not result["continuous"] and result["reason"]
    healthy = advance(advance(reference))
    assert (
        guard.observe(encoded(healthy), now_monotonic_ns=healthy["observer_monotonic_ns"]) == result
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_sha256", "0" * 64),
        ("expected_world_entity", 7),
        ("expected_robot_entity", 8),
        ("max_age_ms", 501),
        ("max_age_ms", True),
        ("run_id", "different"),
        ("now_monotonic_ns", 9_000_000_000),
    ],
)
def test_initial_independent_pins_and_freshness_are_required(tmp_path, monkeypatch, field, value):
    _, sample, cls, kwargs = setup_guard(tmp_path, monkeypatch)
    kwargs[field] = value
    with pytest.raises((ValueError, TypeError)):
        cls.from_pinned_sample(encoded(sample), **kwargs)
