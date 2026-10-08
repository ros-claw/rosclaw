"""Synthetic original-byte join contracts; no live DDS, World or authority."""

import importlib
import itertools
import json

import pytest

from tests.connectors.ros import test_probe_scene_geometry as spatial_contract

spatial_fixture = spatial_contract.fixture


def join_fixture(spatial_fixture):
    geometry, scene, robot, probe, _, _, _ = spatial_fixture
    join = importlib.import_module("probe_scene_join").ProbeSceneJoin(geometry)
    packets = {"scene": scene, "robot": robot, "probe": probe}

    def receive(kind, wall=100.0, unix=None):
        return join.receive(
            kind,
            json.dumps(packets[kind]).encode(),
            received_monotonic_sec=wall,
            received_unix_ns=unix
            or scene["captured_at_unix_ns"] + round((wall - 100) * 1e9) + 1_000_000,
        )

    return join, packets, receive


@pytest.mark.parametrize("order", list(itertools.permutations(("scene", "robot", "probe"))))
def test_every_cross_stream_order_joins_only_original_exact_step(spatial_fixture, order):
    join, packets, receive = join_fixture(spatial_fixture)
    results = []
    # The last producer can capture after the scene's original receipt. It is
    # checked at its own receipt and at join time, without rewriting the scene.
    packets[order[-1]]["captured_at_unix_ns"] += 10_000_000
    for i, kind in enumerate(order):
        rows = receive(kind, 100 + i * 0.02)
        assert len(rows) == (1 if i == 2 else 0)
        results += rows
    assert results[0]["original_sim_step_ns"] == 1_000_000_000
    assert results[0]["original_source_receipts"][order[0]]["monotonic_sec"] == 100
    assert join.snapshot(100.05)["scene_geometry_constraint_satisfied"]
    # Joining later must not extend the oldest original source's lifetime.
    assert not join.snapshot(100.301)["scene_geometry_constraint_satisfied"]


def test_adjacent_source_steps_cannot_be_interpolated_into_counterparts(spatial_fixture):
    join, packets, receive = join_fixture(spatial_fixture)
    receive("scene")
    packets["robot"]["sim_time_sec"] += 0.01
    assert receive("robot", 100.01) == []
    assert receive("probe", 100.02) == []
    result = join.snapshot(100.31)
    assert not result["scene_geometry_constraint_satisfied"]
    assert "lacks timely exact" in result["join_source_fault"]
    packets["robot"]["sim_time_sec"] -= 0.01
    with pytest.raises(ValueError, match="remains latched"):
        receive("robot", 100.32)


@pytest.mark.parametrize("fault", ["repeat", "sequence_gap", "clock_backwards", "stale", "future"])
def test_original_source_discontinuity_and_age_fail_closed(spatial_fixture, fault):
    join, packets, receive = join_fixture(spatial_fixture)
    receive("robot")
    p = packets["robot"]
    if fault != "repeat":
        p["sequence"] += 1
        p["sim_time_sec"] += 0.01
    if fault == "sequence_gap":
        p["sequence"] += 1
    elif fault == "clock_backwards":
        p["sim_time_sec"] -= 0.02
    elif fault == "stale":
        p["captured_at_unix_ns"] -= 400_000_000
    elif fault == "future":
        p["captured_at_unix_ns"] += 400_000_000
    expected = (
        "step/sequence"
        if fault in {"repeat", "sequence_gap", "clock_backwards"}
        else "stale or future"
    )
    with pytest.raises(ValueError, match=expected):
        receive("robot", 100.01)
    assert join.fault


def test_all_step_intermediate_robot_packets_are_not_geometry_or_contact_proof(spatial_fixture):
    join, packets, receive = join_fixture(spatial_fixture)
    for kind in ("scene", "robot", "probe"):
        receive(kind)
    for i in range(1, 6):
        p = packets["robot"]
        p["sequence"] = i
        p["iterations"] = 100 + i
        p["sim_time_sec"] = 1 + i * 0.01
        p["captured_at_unix_ns"] = packets["scene"]["captured_at_unix_ns"] + i * 10_000_000
        assert receive("robot", 100 + i * 0.01) == []
    for kind in ("scene", "probe"):
        packets[kind]["sequence"] += 1
        packets[kind]["physics_iteration" if kind == "scene" else "iterations"] += 5
        packets[kind]["sim_time_sec"] = 1.05
        packets[kind]["captured_at_unix_ns"] += 50_000_000
        rows = receive(kind, 100.05, packets[kind]["captured_at_unix_ns"] + 1_000_000)
    assert len(rows) == 1
    result = join.snapshot(100.06)
    assert result["completed_exact_scene_joins"] == 2
    assert result["sampling_semantics"].startswith("SCENE_20HZ")
    assert not result["backend_health_admitted"] and not result["authorization"]


def test_no_geometry_is_unknown_even_with_complete_native_pairs(spatial_fixture):
    join, _, receive = join_fixture(spatial_fixture)
    receive("robot")
    receive("probe")
    result = join.snapshot(100.1)
    assert not result["scene_geometry_constraint_satisfied"]
    assert result["completed_exact_scene_joins"] == 0


def test_oversized_backlog_rejects_before_any_silent_scene_drop(spatial_fixture):
    join, packets, receive = join_fixture(spatial_fixture)
    for i in range(8):
        p = packets["scene"]
        p.update(sequence=i, physics_iteration=100 + i, sim_time_sec=1 + i * 0.001)
        receive("scene", 100 + i * 0.001)
    p.update(sequence=8, physics_iteration=108, sim_time_sec=1.008)
    with pytest.raises(ValueError, match="backlog exceeded"):
        receive("scene", 100.008)


@pytest.mark.parametrize("clock", [99.9, float("nan"), True])
def test_invalid_sample_clock_latches_source_constraint(spatial_fixture, clock):
    join, _, receive = join_fixture(spatial_fixture)
    receive("robot")
    if clock == 99.9:
        assert not join.snapshot(clock)["scene_geometry_constraint_satisfied"]
    else:
        with pytest.raises(ValueError):
            join.snapshot(clock)
    assert join.fault
