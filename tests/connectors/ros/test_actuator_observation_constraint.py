"""Interlock logic and actual actor callbacks with recorders; no ROS/DDS/physics."""

import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.connectors.ros.test_split_sim_actuator import BINDING, FakePath, load_node

CONFIG = {"run_id": "run", "body_snapshot_hash": "body", "constraint_policy_hash": "a" * 64}


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("actuator_observation_constraint")


def row(sequence=0, wall=100, ready=True):
    return {
        **CONFIG,
        "schema_version": "rosclaw.actor_observation_interlock.v1",
        "sequence": sequence,
        "sampled_monotonic_sec": wall,
        "live_source_constraint_satisfied": ready,
        "robot_sim_time_sec": 1.0 if ready else None,
        "probe_sim_time_sec": 1.0 if ready else None,
        "robot_collision_count": 0,
        "probe_completed_cache_cycles": 1 if ready else 0,
        "source_fault": None,
        "authorization": False,
    }


def test_initial_unknown_blocks_but_fresh_qualified_observation_can_open(module):
    gate = module.ActuatorObservationConstraint(CONFIG, BINDING)
    assert not gate.ready(100, 1) and gate.fault is None
    assert gate.receive(json.dumps(row(ready=False)), 100)
    assert not gate.ready(100.01, 1) and gate.fault is None
    assert gate.receive(json.dumps(row(1, 100.02)), 100.03)
    assert gate.ready(100.04, 1)
    assert not gate.ready(100.18, 1) and gate.fault
    assert gate.receive(json.dumps(row(2, 100.19)), 100.19) is False
    assert not gate.ready(100.19, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "other"},
        {"body_snapshot_hash": "other"},
        {"constraint_policy_hash": "b" * 64},
        {"authorization": True},
        {"sequence": True},
        {"sequence": -1},
        {"sequence": 1},
        {"sampled_monotonic_sec": float("nan")},
        {"sampled_monotonic_sec": 101},
        {"sampled_monotonic_sec": 99},
        {"robot_collision_count": 1},
        {"robot_collision_count": False},
        {"probe_completed_cache_cycles": 0},
        {"probe_sim_time_sec": 2},
        {"robot_sim_time_sec": None},
        {"source_fault": "missed native physics step"},
        {"live_source_constraint_satisfied": 1},
        {"unexpected": 1},
    ],
)
def test_malformed_or_conflicting_source_never_admits(module, changes):
    gate = module.ActuatorObservationConstraint(CONFIG, BINDING)
    assert not gate.receive(json.dumps({**row(), **changes}), 100)
    assert gate.fault and not gate.ready(100, 1)


@pytest.mark.parametrize("raw", ['{"sequence":0,"sequence":1}', "[", "x" * 4097])
def test_raw_malformed_duplicate_or_oversized_json_latches(module, raw):
    gate = module.ActuatorObservationConstraint(CONFIG, BINDING)
    assert not gate.receive(raw, 100) and gate.fault


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence": 0},
        {"sequence": 2},
        {"sampled_monotonic_sec": 100},
        {"live_source_constraint_satisfied": False},
    ],
)
def test_replay_gap_clock_regression_and_source_loss_latch_after_open(module, changes):
    gate = module.ActuatorObservationConstraint(CONFIG, BINDING)
    gate.receive(json.dumps(row()), 100)
    assert gate.ready(100.01, 1)
    assert not gate.receive(json.dumps({**row(1, 100.02), **changes}), 100.02)
    assert not gate.ready(100.03, 1) and gate.fault


def test_actor_actual_callbacks_block_unknown_then_stop_on_clock_loss(monkeypatch, module):
    monkeypatch.setattr(
        FakePath, "extra", {"/evidence/backend_actor_constraint.json": json.dumps(CONFIG)}
    )
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch, backend_required=True)
    assert "/rosclaw_sim/backend_observation_constraint" in actor.subscriptions
    assert not actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace()).success
    assert not actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace()).success
    before = len(actor.publishers["/drive_controller/cmd_vel"])
    actor.command(SimpleNamespace(marker="forbidden"))
    assert len(actor.publishers["/drive_controller/cmd_vel"]) == before + 1  # zero stop only
    actor.observation(SimpleNamespace(data=json.dumps(row(wall=time.monotonic()))))
    assert actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace()).success
    assert actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace()).success
    command = SimpleNamespace(marker="leased_command")
    actor.command(command)
    assert actor.publishers["/drive_controller/cmd_vel"][-1].twist is command
    actor.sim_time = 1.2
    actor.tick()
    assert actor.fault and actor.lease == 0 and not actor.cleaning
    assert actor.publishers["/rosclaw_sim/cleaning_state"][-1].data is False
    actor.observation(SimpleNamespace(data=json.dumps(row(1, time.monotonic()))))
    assert not actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace()).success
    assert not actor.set_hold(SimpleNamespace(data=False), SimpleNamespace()).success


def test_actor_positive_contact_immediately_zeros_lease_and_brush(monkeypatch, module):
    monkeypatch.setattr(
        FakePath, "extra", {"/evidence/backend_actor_constraint.json": json.dumps(CONFIG)}
    )
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch, backend_required=True)
    actor.observation(SimpleNamespace(data=json.dumps(row(wall=time.monotonic()))))
    actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace())
    actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace())
    actor.observation(
        SimpleNamespace(data=json.dumps({**row(1, time.monotonic()), "robot_collision_count": 1}))
    )
    assert actor.fault and not actor.cleaning and actor.lease == 0
    assert actor.publishers["/rosclaw_sim/cleaning_state"][-1].data is False


def test_transport_projection_is_exact_and_cannot_claim_permission(module):
    original = row()
    snapshot = {
        k: v
        for k, v in original.items()
        if k not in {"schema_version", "sequence", "run_id", "body_snapshot_hash", "authorization"}
    }
    assert module.observation_envelope(snapshot, BINDING, 0) == original


def test_actor_original_source_audit_loss_withdraws_ready_and_lease(monkeypatch, module):
    monkeypatch.setattr(
        FakePath, "extra", {"/evidence/backend_actor_constraint.json": json.dumps(CONFIG)}
    )
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch, backend_required=True)
    actor.observation(SimpleNamespace(data=json.dumps(row(wall=time.monotonic()))))
    assert actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace()).success
    actor.observation_audit.dropped = 1
    actor.tick()
    assert actor.fault and actor.lease == 0 and not actor.cleaning
