"""Synthetic SDK daemon supervision producer contracts, never motion proof."""

import importlib
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from tests.connectors.ros.test_navigation_only_executor import compiled_body


def healthy_status(body_id="mobile"):
    return {
        "daemon_instance_id": "daemon_synthetic",
        "daemon_pid": os.getpid(),
        "daemon_uid": os.geteuid(),
        "robot_id": body_id,
        "running": True,
        "southbound_owner": "rosclawd",
        "runtime_state": "RUNNING",
        "supervision_state": "DISARMED",
        "emergency_stop_latched": False,
        "hardware_actions_executed": 0,
        "watchdog": {"running": True, "healthy": True},
        "ledger": {"integrity_verified": True, "write_failed": False},
        "recovery": {"required": False},
        "queue": {"RUNNING": 1},
        "drivers": ["gazebo_ros_expert"],
        "registered_executors": [
            "sandbox.reach:SIMULATION",
            "sandbox.reach:FIXTURE",
            "navigation.navigate_to_pose:SIMULATION",
        ],
    }


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_navigation_supervision")


@pytest.mark.parametrize(
    "field,value",
    [
        ("daemon_instance_id", "new-generation"),
        ("daemon_pid", -1),
        ("daemon_uid", -1),
        ("robot_id", "another"),
        ("running", False),
        ("runtime_state", "STOPPED"),
        ("supervision_state", "ARMED"),
        ("emergency_stop_latched", True),
        ("hardware_actions_executed", 1),
        ("watchdog", {"running": True, "healthy": False}),
        ("ledger", {"integrity_verified": True, "write_failed": True}),
        ("recovery", {"required": True}),
        ("queue", {"RUNNING": 0}),
        ("registered_executors", ["navigation.navigate_to_pose:REAL"]),
        ("drivers", []),
    ],
)
def test_active_sim_status_refuses_generation_health_or_authority_change(module, field, value):
    status = healthy_status()
    status[field] = value
    with pytest.raises(ValueError):
        module.review_active_sim_daemon(
            status,
            instance_id="daemon_synthetic",
            pid=os.getpid(),
            uid=os.geteuid(),
            body_id="mobile",
        )


@pytest.fixture
def producer(module, tmp_path):
    binding = SimulationNavigationBinding.from_body(compiled_body())

    class SyntheticFacts(module.CurrentSimulationFacts):
        def __init__(self):
            self.enrollment = {
                "request": {
                    "body_snapshot_hash": binding.body_snapshot_hash,
                    "sim_run_id": binding.run_id,
                    "manifest_hash": "navbodyworkspace_" + "b" * 64,
                },
                "world_sha256": "c" * 64,
                "source_pins": {"observer_epoch": "d" * 32},
            }
            self.capacity = True
            self.reader = SimpleNamespace(capacity_covers=lambda **_: self.capacity)

        def collect(self):
            now = time.monotonic_ns()
            return {
                "source": {"last_sample": 25, "reviewed_at_monotonic_ns": now},
                "health": {"status": "HEALTHY_SIM_OBSERVATION_ONLY"},
                "health_review_ns": now,
            }

        def close(self):
            pass

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    facts, status, key = SyntheticFacts(), healthy_status(), bytes(range(32))

    class SyntheticLeaseObservation(module.ExternalSimulationLeaseObservation):
        def __init__(self):
            self.terminal = False

        def read(self):
            return {"known": True, "lease_enabled": False, "status": {"terminal": self.terminal}}

    guard = module.SdkNavigationExecutionSupervision(
        binding=binding,
        facts=facts,
        daemon_instance_id="daemon_synthetic",
        daemon_status=lambda: status,
        key=key,
        proof_file=private / "proof.json",
        audit_file=tmp_path / "daemon-supervision-original.jsonl",
        lease_observation=SyntheticLeaseObservation(),
        allowed_action_ids=("synthetic-position", "synthetic-yaw"),
    )
    try:
        yield guard, facts, status, key, private
    finally:
        guard.close()


def test_actual_producer_writes_authenticated_private_expiring_proof(module, producer):
    guard, _, _, key, private = producer
    guard.begin("synthetic-position", duration_sec=1.0)
    proof_module = importlib.import_module("simulation_execution_proof")
    pins = proof_module.ExecutionProofPins(**guard._pins, action_id="synthetic-position")
    verifier = proof_module.ExecutionProofVerifier(pins=pins, key=key)
    first = verifier.verify((private / "proof.json").read_bytes(), now_ns=time.monotonic_ns())
    guard.check("synthetic-position")
    second = verifier.verify((private / "proof.json").read_bytes(), now_ns=time.monotonic_ns())
    assert second["sequence"] > first["sequence"]
    assert (private / "proof.json").stat().st_mode & 0o777 == 0o600
    assert second["expires_ns"] - second["issued_ns"] <= 250_000_000


def test_capacity_refusal_precedes_proof_and_is_terminal(producer):
    guard, facts, _, _, private = producer
    facts.capacity = False
    with pytest.raises(ValueError, match="budget"):
        guard.begin("synthetic-position", duration_sec=1.0)
    assert not (private / "proof.json").exists()
    facts.capacity = True
    with pytest.raises(ValueError, match="latched"):
        guard.begin("synthetic-position", duration_sec=1.0)


def test_daemon_generation_loss_removes_old_proof_and_prevents_resume(producer):
    guard, _, status, _, private = producer
    guard.begin("synthetic-position", duration_sec=1.0)
    status["daemon_instance_id"] = "new-generation"
    with pytest.raises(ValueError, match="identity"):
        guard.check("synthetic-position")
    assert not (private / "proof.json").exists()
    status["daemon_instance_id"] = "daemon_synthetic"
    with pytest.raises(ValueError, match="latched"):
        guard.begin("synthetic-yaw", duration_sec=1.0)


def test_replaced_private_directory_cannot_redirect_cleanup(producer, tmp_path):
    guard, _, _, _, private = producer
    guard.begin("synthetic-position", duration_sec=1.0)
    original = tmp_path / "original"
    private.rename(original)
    private.mkdir(mode=0o700)
    unrelated = private / "proof.json"
    unrelated.write_bytes(b"must survive")
    with pytest.raises(ValueError, match="directory changed"):
        guard.check("synthetic-position")
    assert unrelated.read_bytes() == b"must survive"
    assert not (original / "proof.json").exists()


def test_terminal_sink_cannot_be_hidden_by_healthy_source_and_daemon(producer):
    guard, _, _, _, private = producer
    guard.begin("synthetic-position", duration_sec=1.0)
    guard.lease_observation.terminal = True
    with pytest.raises(ValueError, match="sink state"):
        guard.check("synthetic-position")
    assert not (private / "proof.json").exists()
