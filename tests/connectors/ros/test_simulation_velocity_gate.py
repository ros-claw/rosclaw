"""Synthetic final-sink state contracts; no ROS publisher or physical proof."""

import dataclasses
import importlib
from pathlib import Path

import pytest

from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding


@pytest.fixture
def setup_gate(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    proof = importlib.import_module("simulation_execution_proof")
    gate_module = importlib.import_module("simulation_velocity_gate")
    binding = SimulationNavigationBinding(
        "synthetic",
        "a" * 64,
        "/navigate",
        "/lease",
        "/observe",
        "world",
        "run-1",
        1.0,
        3.0,
        0.05,
        0.1,
        0.2,
        0.5,
        0.005,
        0.01,
    )
    pins = proof.ExecutionProofPins(
        "synthetic",
        "a" * 64,
        "navbodyworkspace_" + "b" * 64,
        "run-1",
        "c" * 64,
        "d" * 32,
        "daemon-synthetic",
        "action-synthetic",
    )
    clock = [10_000_000_000]
    payload = {
        **dataclasses.asdict(pins),
        "schema_version": "rosclaw.sim_execution_supervision.v1",
        "sequence": 1,
        "source_sample": 25,
        "health_sha256": "e" * 64,
        "source_review_ns": clock[0],
        "health_review_ns": clock[0],
        "issued_ns": clock[0],
        "expires_ns": clock[0] + 250_000_000,
    }
    key = bytes(range(32))
    raw = proof.sign_execution_proof(payload, pins=pins, key=key, now_ns=clock[0])
    local = {
        "health": {
            "status": "HEALTHY_SIM_OBSERVATION_ONLY",
            "body_snapshot_hash": "a" * 64,
            "manifest_hash": "navbodyworkspace_" + "b" * 64,
            "sim_run_id": "run-1",
        },
        "health_review_ns": clock[0],
        "source": {
            "continuous": True,
            "terminal": False,
            "reason": None,
            "last_sample": 25,
            "reviewed_at_monotonic_ns": clock[0],
        },
    }
    gate = gate_module.SimulationVelocityGate(
        binding=binding,
        pins=pins,
        key=key,
        collect_local=lambda: local,
        clock=lambda: clock[0],
    )
    return gate, raw, clock, local


def command():
    return {"linear": {"x": 0.1, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.2}}


def test_prelease_gate_never_forwards_and_valid_lease_stays_within_body(setup_gate):
    gate, raw, _, _ = setup_gate
    assert gate.validate_command(command(), raw=raw) is None
    assert gate.watchdog(raw) is False
    assert gate.refresh_lease(raw) == 0.25
    output = gate.validate_command(command(), raw=raw)
    assert output == command() and gate.watchdog(raw) is True
    output["linear"]["x"] = 999
    assert gate.validate_command(command(), raw=raw) == command()


@pytest.mark.parametrize(
    "fault",
    [
        "health",
        "body",
        "run",
        "manifest",
        "source",
        "terminal",
        "reason",
        "cursor",
        "stale_health",
        "future_source",
        "malformed",
        "collector_failure",
    ],
)
def test_local_loss_refuses_valid_daemon_proof_and_latches(setup_gate, fault):
    gate, raw, clock, local = setup_gate
    gate.refresh_lease(raw)
    if fault == "health":
        local["health"]["status"] = "UNKNOWN"
    elif fault in {"body", "run", "manifest"}:
        field = {"body": "body_snapshot_hash", "run": "sim_run_id", "manifest": "manifest_hash"}[
            fault
        ]
        local["health"][field] = "wrong"
    elif fault == "source":
        local["source"]["continuous"] = False
    elif fault == "terminal":
        local["source"]["terminal"] = True
    elif fault == "reason":
        local["source"]["reason"] = "source replaced"
    elif fault == "cursor":
        local["source"]["last_sample"] = 24
    elif fault == "stale_health":
        local["health_review_ns"] = clock[0] - 500_000_001
    elif fault == "future_source":
        local["source"]["reviewed_at_monotonic_ns"] = clock[0] + 1
    elif fault == "malformed":
        local.clear()
    else:

        def broken():
            raise OSError("fixture collector disappeared")

        gate._collect_local = broken
    with pytest.raises(ValueError, match="latched"):
        gate.validate_command(command(), raw=raw)
    with pytest.raises(ValueError, match="latched"):
        gate.refresh_lease(raw)
    assert gate._deadline == 0


@pytest.mark.parametrize(
    "group,axis,value",
    [
        ("linear", "x", 0.201),
        ("linear", "x", -0.201),
        ("angular", "z", 0.501),
        ("linear", "y", 0.001),
        ("linear", "z", 0.001),
        ("angular", "x", 0.001),
        ("angular", "y", 0.001),
        ("linear", "x", float("nan")),
        ("angular", "z", float("inf")),
        ("linear", "x", True),
    ],
)
def test_excessive_nonplanar_or_nonfinite_twist_is_rejected_not_clamped(
    setup_gate, group, axis, value
):
    gate, raw, _, _ = setup_gate
    gate.refresh_lease(raw)
    twist = command()
    twist[group][axis] = value
    with pytest.raises(ValueError, match="latched"):
        gate.validate_command(twist, raw=raw)
    assert gate._deadline == 0


def test_deadman_and_revocation_are_terminal(setup_gate):
    gate, raw, clock, _ = setup_gate
    gate.refresh_lease(raw)
    clock[0] += 250_000_000
    with pytest.raises(ValueError, match="LEASE_EXPIRED"):
        gate.refresh_lease(raw)
    clock[0] -= 250_000_000
    with pytest.raises(ValueError, match="latched"):
        gate.refresh_lease(raw)


def test_normal_revoke_never_allows_old_action_to_resume(setup_gate):
    gate, raw, _, _ = setup_gate
    gate.refresh_lease(raw)
    gate.revoke()
    with pytest.raises(ValueError, match="REVOKED"):
        gate.validate_command(command(), raw=raw)


def test_collection_cannot_cross_lease_deadline_even_if_new_proof_would_be_valid(setup_gate):
    gate, raw, clock, local = setup_gate
    gate.refresh_lease(raw)

    def slow_collection():
        clock[0] += 250_000_000
        return local

    gate._collect_local = slow_collection
    with pytest.raises(ValueError, match="EXPIRED_DURING_COLLECTION"):
        gate.validate_command(command(), raw=raw)


def test_authenticated_cursor_barrier_and_old_deadline_rechecked(setup_gate):
    gate, raw, clock, local = setup_gate
    calls = []

    def collect(sample):
        calls.append(sample)
        return local

    gate._collect_at_least_source = collect
    gate.refresh_lease(raw)
    assert calls == [25]

    def delayed(sample):
        assert sample == 25
        clock[0] += 250_000_000
        return local

    gate._collect_at_least_source = delayed
    with pytest.raises(ValueError, match="EXPIRED_DURING_COLLECTION"):
        gate.refresh_lease(raw)


def test_tampered_proof_cannot_request_cursor_wait(setup_gate):
    gate, raw, _, _ = setup_gate

    def forbidden(sample):
        pytest.fail("unauthenticated proof reached independent cursor barrier")

    gate._collect_at_least_source = forbidden
    with pytest.raises(ValueError, match="latched"):
        gate.refresh_lease(raw.replace(b'"source_sample":25', b'"source_sample":26'))
