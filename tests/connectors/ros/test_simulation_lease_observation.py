"""Independent status observation contracts, never proof of physical stop."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def observation(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_lease_observation")
    status = {
        "schema_version": "rosclaw.sim_sink_status.v1",
        "run_id": "synthetic-world",
        "body_snapshot_hash": "a" * 64,
        "monotonic_ns": 10_000_000_000,
        "lease_expires_ns": 0,
        "terminal": False,
        "lease_updates": 0,
        "received_commands": 0,
        "published_velocity_messages": 0,
        "stop_zero_messages": 0,
        "action_id": None,
        "daemon_instance_id": None,
    }
    path = tmp_path / "status.json"
    path.write_text(json.dumps(status))
    reader = module.ExternalSimulationLeaseObservation(
        tmp_path, filename="status.json", run_id="synthetic-world", body_snapshot_hash="a" * 64
    )
    return reader, path, status


def test_observed_closed_lease_is_not_a_physical_stop_claim(observation):
    reader, _, status = observation
    result = reader.read(now_monotonic_ns=status["monotonic_ns"])
    assert result["known"] is True and result["lease_enabled"] is False
    assert "physical_stop_verified" not in result and "execution_authority" not in result


def test_fresh_active_then_expired_status_observes_actual_deadline(observation):
    reader, path, status = observation
    status.update(
        lease_expires_ns=10_200_000_000,
        lease_updates=1,
        action_id="synthetic-action",
        daemon_instance_id="daemon-synthetic",
    )
    path.write_text(json.dumps(status))
    assert reader.read(now_monotonic_ns=10_100_000_000)["lease_enabled"] is True
    assert reader.read(now_monotonic_ns=10_200_000_000)["lease_enabled"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "wrong"),
        ("body_snapshot_hash", "b" * 64),
        ("monotonic_ns", 10_000_000_001),
        ("monotonic_ns", 9_749_999_999),
        ("lease_expires_ns", 10_250_000_001),
        ("terminal", 1),
        ("lease_updates", True),
        ("received_commands", -1),
        ("schema_version", "unknown"),
        ("unexpected_approval", True),
    ],
)
def test_malformed_or_changed_status_is_unknown_and_cannot_recover(observation, field, value):
    reader, path, status = observation
    original = dict(status)
    status[field] = value
    path.write_text(json.dumps(status))
    result = reader.read(now_monotonic_ns=10_000_000_000)
    assert result["known"] is False and result["lease_enabled"] is None
    path.write_text(json.dumps(original))
    assert reader.read(now_monotonic_ns=10_000_000_000)["known"] is False


def test_missing_file_or_symlink_does_not_prove_revoke(observation, tmp_path):
    reader, path, status = observation
    original = tmp_path / "other.json"
    path.rename(original)
    path.symlink_to(original)
    result = reader.read(now_monotonic_ns=status["monotonic_ns"])
    assert result["known"] is False and result["lease_enabled"] is None


def test_status_clock_rollback_is_not_hidden_by_receive_freshness(observation):
    reader, path, status = observation
    assert reader.read(now_monotonic_ns=10_000_000_000)["known"]
    status["monotonic_ns"] -= 1
    path.write_text(json.dumps(status))
    assert not reader.read(now_monotonic_ns=10_000_000_001)["known"]


def test_terminal_sink_cannot_still_claim_active_lease(observation):
    reader, path, status = observation
    status.update(terminal=True, lease_expires_ns=10_100_000_000)
    path.write_text(json.dumps(status))
    assert reader.read(now_monotonic_ns=10_000_000_000)["lease_enabled"] is None
