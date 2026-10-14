"""Operator fault scope cannot silently drift between SDK runner and actor."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_execution_fault_policy")
    pins = {
        "run_id": "original-run",
        "body_snapshot_hash": "a" * 64,
        "world_sha256": "b" * 64,
        "action_id": "original-first-action",
    }
    policy = {
        "schema_version": "rosclaw.governed_sim_fault.v1",
        "case": "daemon_sigkill",
        "phase": "during_motion",
        **pins,
    }
    return module, pins, policy


def test_original_policy_explicitly_binds_first_action(fixture):
    module, pins, policy = fixture
    raw = json.dumps(policy).encode()
    assert (
        module.parse_daemon_sigkill_policy(
            raw, expected_sha256=hashlib.sha256(raw).hexdigest(), **pins
        )
        == policy
    )


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "body_snapshot_hash",
        "world_sha256",
        "action_id",
        "case",
        "phase",
        "schema_version",
    ],
)
def test_mismatched_operator_scope_is_rejected(fixture, field):
    module, pins, policy = fixture
    policy[field] = "other"
    raw = json.dumps(policy).encode()
    with pytest.raises(ValueError, match="closed Body/run/World/action"):
        module.parse_daemon_sigkill_policy(
            raw, expected_sha256=hashlib.sha256(raw).hexdigest(), **pins
        )


@pytest.mark.parametrize("fault", ["extra", "missing", "duplicate", "sha", "oversize"])
def test_closed_original_policy_and_hash_are_required(fixture, fault):
    module, pins, policy = fixture
    if fault == "extra":
        policy["approved"] = True
    if fault == "missing":
        del policy["action_id"]
    raw = json.dumps(policy).encode()
    if fault == "duplicate":
        raw = raw[:-1] + b',"case":"daemon_sigkill"}'
    if fault == "oversize":
        raw += b" " * 4096
    digest = hashlib.sha256(raw).hexdigest()
    if fault == "sha":
        digest = "f" * 64
    with pytest.raises(ValueError):
        module.parse_daemon_sigkill_policy(raw, expected_sha256=digest, **pins)
