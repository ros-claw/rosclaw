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


@pytest.mark.parametrize("case", ["wrong_body_hash", "out_of_bounds_goal", "wrong_run_id"])
def test_before_dispatch_policy_cannot_be_used_as_motion_fault(fixture, case):
    module, pins, policy = fixture
    policy.update(case=case, phase="before_dispatch")
    raw = json.dumps(policy).encode()
    digest = hashlib.sha256(raw).hexdigest()
    assert (
        module.parse_governed_before_dispatch_fault_policy(raw, expected_sha256=digest, **pins)
        == policy
    )
    with pytest.raises(ValueError, match="closed Body/run/World/action"):
        module.parse_governed_motion_fault_policy(raw, expected_sha256=digest, **pins)


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


def test_emergency_stop_has_same_body_world_action_scope_but_is_not_sigkill(fixture):
    module, pins, policy = fixture
    policy["case"] = "emergency_stop"
    raw = json.dumps(policy).encode()
    digest = hashlib.sha256(raw).hexdigest()
    assert module.parse_governed_motion_fault_policy(raw, expected_sha256=digest, **pins) == policy
    with pytest.raises(ValueError, match="SIGKILL policy"):
        module.parse_daemon_sigkill_policy(raw, expected_sha256=digest, **pins)


def test_lease_loss_is_bound_to_actual_default_lease_owner_scope(fixture):
    module, pins, policy = fixture
    policy["case"] = "lease_loss"
    raw = json.dumps(policy).encode()
    assert (
        module.parse_governed_motion_fault_policy(
            raw, expected_sha256=hashlib.sha256(raw).hexdigest(), **pins
        )["case"]
        == "lease_loss"
    )


def test_readonly_expiry_observation_never_calls_auto_renewing_status(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    runner = importlib.import_module("governed_navigation_run")
    calls = []

    class OwnerClient:
        def get_action_status(self, *args):
            pytest.fail("lease-loss observation invoked automatic renewal")

        def renew_action_lease(self, *args):
            pytest.fail("lease-loss observation renewed canonical lease")

        def call(self, method, params):
            calls.append((method, params))
            return {"state": "RUNNING", "action_lease": {"active": True}}

    assert (
        runner.read_nonrenewing_action_status(OwnerClient(), "original-action")["state"]
        == "RUNNING"
    )
    assert calls == [("action.status", {"action_id": "original-action"})]
