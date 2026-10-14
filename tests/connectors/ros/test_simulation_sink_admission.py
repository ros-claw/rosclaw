"""Synthetic explicit SDK context contracts; not operator or motion evidence."""

import copy
import hashlib
import hmac
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from tests.connectors.ros.test_navigation_only_executor import compiled_body


@pytest.fixture
def context(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    m = importlib.import_module("simulation_sink_admission")
    names = ["CI/CD", "First Boot Acceptance", "Data Flywheel Gate"]
    p03 = {
        "status": "PASS_P03_NORMAL_INTEGRATION_CURRENT_ROS_SOURCE_DAEMON_BINDING_NOT_L2",
        "source": "a" * 40,
        "merge_commit": "b" * 40,
        "full_P03_done": True,
        "G2_done": True,
        "source_runs": [
            {"name": name, "headSha": "a" * 40, "status": "completed", "conclusion": "success"}
            for name in names
        ],
        "main_runs": [
            {"name": name, "headSha": "b" * 40, "status": "completed", "conclusion": "success"}
            for name in names + ["SeekDB 1.4 Engine Gate"]
        ],
        "actual_main": {
            "status": "PASS_P03_ACTUAL_MAIN_TWELVE_ORIGINAL_PASSIVE_DUAL_WORLDS",
            "source": "b" * 40,
            "original_Worlds": 12,
            "pinned_CLI_readonly_compiled_artifact_cases": 12,
            "raw_independent_recomputation": "PASS",
            "daemon_generation_binding": "PASS",
        },
    }
    p03_raw = json.dumps(p03).encode()
    digest = hashlib.sha256(p03_raw).hexdigest()
    binding = SimulationNavigationBinding.from_body(compiled_body())
    facts = SimpleNamespace(
        enrollment={
            "request": {"manifest_hash": "navbodyworkspace_" + "c" * 64},
            "world_sha256": "d" * 64,
            "source_pins": {"observer_epoch": "e" * 32},
        },
        enrollment_sha256="f" * 64,
    )
    payload = {
        "schema_version": "rosclaw.governed_sim_sink_admission.v1",
        "body_id": binding.body_id,
        "body_snapshot_hash": binding.body_snapshot_hash,
        "manifest_hash": facts.enrollment["request"]["manifest_hash"],
        "run_id": binding.run_id,
        "world_sha256": "d" * 64,
        "observer_epoch": "e" * 32,
        "daemon_instance_id": "daemon_synthetic",
        "daemon_pid": 123,
        "daemon_uid": 1000,
        "allowed_action_ids": ["position", "yaw"],
        "facts_enrollment_sha256": "f" * 64,
        "P03_review_sha256": digest,
        "P03_merge_commit": "b" * 40,
        "issued_ns": 10_000_000_000,
        "expires_ns": 40_000_000_000,
    }
    return m, binding, facts, p03_raw, digest, payload, bytes(range(32))


def parse(context, payload=None):
    m, binding, facts, p03_raw, digest, original, key = context
    payload = copy.deepcopy(original) if payload is None else payload
    signature = hmac.new(key, m._canonical(payload), hashlib.sha256).hexdigest()
    raw = m._canonical({"payload": payload, "hmac_sha256": signature})
    return m.authenticate_sink_admission(
        raw,
        key=key,
        binding=binding,
        facts=facts,
        p03_raw=p03_raw,
        p03_sha256=digest,
        now_ns=10_000_000_001,
    )


def test_context_binds_scope_and_accepted_historical_prerequisite(context):
    result = parse(context)
    assert result["allowed_action_ids"] == ["position", "yaw"]
    assert "permit_id" not in result and "real_authority" not in result


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "wrong"},
        {"body_snapshot_hash": "a" * 64},
        {"world_sha256": "a" * 64},
        {"observer_epoch": "a" * 32},
        {"manifest_hash": "navbodyworkspace_" + "a" * 64},
        {"facts_enrollment_sha256": "a" * 64},
        {"P03_review_sha256": "a" * 64},
        {"P03_merge_commit": "a" * 40},
        {"expires_ns": 130_000_000_001},
        {"issued_ns": 10_000_000_002},
        {"daemon_pid": True},
        {"daemon_uid": -1},
        {"allowed_action_ids": ["position", "position"]},
        {"allowed_action_ids": ["a", "b", "c"]},
        {"approved": True},
        {"REAL": True},
    ],
)
def test_signed_wrong_context_never_becomes_admission(context, changes):
    payload = {**context[5], **changes}
    with pytest.raises(ValueError):
        parse(context, payload)


@pytest.mark.parametrize(
    "fault",
    [
        "not_closed",
        "source_workflow_failed",
        "main_source_changed",
        "missing_workflow",
        "old_actual_main",
        "missing_world",
        "missing_cli",
        "raw_not_verified",
    ],
)
def test_historical_pass_or_partial_integration_cannot_satisfy_p03(context, fault):
    m, _, _, raw, _, _, _ = context
    p = json.loads(raw)
    if fault == "not_closed":
        p["full_P03_done"] = False
    elif fault == "source_workflow_failed":
        p["source_runs"][0]["conclusion"] = "failure"
    elif fault == "main_source_changed":
        p["main_runs"][0]["headSha"] = "old"
    elif fault == "missing_workflow":
        p["main_runs"].pop()
    elif fault == "old_actual_main":
        p["actual_main"]["source"] = "old"
    elif fault == "missing_world":
        p["actual_main"]["original_Worlds"] = 11
    elif fault == "missing_cli":
        p["actual_main"]["pinned_CLI_readonly_compiled_artifact_cases"] = 11
    else:
        p["actual_main"]["raw_independent_recomputation"] = "NOT_RUN"
    changed = json.dumps(p).encode()
    with pytest.raises(ValueError):
        m.verify_p03_prerequisite(changed, expected_sha256=hashlib.sha256(changed).hexdigest())


def test_tampered_prerequisite_bytes_are_not_reinterpreted_as_current_authority(context):
    m, _, _, raw, digest, _, _ = context
    with pytest.raises(ValueError, match="pinned"):
        m.verify_p03_prerequisite(raw + b" ", expected_sha256=digest)
