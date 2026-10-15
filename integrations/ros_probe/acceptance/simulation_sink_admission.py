"""Explicit operator SDK SIM context, supplementary to rosclawd authority.

The fixture operator signs a bounded context after accepted stopped G1/L1 and
actual independent daemon socket enrollment. It cannot authorize REAL, invent a
Body, replace current facts or become a canonical Session/Lease/Permit.
"""

import hashlib
import hmac
import json

from simulation_execution_proof import ExecutionProofPins, _canonical, _unique_object


def verify_p03_prerequisite(raw, *, expected_sha256):
    if (
        type(raw) is not bytes
        or not 0 < len(raw) <= 128_000
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError("independently pinned P03 closure bytes required")
    review = json.loads(raw, object_pairs_hook=_unique_object)
    if (
        type(review) is not dict
        or review.get("status")
        != "PASS_P03_NORMAL_INTEGRATION_CURRENT_ROS_SOURCE_DAEMON_BINDING_NOT_L2"
        or review.get("full_P03_done") is not True
        or review.get("G2_done") is not True
    ):
        raise ValueError("P03 actual-main and normal integration must be accepted first")
    names = {"CI/CD", "First Boot Acceptance", "Data Flywheel Gate"}
    for field, expected, source in (
        ("source_runs", names, review["source"]),
        ("main_runs", names | {"SeekDB 1.4 Engine Gate"}, review["merge_commit"]),
    ):
        rows = review[field]
        if (
            type(rows) is not list
            or len(rows) != len(expected)
            or any(type(r) is not dict for r in rows)
            or {r["name"] for r in rows} != expected
            or any(
                r["headSha"] != source or r["status"] != "completed" or r["conclusion"] != "success"
                for r in rows
            )
        ):
            raise ValueError("all original P03 source/main workflows must be accepted")
    actual = review["actual_main"]
    if (
        type(actual) is not dict
        or actual.get("status") != "PASS_P03_ACTUAL_MAIN_TWELVE_ORIGINAL_PASSIVE_DUAL_WORLDS"
        or actual.get("source") != review["merge_commit"]
        or actual.get("original_Worlds") != 12
        or actual.get("pinned_CLI_readonly_compiled_artifact_cases") != 12
        or actual.get("raw_independent_recomputation") != "PASS"
        or actual.get("daemon_generation_binding") != "PASS"
    ):
        raise ValueError("P03 actual-main raw/CLI/daemon proof missing")
    return review["merge_commit"]


def authenticate_sink_admission(raw, *, key, binding, facts, p03_raw, p03_sha256, now_ns):
    """SDK sink parser. Current ROS/source and real socket checks follow parsing."""
    if (
        type(key) is not bytes
        or len(key) != 32
        or type(raw) is not bytes
        or not 0 < len(raw) <= 16_000
    ):
        raise ValueError("bounded authenticated SDK operator context required")
    envelope = json.loads(raw, object_pairs_hook=_unique_object)
    if type(envelope) is not dict or set(envelope) != {"payload", "hmac_sha256"}:
        raise ValueError("closed authenticated SDK operator context required")
    payload = envelope["payload"]
    expected = hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()
    if type(envelope["hmac_sha256"]) is not str or not hmac.compare_digest(
        envelope["hmac_sha256"], expected
    ):
        raise ValueError("SDK operator admission authentication failed")
    keys = {
        "schema_version",
        "body_id",
        "body_snapshot_hash",
        "manifest_hash",
        "run_id",
        "world_sha256",
        "observer_epoch",
        "daemon_instance_id",
        "daemon_pid",
        "daemon_uid",
        "allowed_action_ids",
        "facts_enrollment_sha256",
        "P03_review_sha256",
        "P03_merge_commit",
        "issued_ns",
        "expires_ns",
    }
    if (
        type(payload) is not dict
        or set(payload) != keys
        or payload["schema_version"] != "rosclaw.governed_sim_sink_admission.v1"
    ):
        raise ValueError("closed SIM operator admission required")
    if (
        type(now_ns) is not int
        or now_ns <= 0
        or any(
            type(payload[k]) is not int
            for k in ("issued_ns", "expires_ns", "daemon_pid", "daemon_uid")
        )
        or not payload["issued_ns"] <= now_ns < payload["expires_ns"]
        or not 0 < payload["expires_ns"] - payload["issued_ns"] <= 120_000_000_000
        or payload["daemon_pid"] <= 0
        or payload["daemon_uid"] < 0
    ):
        raise ValueError("bounded live SDK operator admission and actual daemon identity required")
    merge = verify_p03_prerequisite(p03_raw, expected_sha256=p03_sha256)
    if payload["P03_review_sha256"] != p03_sha256 or payload["P03_merge_commit"] != merge:
        raise ValueError("SDK operator context does not bind accepted P03 closure")
    e = facts.enrollment
    if payload["facts_enrollment_sha256"] != facts.enrollment_sha256:
        raise ValueError("operator context differs from independently pinned facts enrollment")
    expected_context = {
        "body_id": binding.body_id,
        "body_snapshot_hash": binding.body_snapshot_hash,
        "manifest_hash": e["request"]["manifest_hash"],
        "run_id": binding.run_id,
        "world_sha256": e["world_sha256"],
        "observer_epoch": e["source_pins"]["observer_epoch"],
    }
    if any(payload[k] != v for k, v in expected_context.items()):
        raise ValueError("operator context Body/World/run/source differs from independent facts")
    actions = payload["allowed_action_ids"]
    if (
        type(actions) is not list
        or not 1 <= len(actions) <= 2
        or any(type(x) is not str for x in actions)
        or len(set(actions)) != len(actions)
    ):
        raise ValueError("one or two unique operator-scoped action identities required")
    for action_id in actions:
        ExecutionProofPins(
            **expected_context,
            daemon_instance_id=payload["daemon_instance_id"],
            action_id=action_id,
        )
    return dict(payload)
