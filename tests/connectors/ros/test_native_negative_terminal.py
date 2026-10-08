"""Real migration/receipt contracts; no model/ROS/physical negative episode."""

import hashlib
import importlib.util
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.kernel.contracts import ExecutionReceipt

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def negative(tmp_path):
    root = tmp_path
    db_path = root / "home/agentd/missions.db"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as db:
        for name in ("026_task_kernel_sqlite.sql", "018_action_txns_sqlite.sql"):
            db.executescript((REPO / "src/rosclaw/storage/migrations" / name).read_text())
        db.execute(
            "insert into tasks(task_id,mission_id,root_goal,mode,body_id,workspace_path,state,created_at,updated_at,terminal_reason) values(?,?,?,?,?,?,?,?,?,?)",
            (
                "root",
                "native_mission",
                "完成整个房间清扫。",
                "SIMULATION",
                "body",
                str(root),
                "FAILED",
                "now",
                "now",
                "actual_failure",
            ),
        )
        db.execute(
            "insert into action_txns(txn_id,idempotency_key,request_hash,pi_session_id,mission_id,context_lease_id,context_revision,body_hash,mode,capability_id,arguments_hash,risk_tier,action_id,state,created_at,expires_at) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "txn",
                "idem",
                "req",
                "pi",
                "native_mission",
                "ctx",
                1,
                "body_hash",
                "SIMULATION",
                "coverage.execute",
                "args",
                "MEDIUM",
                "failed_action",
                "FAILED",
                "now",
                "later",
            ),
        )
    (root / "execution_config.json").write_text(
        json.dumps({"body_id": "body", "body_snapshot_hash": "body_hash"})
    )
    actions = root / "actions"
    actions.mkdir()
    failure = actions / "source.failed.json"
    failure.write_text(
        '{"action_id":"failed_action","verification_status":"NOT_VERIFIED","error":"actual source fault"}'
    )
    receipt = ExecutionReceipt(
        action_id="failed_action",
        trace_id="trace",
        mode="SIMULATION",
        body_id="body",
        body_snapshot_hash="body_hash",
        capability_id="coverage.execute",
        final_state="FAILED",
        evidence_level="REQUESTED",
        verification_result={
            "failure_artifact": {
                "path": str(failure),
                "sha256": hashlib.sha256(failure.read_bytes()).hexdigest(),
            }
        },
    ).to_dict()
    result = {"action_id": "failed_action", "receipt": receipt}
    spec = importlib.util.spec_from_file_location(
        "negative_native_test", REPO / "integrations/ros_probe/acceptance/native.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = SimpleNamespace(get_execution_receipt=lambda _: deepcopy(result))
    return root, db_path, module, client, result


@pytest.mark.parametrize("state", ["FAILED", "BLOCKED"])
def test_real_negative_terminal_and_canonical_receipt_are_read_without_state_promotion(
    negative, state
):
    root, path, module, client, _ = negative
    with sqlite3.connect(path) as db:
        db.execute("update tasks set state=?", (state,))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = module.negative_native_progress(root, "D5", client)
    assert result["task"]["state"] == state and not result["task_state_modified"]
    assert result["physical_acceptance"] == "NOT_VERIFIED"
    assert result["requires_independent_stop_and_closed_source_and_scenario_validation"]
    assert result["canonical_receipts"][0]["receipt"]["final_state"] == "FAILED"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("state", ["RUNNING", "WAITING_INPUT", "RECOVERING", "CANCELLED"])
def test_missing_negative_root_terminal_remains_pending(negative, state):
    root, path, module, client, _ = negative
    with sqlite3.connect(path) as db:
        db.execute("update tasks set state=?", (state,))
    result = module.negative_native_progress(root, "D4", client)
    assert result["status"] == "PENDING_REAL_TERMINAL_STATE" and result["task_state"] == state


def test_negative_success_is_an_acceptance_failure(negative):
    root, path, module, client, _ = negative
    with sqlite3.connect(path) as db:
        db.execute("update tasks set state='SUCCEEDED'")
    with pytest.raises(RuntimeError, match="incorrectly reports success"):
        module.negative_native_progress(root, "D5", client)


@pytest.mark.parametrize(
    "fault", ["receipt_action", "body", "mode", "state", "artifact_hash", "artifact_escape"]
)
def test_negative_refuses_wrong_canonical_identity_or_retained_source(negative, fault):
    root, _, module, client, result = negative
    receipt = result["receipt"]
    if fault == "receipt_action":
        receipt["action_id"] = "foreign"
    elif fault == "body":
        receipt["body_snapshot_hash"] = "foreign"
    elif fault == "mode":
        receipt["execution_mode"] = "REAL"
    elif fault == "state":
        receipt["final_state"] = "COMPLETED"
    elif fault == "artifact_hash":
        (root / "actions/source.failed.json").write_text("changed")
    else:
        outside = root / "outside.failed.json"
        outside.write_text("foreign")
        receipt["verification_result"]["failure_artifact"]["path"] = str(outside)
    with pytest.raises(ValueError):
        module.negative_native_progress(root, "D5", client)


@pytest.mark.parametrize(
    "field,value",
    [("mission_id", "foreign"), ("state", "RECEIPT_PENDING"), ("body_hash", "foreign")],
)
def test_unmatched_or_unclosed_transaction_does_not_become_negative_acceptance(
    negative, field, value
):
    root, path, module, client, _ = negative
    with sqlite3.connect(path) as db:
        # The column names are this fixed test parametrization, not user input.
        db.execute(f"update action_txns set {field}=?", (value,))
    result = module.negative_native_progress(root, "D5", client)
    assert result["status"] == "PENDING_REAL_FAILED_COVERAGE_RECEIPT"


@pytest.mark.parametrize("fault", ["case", "config", "bundle", "extra_root", "goal", "body"])
def test_negative_requires_typed_isolated_single_root_sources(negative, fault):
    root, path, module, client, _ = negative
    case = "D5"
    if fault == "case":
        case = []
    elif fault == "config":
        (root / "execution_config.json").write_text("[]")
    elif fault == "bundle":
        client.get_execution_receipt = lambda _: []
    else:
        with sqlite3.connect(path) as db:
            if fault == "extra_root":
                db.execute(
                    "insert into tasks(task_id,root_goal,mode,workspace_path,state,created_at,updated_at) values('second','other','SIMULATION','workspace','FAILED','now','now')"
                )
            elif fault == "goal":
                db.execute("update tasks set root_goal='different'")
            else:
                db.execute("update tasks set body_id='foreign'")
    with pytest.raises(ValueError):
        module.negative_native_progress(root, case, client)


@pytest.mark.parametrize(
    "fault", [None, "body", "mission", "run", "hash", "accounting", "state", "action", "case"]
)
def test_actual_canonical_blocked_partial_temporal_artifact_is_negative_only(negative, fault):
    from rosclaw.connectors.ros.verification.mission import replay_coverage
    from tests.connectors.ros.test_temporal_mission_runtime import evidence, sample

    root, db, module, client, result = negative
    ev = evidence([sample(0, 0, 0.005, [0]), sample(0.1, 1, 0.015, [0])])
    ev.update(body_snapshot_hash="body_hash", action_ids=["failed_action"])
    config = {
        "body_id": "body",
        "body_snapshot_hash": "body_hash",
        "dynamic_fixture_admission": {"mission_id": "mission", "run_id": "run"},
    }
    (root / "execution_config.json").write_text(json.dumps(config))
    coverage, temporal = replay_coverage(ev)
    if fault == "body":
        ev["body_snapshot_hash"] = "foreign"
    elif fault == "mission":
        ev["mission_id"] = "foreign"
    elif fault == "run":
        ev["occupancy_binding"]["run_id"] = "foreign"
    elif fault == "action":
        ev["action_ids"] = ["foreign"]
    path = root / "actions/rosevidence_partial.json"
    path.write_text(json.dumps(ev))
    receipt = result["receipt"]
    receipt["final_state"] = "COMPLETED" if fault == "state" else "BLOCKED"
    receipt["verification_result"] = {
        "mission_id": "mission",
        "coverage_ratio": coverage["coverage_ratio"],
        "time_paired_accounting": {} if fault == "accounting" else temporal,
        "evidence_artifact": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }
    if fault == "hash":
        path.write_text(path.read_text() + " ")
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    if fault:
        with pytest.raises(ValueError):
            module.negative_native_progress(root, "D5" if fault == "case" else "D4", client)
    else:
        answer = module.negative_native_progress(root, "D4", client)
        assert answer["physical_acceptance"] == "NOT_VERIFIED"
        assert answer["task"]["state"] == "FAILED"
        assert answer["canonical_receipts"][0]["receipt"]["final_state"] == "BLOCKED"
        assert coverage["coverage_ratio"] == 0.5
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
