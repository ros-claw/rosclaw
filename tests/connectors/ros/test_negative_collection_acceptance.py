"""Synthetic D5 source contracts and actual SQLite migrations; no SDK/World run."""

import base64
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from tests.connectors.ros import test_negative_stop_evidence as brush_contract
from tests.connectors.ros.test_native_negative_terminal import negative  # noqa: F401


@pytest.fixture
def closed(negative, monkeypatch):  # noqa: F811
    root, _, native, client, _ = negative
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("negative_collection_acceptance")
    terminal = native.negative_native_progress(root, "D5", client)
    (root / "native-negative-terminal.json").write_text(json.dumps(terminal))
    monkeypatch.setitem(brush_contract.BINDING, "body_snapshot_hash", "body_hash")
    brush_path, stop = brush_contract.fixture(root)
    renamed = root / "brush-events-synthetic.jsonl"
    brush_path.rename(renamed)
    Path(str(brush_path) + ".summary.json").rename(Path(str(renamed) + ".summary.json"))
    (root / "brush_binding.json").write_text(json.dumps(brush_contract.BINDING))
    plan = {"run_id": "run", "body_snapshot_hash": "body_hash", "constraint_policy_hash": "policy"}
    (root / "backend-stack-source-plan.json").write_text(json.dumps(plan))
    policy = {
        **plan,
        "schema_version": "rosclaw.collection_pause_fixture.v1",
        "source": "operator_controlled_SIM_collection_fault",
        "approved": True,
        "pause_wall_sec": 2,
    }
    policy_raw = json.dumps(policy).encode()
    policy_sha = hashlib.sha256(policy_raw).hexdigest()
    values = {
        "policy": policy,
        "request": {"fixture_policy_sha256": policy_sha},
        "requested": {
            "fixture_policy_sha256": policy_sha,
            "fixture_policy": policy,
            "observer_pid": 12345,
            "World_or_robot_signaled": False,
        },
        "applied": {
            "fixture_policy_sha256": policy_sha,
            "observer_pid": 12345,
            "actual_linux_process_state": "T",
            "confirmed_monotonic_sec": 100,
        },
        "before": {
            "policy_sha256": policy_sha,
            "backend_projection": {"snapshot": {"sampled_monotonic_sec": 99.9}},
        },
        "projection": {"snapshot": {"sampled_monotonic_sec": 100}},
    }
    names = {
        "policy": "policy",
        "request": "request",
        "requested": "record",
        "applied": "applied",
        "before": "before",
    }
    originals = {}
    for key, value in values.items():
        raw = json.dumps(value).encode()
        originals[key] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "original_base64": base64.b64encode(raw).decode(),
        }
        if key in names:
            (root / f"backend-collection-pause-{names[key]}.json").write_bytes(raw)
    loss = {
        "schema_version": "rosclaw.collection_pause_source_loss.v1",
        "observed_monotonic_sec": 100.3,
        "collector_stale_at_first_fault": True,
        "fault_observed_after_actual_pause": True,
        "authorization": False,
        "physical_stop_proof": "NOT_MEASURED",
        "originals": originals,
        "post_fault_contact_state": "UNKNOWN_REQUIRES_INDEPENDENT_SOURCE",
    }
    (root / "backend-collection-source-loss.json").write_text(json.dumps(loss))
    observer = root / "backend-observer"
    observer.mkdir()
    audit = observer / "backend-observation-events.jsonl"
    audit.write_text("synthetic prefix source; not an SDK tape\n")
    summary = Path(str(audit) + ".summary.json")
    summary.write_text("{}")
    prefix = {
        "authorization": False,
        "physical_acceptance": "NOT_VERIFIED",
        "constraint_policy_hash": "policy",
        "source_window": "PREFIX_BEFORE_REGISTERED_SOURCE_LOSS",
        "post_prefix_source_health": "UNKNOWN",
        "original_service_wire_required": True,
        "original_service_SDK_wire_replays": 1,
        "spatial_source_join_required": True,
        "completed_exact_scene_joins": 1,
        "robot_collision_count": 0,
        "probe_completed_cache_cycles": 1,
        "healthy_prefix_monotonic_sec": 99.9,
        "original_source_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    }
    (root / "closed-collection-prefix-source.json").write_text(json.dumps(prefix))
    return module, root, stop


def test_d5_negative_requires_real_terminal_and_independent_stop_sources(closed):
    module, root, stop = closed
    result = module.validate_collection_negative(root, stop)
    assert result["task_kernel_succeeded"] is False
    assert result["post_loss_robot_contacts"] == "UNKNOWN"
    assert result["other_D5_subcases"] == "NOT_RUN"


@pytest.mark.parametrize(
    "fault", ["stale_flag", "clock", "original", "source_hash", "success", "pose"]
)
def test_d5_negative_cannot_promote_unknown_missing_or_changed_evidence(closed, fault):
    module, root, stop = closed
    if fault in {"stale_flag", "clock", "original"}:
        path = root / "backend-collection-source-loss.json"
        loss = json.loads(path.read_bytes())
        if fault == "stale_flag":
            loss["collector_stale_at_first_fault"] = False
        elif fault == "clock":
            loss["observed_monotonic_sec"] = 100.01
        else:
            loss["originals"]["applied"]["sha256"] = "changed"
        path.write_text(json.dumps(loss))
    elif fault == "source_hash":
        (root / "backend-observer/backend-observation-events.jsonl").write_text("changed")
    elif fault == "success":
        (root / "actions/illegal.verification.json").write_text("{}")
    else:
        stop["samples"][-1]["x"] = 1
    with pytest.raises(ValueError):
        module.validate_collection_negative(root, stop)
