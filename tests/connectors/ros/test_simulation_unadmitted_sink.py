"""Unadmitted probe boundary contracts; no real ROS or lease request."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_unadmitted_sink")
    sink = {
        "run_id": "run",
        "body_snapshot_hash": "body",
        "terminal": False,
        "action_id": None,
        "daemon_instance_id": None,
        "lease_expires_ns": 0,
        "lease_updates": 0,
        "received_commands": 0,
        "published_velocity_messages": 0,
    }
    (tmp_path / "execution-control").mkdir()
    return module, tmp_path, sink


def run(fixture):
    module, root, sink = fixture
    (root / "governed-sink-status.json").write_text(json.dumps(sink))
    return module.require_unadmitted_sink(root, run_id="run", body_snapshot_hash="body")


def test_never_admitted_idle_sink_is_explicitly_identified(fixture):
    assert run(fixture) == fixture[-1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "other"),
        ("body_snapshot_hash", "other"),
        ("terminal", True),
        ("action_id", "old"),
        ("daemon_instance_id", "old"),
        ("lease_expires_ns", 1),
        ("lease_updates", 1),
        ("received_commands", 1),
        ("published_velocity_messages", 1),
        ("lease_updates", False),
    ],
)
def test_any_prior_admission_or_wrong_context_prevents_probe(fixture, field, value):
    fixture[-1][field] = value
    with pytest.raises(ValueError, match="never-admitted"):
        run(fixture)


@pytest.mark.parametrize("name", ["admission.json", "proof.json"])
@pytest.mark.parametrize("symlink", [False, True])
def test_any_admission_proof_or_dangling_symlink_prevents_probe(fixture, name, symlink):
    _, root, _ = fixture
    path = root / "execution-control" / name
    if symlink:
        path.symlink_to(root / "nonexistent")
    else:
        path.write_text("{}")
    with pytest.raises(ValueError, match="any operator admission"):
        run(fixture)


@pytest.mark.parametrize("fault", [None, "authentication", "published", "accepted"])
def test_rejected_run_requires_identity_failure_and_zero_prior_admission(fixture, fault):
    module, root, sink = fixture
    sink["terminal"] = True
    reason = "operator context Body/World/run/source differs from independent facts"
    if fault == "authentication":
        reason = "SDK operator admission authentication failed"
    if fault == "published":
        sink["published_velocity_messages"] = 1
    if fault == "accepted":
        sink["daemon_instance_id"] = "accepted-instance"
    (root / "governed-sink-terminal.json").write_text(
        json.dumps({"reason": reason, "status": sink})
    )
    if fault:
        with pytest.raises(ValueError, match="before any SDK admission"):
            module.require_rejected_run_sink(root, run_id="run", body_snapshot_hash="body")
    else:
        assert (
            module.require_rejected_run_sink(root, run_id="run", body_snapshot_hash="body") == sink
        )
