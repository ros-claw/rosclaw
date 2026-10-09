"""Actual owned Linux process signals; no ROS, World or physical stop claim."""

import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("backend_stack"), importlib.import_module(
        "owned_collection_pause"
    )


def policy(tmp_path):
    plan = {
        key: key + "-fixture" for key in ("run_id", "body_snapshot_hash", "constraint_policy_hash")
    }
    source = {
        **plan,
        "schema_version": "rosclaw.collection_pause_fixture.v1",
        "source": "operator_controlled_SIM_collection_fault",
        "approved": True,
        "pause_wall_sec": 1,
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(source))
    (tmp_path / "backend-collection-pause-request.json").write_text(
        json.dumps({"fixture_policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    )
    return plan, path


def state(child):
    return Path(f"/proc/{child.pid}/stat").read_text().split(") ", 1)[1].split()[0]


def await_state(child, stopped):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if (state(child) == "T") is stopped:
            return
        time.sleep(0.01)
    pytest.fail("owned child did not reach expected Linux process state")


def test_only_owned_observer_pauses_and_resumes_on_policy_change(modules, tmp_path):
    stack, pause_module = modules
    children = stack.OwnedStackChildren(tmp_path, time.monotonic() + 15)
    argv = [sys.executable, "-c", "import time; time.sleep(20)"]
    observer = children.start("backend-independent-observer", argv)
    independent = children.start("backend-witness", argv)
    plan, path = policy(tmp_path)
    pause = pause_module.OwnedCollectionPause(tmp_path, children, plan, path)
    try:
        pause.poll()
        await_state(observer, True)
        assert state(independent) != "T"
        original = json.loads((tmp_path / "backend-collection-pause-record.json").read_text())
        assert original["observer_pid"] == observer.pid
        assert original["World_or_robot_signaled"] is False
        assert original["physical_stop_proof"] == "NOT_MEASURED"
        path.write_text("{}")
        pause.resume_at = time.monotonic() - 1
        with pytest.raises(ValueError, match="policy changed"):
            pause.poll()
        await_state(observer, False)
        resumed = json.loads((tmp_path / "backend-collection-resume-record.json").read_text())
        assert resumed["healthy_source_restored"] is False
        pause.resume()  # Idempotent cleanup does not overwrite original evidence.
    finally:
        pause.resume()
        children.close()


def test_unregistered_request_never_signals_a_child(modules, tmp_path):
    stack, pause_module = modules
    children = stack.OwnedStackChildren(tmp_path, time.monotonic() + 10)
    plan, _ = policy(tmp_path)
    pause = pause_module.OwnedCollectionPause(tmp_path, children, plan)
    with pytest.raises(ValueError, match="unregistered"):
        pause.poll()
    assert not (tmp_path / "backend-collection-pause-record.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        {"approved": False},
        {"pause_wall_sec": True},
        {"pause_wall_sec": 11},
        {"run_id": "other"},
        {"pid": 12345},
    ],
)
def test_invalid_policy_refused_before_signal(modules, tmp_path, mutation):
    stack, pause_module = modules
    plan, path = policy(tmp_path)
    source = json.loads(path.read_text())
    source.update(mutation)
    path.write_text(json.dumps(source))
    children = stack.OwnedStackChildren(tmp_path, time.monotonic() + 10)
    with pytest.raises(ValueError, match="exact bounded"):
        pause_module.OwnedCollectionPause(tmp_path, children, plan, path)


def test_missing_owned_observer_refused(modules, tmp_path):
    stack, pause_module = modules
    plan, path = policy(tmp_path)
    children = stack.OwnedStackChildren(tmp_path, time.monotonic() + 10)
    pause = pause_module.OwnedCollectionPause(tmp_path, children, plan, path)
    with pytest.raises(ValueError, match="one live owned"):
        pause.poll()
    assert not (tmp_path / "backend-collection-pause-record.json").exists()


def test_registered_pause_automatically_resumes_at_frozen_wall_duration(modules, tmp_path):
    stack, pause_module = modules
    children = stack.OwnedStackChildren(tmp_path, time.monotonic() + 10)
    observer = children.start(
        "backend-independent-observer",
        [sys.executable, "-c", "import time; time.sleep(20)"],
    )
    plan, path = policy(tmp_path)
    pause = pause_module.OwnedCollectionPause(tmp_path, children, plan, path)
    try:
        pause.poll()
        await_state(observer, True)
        original_deadline = pause.resume_at
        while time.monotonic() < original_deadline + 0.1:
            pause.poll()
            time.sleep(0.01)
        await_state(observer, False)
        assert pause.done and pause.resume_at == original_deadline
        assert observer.poll() is None
    finally:
        pause.resume()
        children.close()


def test_registered_declaration_binds_actual_constraint_before_launch(modules, tmp_path):
    stack, pause_module = modules
    plan, path = policy(tmp_path)
    declaration = json.loads(path.read_bytes())
    declaration.pop("constraint_policy_hash")
    declaration["schema_version"] = "rosclaw.collection_pause_registration.v1"
    path.write_text(json.dumps(declaration))
    generated = pause_module.materialize_registered_policy(tmp_path, plan, path)
    children = stack.OwnedStackChildren(tmp_path, time.monotonic() + 10)
    pause_module.OwnedCollectionPause(tmp_path, children, plan, generated)
    retained = json.loads(
        (tmp_path / "backend-collection-pause-registration-original.json").read_bytes()
    )
    assert retained["registration_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert retained["policy_sha256"] == hashlib.sha256(generated.read_bytes()).hexdigest()
    assert retained["authorization"] is False


def test_registration_cannot_bind_other_body_or_choose_a_pid(modules, tmp_path):
    _, pause_module = modules
    plan, path = policy(tmp_path)
    declaration = json.loads(path.read_bytes())
    declaration.pop("constraint_policy_hash")
    declaration.update(schema_version="rosclaw.collection_pause_registration.v1", pid=1234)
    path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="closed preregistered"):
        pause_module.materialize_registered_policy(tmp_path, plan, path)
    assert not (tmp_path / "backend-collection-pause-policy.json").exists()
