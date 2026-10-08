"""Owned process lifecycle and original fault retention, without ROS or World."""

import importlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("backend_stack")


def test_source_fault_withdraws_observer_preserves_independent_processes(module, tmp_path):
    children = module.OwnedStackChildren(tmp_path, time.monotonic() + 20)
    # Plain Python sleepers are not simulated World/robot evidence.
    argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    world = children.start("backend-gazebo", argv)
    observer = children.start("backend-independent-observer", argv)
    witness = children.start("backend-witness", argv)
    actor = children.start("backend-sim-actuator", argv)
    try:
        latch = module.RuntimeFaultLatch(tmp_path, children)
        original = b'{"snapshot":{"source_fault":"missing source"}}\n'
        latch.fail("source rejected", original_observation=original)
        observer.wait(timeout=3)
        assert world.poll() is None and witness.poll() is None and actor.poll() is None
        stored = latch.path.read_bytes()
        latch.fail("later error", world_alive=False)
        assert latch.path.read_bytes() == stored
        record = json.loads(stored)
        import base64

        assert base64.b64decode(record["original_observation_base64"]) == original
        assert record["physical_stop_proof"] == "NOT_MEASURED"
        assert record["physical_acceptance"] == "FAILED"
        assert record["authorization"] is False
    finally:
        children.close()
    assert all(child.poll() is not None for _, child in children.children)


def test_world_exit_cannot_be_claimed_as_physical_stop(module, tmp_path):
    children = module.OwnedStackChildren(tmp_path, time.monotonic() + 10)
    child = children.start("backend-gazebo", [sys.executable, "-c", "raise SystemExit(7)"])
    try:
        child.wait(timeout=3)
        module.RuntimeFaultLatch(tmp_path, children).fail("World lost", world_alive=False)
        record = json.loads((tmp_path / "backend-stack-fault.json").read_text())
        assert record["failed_dependencies"] == [{"name": "backend-gazebo", "returncode": 7}]
        assert record["physical_stop_proof"] == "MISSING_STOP_PROOF"
    finally:
        children.close()


def test_deadline_is_not_reset_by_starting_dependencies(module, tmp_path):
    deadline = time.monotonic() - 1
    children = module.OwnedStackChildren(tmp_path, deadline)
    with pytest.raises(ValueError, match="deadline exhausted"):
        children.start("forbidden", [sys.executable, "-c", "raise SystemExit(0)"])
    assert children.children == [] and list(tmp_path.iterdir()) == []
    assert module.remaining_seconds(time.monotonic() + 100, 3) == 3


def test_world_is_signalled_after_owned_sources_flush(module, tmp_path, monkeypatch):
    events = []

    class Child:
        def __init__(self, pid):
            self.pid = pid

        def wait(self, timeout):
            events.append(("wait", self.pid))

    monkeypatch.setattr(os, "killpg", lambda pid, sig: events.append((sig, pid)))
    children = module.OwnedStackChildren(tmp_path, time.monotonic() + 10)
    children.children = [("backend-gazebo", Child(1)), ("source", Child(2))]
    children.close()
    assert events == [(signal.SIGINT, 2), ("wait", 2), (signal.SIGINT, 1), ("wait", 1)]
