"""Synthetic SDK lifecycle RPC contracts; never actual fault acceptance."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_owned_lifecycle_fault")
    compiled = tmp_path / "generic-compiled"
    compiled.mkdir()
    (compiled / "navigation-requirements.json").write_text(
        json.dumps({"lifecycle_nodes": {"collision_monitor": "/arbitrary/safety_node"}})
    )
    services = [
        {"name": "/arbitrary/safety_node/" + suffix, "srv_type": "lifecycle_msgs/srv/" + kind}
        for suffix, kind in [("get_state", "GetState"), ("change_state", "ChangeState")]
    ]
    raw = json.dumps({"graph": {"services": services}}).encode()
    (tmp_path / "native.json").write_bytes(raw)
    monkeypatch.setattr(
        module,
        "read_current_native_pointer",
        lambda root: ({"file": "native.json", "sha256": hashlib.sha256(raw).hexdigest()}, 0),
    )

    class Transport:
        state = 3
        ok = True
        acknowledge = True

        def __init__(self):
            self.calls = []

        def call_service(self, service, request, *, service_type, timeout_sec):
            self.calls.append((service, request, service_type, timeout_sec))
            if service.endswith("get_state"):
                values = {"current_state": {"id": self.state}}
            else:
                if self.acknowledge:
                    self.state = {4: 2, 3: 3}[request["transition"]["id"]]
                values = {"success": self.acknowledge}
            return SimpleNamespace(
                ok=self.ok,
                data={"values": values},
                error=None if self.ok else "synthetic transport failure",
            )

    transport = Transport()
    actor = module.OwnedCollisionMonitorFault(
        tmp_path, transport, run_id="run", body_snapshot_hash="body"
    )
    return actor, transport, tmp_path


def test_only_declared_lifecycle_is_changed_and_restoration_requires_terminal_context(fixture):
    actor, transport, root = fixture
    assert actor.disable()["actual_state_id"] == 2
    with pytest.raises(OSError):
        actor.restore_after_terminal_stop()
    (root / "governed-sink-status.json").write_text(
        json.dumps(
            {"terminal": True, "lease_expires_ns": 0, "run_id": "run", "body_snapshot_hash": "body"}
        )
    )
    restored = actor.restore_after_terminal_stop()
    assert restored["actual_state_id"] == 3
    transitions = [
        request["transition"]["id"]
        for _, request, _, _ in transport.calls
        if "transition" in request
    ]
    assert transitions == [4, 3]
    assert all(
        service.startswith("/arbitrary/safety_node/") and timeout == 1
        for service, _, _, timeout in transport.calls
    )


@pytest.mark.parametrize("fault", ["transport", "acknowledgement", "initial_state"])
def test_failed_exchange_or_wrong_state_never_reports_successful_disable(fixture, fault):
    actor, transport, _ = fixture
    if fault == "transport":
        transport.ok = False
    elif fault == "acknowledgement":
        transport.acknowledge = False
    else:
        transport.state = 2
    with pytest.raises(ValueError):
        actor.disable()


@pytest.mark.parametrize(
    "field,value",
    [
        ("terminal", False),
        ("lease_expires_ns", 1),
        ("run_id", "other"),
        ("body_snapshot_hash", "other"),
    ],
)
def test_restoration_cannot_operate_on_live_or_different_sink(fixture, field, value):
    actor, transport, root = fixture
    sink = {
        "terminal": True,
        "lease_expires_ns": 0,
        "run_id": "run",
        "body_snapshot_hash": "body",
        field: value,
    }
    (root / "governed-sink-status.json").write_text(json.dumps(sink))
    with pytest.raises(ValueError, match="same-context terminal"):
        actor.restore_after_terminal_stop()
    assert transport.calls == []
