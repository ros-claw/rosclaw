"""Source contracts for actual read-only lifecycle polling; no ROS Node."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = (
    Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance/lifecycle_readiness.py"
)
spec = importlib.util.spec_from_file_location("fixture_lifecycle_readiness", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def snapshot():
    return {
        "schema_version": m.SCHEMA,
        "source": "actual_read_only_GetState_responses",
        "responses": {
            name: {
                "service": "/" + name + "/get_state",
                "state_id": 3,
                "state_label": "active",
                "received_monotonic_sec": 99.5,
            }
            for name in m.REQUIRED_NODES
        },
        "ready": True,
    }


def test_all_seven_fresh_direct_active_responses_required():
    assert m.readiness(snapshot(), now=100)


@pytest.mark.parametrize(
    "key,value",
    [
        ("state_id", True),
        ("state_id", 2),
        ("state_label", "inactive"),
        ("received_monotonic_sec", 98),
        ("received_monotonic_sec", 101),
        ("received_monotonic_sec", float("nan")),
        ("service", "/wrong/get_state"),
    ],
)
def test_self_declared_readiness_cannot_hide_stale_or_foreign_response(key, value):
    value_snapshot = snapshot()
    value_snapshot["responses"]["coverage_server"][key] = value
    assert not m.readiness(value_snapshot, now=100)


def test_missing_actual_response_fails_closed():
    value = snapshot()
    value["responses"]["coverage_server"] = None
    assert not m.readiness(value, now=100)


class Future:
    def __init__(self):
        self.complete = False
        self.cancelled = False

    def done(self):
        return self.complete

    def result(self):
        return SimpleNamespace(current_state=SimpleNamespace(id=3, label="active"))

    def cancel(self):
        self.cancelled = True


class Client:
    def __init__(self):
        self.future = None
        self.calls = 0
        self.removed = []

    def service_is_ready(self):
        return True

    def call_async(self, request):
        self.calls += 1
        self.future = Future()
        return self.future

    def remove_pending_request(self, future):
        self.removed.append(future)


def test_bounded_poll_uses_only_getstate_then_timeout_revokes_ready(monkeypatch, tmp_path):
    clients = {}

    def create(service_type, endpoint):
        assert endpoint.endswith("/get_state")
        clients[endpoint] = Client()
        return clients[endpoint]

    clock = [100.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])
    probe = m.LifecycleProbe(
        SimpleNamespace(create_client=create), SimpleNamespace(Request=lambda: object()), tmp_path
    )
    try:
        probe.poll()
        assert len(clients) == 7 and all(c.calls == 1 for c in clients.values())
        probe.poll()
        assert all(c.calls == 1 for c in clients.values())
        for c in clients.values():
            c.future.complete = True
        clock[0] += 0.25
        probe.poll()
        assert m.readiness(
            __import__("json").loads((tmp_path / "lifecycle-readiness.json").read_text()),
            now=clock[0],
        )
        old = [c.future for c in clients.values()]
        clock[0] += 1.25
        probe.poll()
        assert not m.readiness(
            __import__("json").loads((tmp_path / "lifecycle-readiness.json").read_text()),
            now=clock[0],
        )
        assert all(f.cancelled for f in old)
        assert all(len(c.removed) == 1 for c in clients.values())
    finally:
        probe.close()
