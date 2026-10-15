"""Actual response semantics and isolated client cache, no ROS imports."""

import importlib
from concurrent.futures import Future
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_persistent_lease_bridge")


class Node:
    types = ["std_srvs/srv/SetBool"]

    def resolve_service_name(self, service):
        return service

    def get_service_names_and_types(self):
        return [("/sim/lease", self.types)]


class Client:
    def __init__(self):
        self.calls = []
        self.available = True
        self.pending = False
        self.futures = []

    def wait_for_service(self, timeout):
        return self.available

    def call_async(self, args):
        self.calls.append(args)
        f = Future()
        self.futures.append(f)
        if not self.pending:
            f.set_result({"success": args["data"], "message": "original"})
        return f


def wrapper(module):
    made = []
    audit = []
    other = []

    def create(node, service):
        c = Client()
        made.append(c)
        return c

    def upstream(*args):
        other.append(args)
        return {"original_upstream": True}

    return (
        module.PersistentLeaseService(
            "/sim/lease",
            upstream,
            create_client=create,
            prepare_request=lambda v: dict(v),
            extract_values=lambda v: dict(v),
            audit=audit.append,
        ),
        made,
        audit,
        other,
    )


def test_one_client_real_responses_including_refusal_and_revoke(module):
    bridge, made, audit, _ = wrapper(module)
    node = Node()
    assert bridge(node, "/sim/lease", {"data": True}) == {"success": True, "message": "original"}
    assert bridge(node, "/sim/lease", {"data": False}) == {"success": False, "message": "original"}
    assert len(made) == 1 and made[0].calls == [{"data": True}, {"data": False}]
    assert sum(v["event"] == "PERSISTENT_LEASE_CLIENT_CREATED" for v in audit) == 1
    assert len([v for v in audit if v["event"] == "ACTUAL_LEASE_SERVICE_RESPONSE"]) == 2


def test_other_service_uses_unmodified_upstream(module):
    bridge, made, _, other = wrapper(module)
    assert bridge(Node(), "/other", {"any": "original"}) == {"original_upstream": True}
    assert not made and len(other) == 1


def test_cache_is_scoped_to_the_actual_bridge_node(module):
    bridge, made, _, _ = wrapper(module)
    first = Node()
    second = Node()
    bridge(first, "/sim/lease", {"data": True})
    bridge(second, "/sim/lease", {"data": True})
    assert len(made) == 2


@pytest.mark.parametrize("args", [None, [], {"data": 1}, {"data": True, "extra": True}])
def test_request_shape_cannot_expand_lease_protocol(module, args):
    bridge, made, _, _ = wrapper(module)
    with pytest.raises(ValueError, match="SetBool"):
        bridge(Node(), "/sim/lease", args)
    assert not made


def test_changed_type_and_lost_service_never_reuse_cached_success(module):
    bridge, made, _, _ = wrapper(module)
    node = Node()
    bridge(node, "/sim/lease", {"data": True})
    node.types = ["other/srv/Type"]
    with pytest.raises(ValueError, match="type"):
        bridge(node, "/sim/lease", {"data": True})
    node.types = ["std_srvs/srv/SetBool"]
    made[0].available = False
    with pytest.raises(ValueError, match="unavailable"):
        bridge(node, "/sim/lease", {"data": True})
    assert len(made[0].calls) == 1


def test_response_timeout_cancels_actual_future_without_fake_ack(module):
    bridge, made, audit, _ = wrapper(module)
    node = Node()
    bridge(node, "/sim/lease", {"data": False})
    made[0].pending = True
    with pytest.raises(RuntimeError, match="actual lease service response"):
        bridge(node, "/sim/lease", {"data": True}, server_response_timeout=0.001)
    assert made[0].futures[-1].cancelled()
    assert audit[-1]["event"] == "LEASE_SERVICE_RESPONSE_TIMEOUT"
