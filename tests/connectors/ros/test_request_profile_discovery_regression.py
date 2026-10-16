"""Author-only offline regressions; real request/receive with a scripted socket."""

import json

import pytest
import websocket

from rosclaw.connectors.ros.discovery.graph import RosGraphDiscovery
from rosclaw.connectors.ros.discovery.rosapi_resolver import (
    RosApiDetectionError,
    RosApiResolver,
    RosVersion,
    make_ros2_profile,
)
from rosclaw.connectors.ros.transport.base import RosTransportResult
from rosclaw.connectors.ros.transport.rosbridge import RosbridgeTransport


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now


class ScriptSocket:
    connected = True

    def __init__(self, clock, frames, send_cost=0):
        self.clock = clock
        self.frames = list(frames)
        self.send_cost = send_cost
        self.sent = []
        self.timeouts = []

    def send(self, payload):
        self.sent.append(json.loads(payload))
        self.clock.now += self.send_cost

    def settimeout(self, timeout):
        self.timeout = timeout
        self.timeouts.append(timeout)

    def recv(self):
        delay, frame = self.frames.pop(0) if self.frames else (self.timeout, None)
        self.clock.now += min(delay, self.timeout)
        if delay >= self.timeout or frame is None:
            raise websocket.WebSocketTimeoutException()
        if isinstance(frame, Exception):
            raise frame
        if isinstance(frame, dict):
            frame = {**frame, "id": frame.get("id", self.sent[-1]["id"])}
        return json.dumps(frame)


def wire(monkeypatch, frames, retries=0, send_cost=0):
    clock = Clock()
    monkeypatch.setattr(
        "rosclaw.connectors.ros.transport.rosbridge.time.monotonic", clock.monotonic
    )
    sock = ScriptSocket(clock, frames, send_cost)
    transport = RosbridgeTransport(max_retries=retries)
    transport._ws = sock
    return transport, sock, clock


def success(**values):
    return {"op": "service_response", "result": True, "values": values}


@pytest.mark.parametrize("frame", [None, {"id": "other", "op": "publish"}])
def test_one_budget_across_idle_and_unrelated(monkeypatch, frame):
    transport, sock, clock = wire(monkeypatch, [(0.25, frame)] * 30, retries=2)
    result = transport.call_service("/x", {}, timeout_sec=1)
    assert not result.ok and "deadline" in result.error
    assert clock.now == pytest.approx(1)
    assert sock.timeouts == pytest.approx([1, 0.75, 0.5, 0.25])
    assert len(sock.sent) == 1


def test_progress_is_not_success_and_generated_id_matches(monkeypatch):
    transport, sock, _ = wire(
        monkeypatch,
        [(0.1, {"op": "status", "level": "info"}), (0.1, success(version=2))],
    )
    result = transport.call_service("/x", {}, timeout_sec=1)
    assert result.ok and result.data["values"] == {"version": 2}
    assert result.request_id == sock.sent[0]["id"]
    assert result.request_id.startswith("rosclaw_")
    assert len(sock.timeouts) == 2


@pytest.mark.parametrize(
    "frame,error",
    [
        ({"op": "status", "level": "error", "msg": "service permission denied"}, "permission"),
        ({"op": "status", "level": "bogus"}, "malformed"),
        ({"op": "service_response", "result": False, "values": {}}, "result:false"),
        ({"op": "service_response", "result": 1, "values": {}}, "malformed"),
        ({"op": "service_response", "values": []}, "malformed"),
        ({"op": "publish", "values": {}}, "operation"),
        ([], "malformed"),
    ],
)
def test_service_failures_are_terminal(monkeypatch, frame, error):
    transport, sock, _ = wire(monkeypatch, [(0.1, frame)], retries=2)
    result = transport.request({"op": "call_service", "service": "/x", "id": "fixed"}, 1)
    assert not result.ok and error in result.error
    assert result.request_id == "fixed"
    assert len(sock.sent) == 1


def test_nonservice_status_ack(monkeypatch):
    transport, _, _ = wire(monkeypatch, [(0.1, {"op": "status", "level": "info"})])
    assert transport.request({"op": "advertise", "topic": "/x"}, 1).ok


def test_send_duration_consumes_budget(monkeypatch):
    transport, sock, clock = wire(monkeypatch, [], send_cost=2)
    result = transport.call_service("/x", {}, timeout_sec=1)
    assert not result.ok and "deadline" in result.error
    assert clock.now == 2  # known noninterruptible send/connect boundary
    assert sock.timeouts == []


def test_retry_uses_remaining_budget_and_same_id(monkeypatch):
    transport, first, clock = wire(monkeypatch, [(0.2, RuntimeError("broken"))], retries=1)
    second = ScriptSocket(clock, [])

    def connect():
        if transport._ws is None:
            transport._ws = second
        return RosTransportResult(ok=True)

    monkeypatch.setattr(transport, "connect", connect)
    result = transport.call_service("/x", {}, timeout_sec=1)
    assert not result.ok and clock.now == pytest.approx(1)
    assert first.sent[0]["id"] == second.sent[0]["id"]
    assert second.timeouts == pytest.approx([0.8])


def test_retry_count_remains_bounded(monkeypatch):
    transport, _, _ = wire(monkeypatch, [], retries=2)
    calls = []

    def fail(message):
        calls.append(message)
        return RosTransportResult(ok=False, error="send failed")

    monkeypatch.setattr(transport, "send", fail)
    assert not transport.call_service("/x", {}, timeout_sec=1).ok
    assert len(calls) == 3


class Replies:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def call_service(self, service, args, **kwargs):
        self.calls.append((service, args))
        assert self.replies, service
        return self.replies.pop(0)


def reply(**values):
    return RosTransportResult(ok=True, data=success(**values))


def absent():
    return RosTransportResult(ok=False, error="service not found")


@pytest.mark.parametrize("version", [None, True, False, 0, -1, 3, 2.0, "2", {}, []])
def test_invalid_version_never_triggers_ros1_fallback(version):
    transport = Replies(reply(version=version), reply(version=version))
    resolver = RosApiResolver(transport)
    with pytest.raises(RosApiDetectionError):
        resolver.resolve()
    assert resolver._profile is None
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "error", ["service timeout", "service permission denied", "host not found"]
)
def test_transport_error_is_not_service_absence(error):
    fail = RosTransportResult(ok=False, error=error)
    transport = Replies(fail, fail)
    with pytest.raises(RosApiDetectionError):
        RosApiResolver(transport).resolve()
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "last",
    [
        reply(value=""),
        reply(value="humble"),
        reply(value="unknown"),
        reply(value=[]),
        reply(value="{}"),
        RosTransportResult(ok=False, error="timeout"),
    ],
)
def test_missing_or_invalid_ros1_distro_is_not_unknown_profile(last):
    transport = Replies(absent(), absent(), last)
    resolver = RosApiResolver(transport)
    with pytest.raises(RosApiDetectionError):
        resolver.resolve()
    assert resolver._profile is None


@pytest.mark.parametrize("value", ["noetic", "/opt/ros/noetic", '"noetic\\n"'])
def test_positive_ros1_evidence_and_cache(value):
    transport = Replies(absent(), absent(), reply(value=value))
    resolver = RosApiResolver(transport)
    profile = resolver.resolve()
    assert profile.version == RosVersion.ROS1 and profile.distro == "noetic"
    assert resolver.resolve() is profile and len(transport.calls) == 3


def test_explicit_ros_versions():
    for version, distro in [(1, "noetic"), (2, "humble")]:
        resolver = RosApiResolver(Replies(reply(version=version, distro=distro)))
        assert resolver.resolve().version.value == f"ros{version}"
        assert resolver.resolve().distro == distro


@pytest.mark.parametrize(
    "data", [None, [], {"op": "status"}, {"result": False, "values": {"version": 2}}]
)
def test_bad_envelope_does_not_prove_absence(data):
    fail = RosTransportResult(ok=True, data=data)
    transport = Replies(fail, fail)
    with pytest.raises(RosApiDetectionError):
        RosApiResolver(transport).resolve()
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({}, ""),
        ({"/arbitrary/_action/send_goal": "pkg/action/Compute_SendGoal"}, "pkg/action/Compute"),
        ({"/arbitrary/_action/get_result": "pkg/action/Compute_GetResult"}, "pkg/action/Compute"),
        ({"/arbitrary/_action/send_goal": "pkg/srv/Compute_SendGoal"}, ""),
        ({"/other/_action/send_goal": "pkg/action/Compute_SendGoal"}, ""),
        (
            {
                "/arbitrary/_action/send_goal": "pkg/action/A_SendGoal",
                "/arbitrary/_action/get_result": "pkg/action/B_GetResult",
            },
            "",
        ),
    ],
)
def test_action_fallback_only_uses_genuine_type_metadata(metadata, expected):
    transport = Replies(absent())
    graph = RosGraphDiscovery(transport, make_ros2_profile("/rosapi", "humble"))
    graph._service_types = metadata
    assert graph.get_action_type("/arbitrary") == expected
    assert transport.calls == [("/rosapi/action_type", {"action": "/arbitrary"})]


def test_action_direct_metadata_conflict_is_unresolved():
    graph = RosGraphDiscovery(
        Replies(reply(type="pkg/action/A")), make_ros2_profile("/rosapi", "humble")
    )
    graph._service_types = {"/x/_action/send_goal": "pkg/action/B_SendGoal"}
    assert graph.get_action_type("/x") == ""
