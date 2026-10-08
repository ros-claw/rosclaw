"""Execute fixture node code against endpoint recorders without importing ROS."""

import ast
import io
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from types import ModuleType, SimpleNamespace

import pytest

from rosclaw.connectors.ros.verification.brush_timeline import BrushStateEvent, BrushStateTimeline

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"
BINDING = {
    "run_id": "run",
    "body_snapshot_hash": "body",
    "attachment_hash": "brush",
    "producer_id": "actuator",
}
PROFILE = SimpleNamespace(physical_radius_m=0.25, to_dict=lambda: {"name": "fixture"})


class FakePath:
    def __init__(self, path):
        self.path = str(path)

    def __truediv__(self, name):
        return FakePath(self.path + "/" + name)

    def read_text(self):
        return {
            "/evidence/fixture_profile.json": json.dumps(PROFILE.to_dict()),
            "/evidence/contact_topics.json": json.dumps(["/wheel_left", "/wheel_right"]),
            "/evidence/run_id.txt": "run",
            "/evidence/brush_binding.json": json.dumps(BINDING),
        }[self.path]

    def open(self, *args, **kwargs):
        return io.StringIO()

    def exists(self):
        return True


class FakeNode:
    def __init__(self, name):
        self.publishers, self.subscriptions, self.services = {}, {}, {}
        self.sim_time = 1.0

    def declare_parameter(self, name, default):
        return SimpleNamespace(value=True if name == "split_actuator" else default)

    def create_publisher(self, kind, topic, qos):
        sent = []
        self.publishers[topic] = sent
        return SimpleNamespace(publish=sent.append)

    def create_subscription(self, kind, topic, callback, qos, **kwargs):
        self.subscriptions[topic] = callback

    def create_service(self, kind, name, callback, **kwargs):
        self.services[name] = callback

    def create_timer(self, *args, **kwargs):
        pass

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=int(self.sim_time * 1e9), to_msg=lambda: None)
        )


def load_node(filename, classname, monkeypatch):
    tree = ast.parse((ROOT / filename).read_text())
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and getattr(n, "name", None) != "main"
    ]
    env = {
        "Node": FakeNode,
        "Path": FakePath,
        "json": json,
        "math": math,
        "time": time,
        "datetime": datetime,
        "UTC": UTC,
        "Lock": Lock,
        "asdict": __import__("dataclasses").asdict,
        "BrushStateEvent": BrushStateEvent,
        "BrushStateTimeline": BrushStateTimeline,
        "PROFILES": {"fixture": PROFILE},
        "CoverageAuditLog": lambda *a, **kw: SimpleNamespace(emit=lambda *a, **kw: None),
        "QoSProfile": lambda **kw: None,
        "ReliabilityPolicy": SimpleNamespace(BEST_EFFORT=None, RELIABLE=None),
        "DurabilityPolicy": SimpleNamespace(TRANSIENT_LOCAL=None),
        "Clock": lambda **kw: None,
        "ClockType": SimpleNamespace(STEADY_TIME=None),
        "MutuallyExclusiveCallbackGroup": lambda: None,
    }
    for name in [
        "PointStamped",
        "PolygonStamped",
        "PoseWithCovarianceStamped",
        "Twist",
        "TwistStamped",
        "CollisionMonitorState",
        "OccupancyGrid",
        "NavPath",
        "TFMessage",
        "Marker",
        "Contacts",
        "SetBool",
    ]:
        env[name] = lambda **kw: SimpleNamespace(**kw)
    env["TwistStamped"] = lambda: SimpleNamespace(header=SimpleNamespace())
    env["Bool"] = env["String"] = lambda **kw: SimpleNamespace(**kw)
    module = ModuleType("sim_actuator")
    module.load_binding = lambda _: BINDING.copy()
    monkeypatch.setitem(__import__("sys").modules, "sim_actuator", module)
    exec(compile(tree, str(ROOT / filename), "exec"), env)
    return env[classname]()


def test_passive_node_creates_no_drive_cleaner_or_lease_authority(monkeypatch):
    observer = load_node("witness.py", "Witness", monkeypatch)
    assert set(observer.publishers) == {"/rosclaw_sim/observation"}
    assert not observer.services
    assert observer.velocity is None and observer.cleaning_state is None
    assert "/rosclaw_sim/brush_events" in observer.subscriptions
    for method in ("publish_velocity", "set_cleaning", "heartbeat"):
        args = (None,) if method == "publish_velocity" else (None, None)
        with pytest.raises(RuntimeError, match="passive observer"):
            getattr(observer, method)(*args)


def test_separate_actuator_requires_lease_and_stops_cleaner_on_expiry(monkeypatch):
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch)
    assert not any("ground_truth" in topic or "contact" in topic for topic in actor.subscriptions)
    assert set(actor.services) == {"/rosclaw_sim/cleaning", "/rosclaw_sim/lease"}
    response = actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace())
    assert response.success is False and not actor.cleaning
    actor.tick()  # initial OFF watermark
    actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace())
    actor.sim_time = 1.1
    assert actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace()).success
    actor.sim_time = 1.2
    actor.lease = time.monotonic() - 1
    actor.tick()
    assert not actor.cleaning and actor.publishers["/drive_controller/cmd_vel"]
    events = [json.loads(m.data) for m in actor.publishers["/rosclaw_sim/brush_events"]]
    assert [r["event"]["kind"] for r in events] == [
        "WATERMARK",
        "TRANSITION",
        "TRANSITION",
        "WATERMARK",
    ]
    assert [r["event"]["enabled"] for r in events] == [False, True, False, False]
    timeline = BrushStateTimeline(**BINDING)
    for row in events:
        timeline.append(BrushStateEvent(**row["event"]), artifact_hash=row["artifact_hash"])
    assert timeline.state_at(1.15)["enabled"] is True
    assert timeline.state_at(1.2)["status"] == "PENDING"  # exclusive closure


def test_failed_transition_publication_fails_closed(monkeypatch):
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch)
    actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace())
    actor.event = lambda _: (_ for _ in ()).throw(RuntimeError("publisher failed"))
    response = actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace())
    assert response.success is False
    assert actor.fault and not actor.cleaning and actor.lease == 0
    before = len(actor.publishers["/drive_controller/cmd_vel"])
    actor.command(SimpleNamespace())
    assert len(actor.publishers["/drive_controller/cmd_vel"]) == before


def test_observer_waits_for_watermark_and_exact_disable_never_receives_on_credit(monkeypatch):
    observer = load_node("witness.py", "Witness", monkeypatch)
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch)
    observer.pose = {"x": 0, "y": 0, "yaw": 0, "time_sec": 1.15}
    observer.last_pose = time.monotonic()
    observer.contact_seen = {t: time.monotonic() for t in observer.wheel_topics}
    observer.tick()
    assert not observer.publishers["/rosclaw_sim/observation"] and observer.brush_fault is None
    actor.tick()
    actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace())
    actor.sim_time = 1.1
    actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace())
    for msg in actor.publishers["/rosclaw_sim/brush_events"]:
        observer.brush_event(msg)
    observer.tick()
    assert not observer.publishers["/rosclaw_sim/observation"]
    actor.sim_time = 1.2
    actor.tick()
    observer.brush_event(actor.publishers["/rosclaw_sim/brush_events"][-1])
    observer.tick()
    first = json.loads(observer.publishers["/rosclaw_sim/observation"][-1].data)
    assert first["cleaning_enabled"] and first["observation_complete"]
    assert first["brush_source_binding"] == BINDING
    actor.set_cleaning(SimpleNamespace(data=False), SimpleNamespace())
    observer.brush_event(actor.publishers["/rosclaw_sim/brush_events"][-1])
    observer.pose = {**observer.pose, "time_sec": 1.2}
    observer.tick()
    assert len(observer.publishers["/rosclaw_sim/observation"]) == 1
    actor.sim_time = 1.3
    actor.tick()
    observer.brush_event(actor.publishers["/rosclaw_sim/brush_events"][-1])
    observer.tick()
    second = json.loads(observer.publishers["/rosclaw_sim/observation"][-1].data)
    assert not second["cleaning_enabled"] and second["observation_complete"]
    assert not observer.velocity and not observer.services


@pytest.mark.parametrize("fault", [None, "body", "attachment", "watermark", "enabled", "missing"])
def test_daemon_independent_brush_binding_and_pair_guard(fault):
    from rosclaw.connectors.ros.mission.executor import SimulationWitness

    witness = SimulationWitness.__new__(SimulationWitness)
    witness.brush_binding = BINDING.copy()
    witness.latest, witness.samples, witness.tracking, witness.action_fault = None, [], True, None
    witness.lock = Lock()
    pair = {
        "status": "PAIRED",
        "enabled": True,
        "watermark_sim_time_sec": 1.2,
        "state_event_hash": "state",
        "watermark_event_hash": "watermark",
        "pair_chain_hash": "chain",
    }
    record = {
        "time_sec": 1.15,
        "cleaning_enabled": True,
        "observation_complete": True,
        "collision_count": 0,
        "brush_source_binding": BINDING.copy(),
        "brush_state_pair": pair,
        "brush_source_fault": None,
    }
    if fault in ("body", "attachment"):
        record["brush_source_binding"][
            fault + "_snapshot_hash" if fault == "body" else "attachment_hash"
        ] = "other"
    elif fault == "watermark":
        pair["watermark_sim_time_sec"] = 1.15
    elif fault == "enabled":
        pair["enabled"] = False
    elif fault == "missing":
        record.pop("brush_state_pair")
    witness._record(record)
    if fault is None:
        assert witness.fresh()["observation_complete"]
    else:
        with pytest.raises(RuntimeError, match="incomplete"):
            witness.fresh()
        record["brush_state_pair"] = {**pair, "enabled": True, "watermark_sim_time_sec": 1.2}
        record["brush_source_binding"] = BINDING.copy()
        witness._record(record)
        with pytest.raises(RuntimeError, match="incomplete"):
            witness.fresh()  # one bad frame never vanishes under a later good frame
