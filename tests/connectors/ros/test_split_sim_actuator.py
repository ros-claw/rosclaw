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
    extra = {}

    def __init__(self, path):
        self.path = str(path)

    def __truediv__(self, name):
        return FakePath(self.path + "/" + name)

    def read_text(self):
        if self.path in self.extra:
            return self.extra[self.path]
        return {
            "/evidence/fixture_profile.json": json.dumps(PROFILE.to_dict()),
            "/evidence/contact_topics.json": json.dumps(["/wheel_left", "/wheel_right"]),
            "/evidence/run_id.txt": "run",
            "/evidence/brush_binding.json": json.dumps(BINDING),
        }[self.path]

    def open(self, *args, **kwargs):
        return io.StringIO()

    def write_text(self, text):
        return len(text)

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


def load_node(
    filename, classname, monkeypatch, *, dynamic=False, runtime_policy=None, brush_binding=None
):
    tree = ast.parse((ROOT / filename).read_text())
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and getattr(n, "name", None) != "main"
    ]

    class ConfiguredNode(FakeNode):
        def declare_parameter(self, name, default):
            if name == "dynamic_physics":
                return SimpleNamespace(value=dynamic)
            return super().declare_parameter(name, default)

    from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
    from rosclaw.connectors.ros.verification.occupancy_geometry import (
        OccupancyProjector,
        parse_physics_packet,
    )

    env = {
        "Node": ConfiguredNode,
        "SimpleNamespace": SimpleNamespace,
        "load_frozen_sim_runtime_policy": lambda _: runtime_policy,
        "CoverageVerifier": CoverageVerifier,
        "OccupancyProjector": OccupancyProjector,
        "parse_physics_packet": parse_physics_packet,
        "Path": FakePath,
        "json": json,
        "math": math,
        "time": time,
        "datetime": datetime,
        "UTC": UTC,
        "Lock": Lock,
        "deque": __import__("collections").deque,
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
    module.load_binding = lambda _: (brush_binding or BINDING).copy()
    monkeypatch.setitem(__import__("sys").modules, "sim_actuator", module)
    profiles = ModuleType("profiles")
    profiles.PROFILES = {"fixture": PROFILE}
    monkeypatch.setitem(__import__("sys").modules, "profiles", profiles)
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
    assert set(actor.services) == {
        "/rosclaw_sim/cleaning",
        "/rosclaw_sim/lease",
        "/rosclaw_sim/hold",
    }
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


def test_hold_zeros_drive_blocks_commands_and_cannot_be_reenabled_by_lease(monkeypatch):
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch)
    actor.tick()
    actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace())
    actor.sim_time = 1.1
    assert actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace()).success
    actor.sim_time = 1.2
    assert actor.set_hold(SimpleNamespace(data=True), SimpleNamespace()).success
    assert actor.holding and not actor.cleaning
    before = len(actor.publishers["/drive_controller/cmd_vel"])
    actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace())
    assert not actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace()).success
    actor.command(SimpleNamespace())
    assert len(actor.publishers["/drive_controller/cmd_vel"]) == before
    actor.lease = 0
    assert not actor.set_hold(SimpleNamespace(data=False), SimpleNamespace()).success
    assert actor.holding
    actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace())
    assert actor.set_hold(SimpleNamespace(data=False), SimpleNamespace()).success
    assert not actor.cleaning  # hold release alone does not enable the brush


def test_hold_transition_failure_stops_without_claiming_acknowledgment(monkeypatch):
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch)
    actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace())
    actor.cleaning = True
    actor.event = lambda _: (_ for _ in ()).throw(RuntimeError("source failed"))
    assert not actor.set_hold(SimpleNamespace(data=True), SimpleNamespace()).success
    assert actor.holding and actor.fault and not actor.cleaning and actor.lease == 0


def test_observer_waits_for_watermark_and_exact_disable_never_receives_on_credit(monkeypatch):
    observer = load_node("witness.py", "Witness", monkeypatch)
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch)
    observer.pose = {"x": 0, "y": 0, "yaw": 0, "time_sec": 1.15}
    observer.last_pose = time.monotonic()
    observer.contact_seen = {t: time.monotonic() for t in observer.support_topics}
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
        "pose_sim_time_sec": 1.15,
        "previous_pair_chain_hash": "GENESIS",
        "last_sequence": 2,
    }
    from rosclaw.connectors.ros.diagnosis.coverage_audit import digest

    pair["pair_chain_hash"] = digest(
        {
            "previous": "GENESIS",
            "pose_sim_time_sec": 1.15,
            "state_event_hash": "state",
            "watermark_event_hash": "watermark",
            "enabled": True,
        }
    )
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


@pytest.mark.parametrize("fault", [None, "sequence", "geometry", "stale"])
def test_actual_witness_callbacks_pair_component_packet_with_brush_and_apply_blocking(
    monkeypatch, fault
):
    import copy

    from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
    from rosclaw.connectors.ros.verification.occupancy import OccupancyAccounting

    grid = {
        "width": 4,
        "height": 4,
        "resolution": 0.1,
        "origin": [-0.2, -0.2],
        "accessible_cells": list(range(16)),
        "cleaning_polygon": [[-0.04, -0.04], [0.04, -0.04], [0.04, 0.04], [-0.04, 0.04]],
    }
    binding = {k: BINDING[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")}
    binding.update(
        world_name="fixture_world",
        body_model_name="anonymous_body",
        obstacle_names=["anonymous_blocker"],
        scene_model_names=["anonymous_body", "anonymous_blocker"],
        mission_id="mission",
        grid=grid,
        world_to_map_xyyaw=[0, 0, 0],
        frame_transform_source="simulator_operator_fixture_policy",
        map_world_identity_approved=True,
    )
    monkeypatch.setattr(FakePath, "extra", {"/evidence/physics_binding.json": json.dumps(binding)})
    observer = load_node("witness.py", "Witness", monkeypatch, dynamic=True)
    assert set(observer.publishers) == {"/rosclaw_sim/observation"} and not observer.services
    assert "/rosclaw_sim/physics_snapshot" in observer.subscriptions
    assert "/rosclaw_sim/ground_truth" not in observer.subscriptions
    observer.map_observation(
        SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            data=[0] * 16,
            info=SimpleNamespace(
                width=4,
                height=4,
                resolution=0.1,
                origin=SimpleNamespace(
                    position=SimpleNamespace(x=-0.2, y=-0.2),
                    orientation=SimpleNamespace(x=0, y=0, z=0, w=1),
                ),
            ),
        )
    )
    assert observer.physics_map_verified and observer.physics_fault is None
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch)
    actor.sim_time = 0
    actor.tick()
    actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace())
    actor.sim_time = 0.01
    actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace())
    actor.sim_time = 0.06
    actor.tick()
    for msg in actor.publishers["/rosclaw_sim/brush_events"]:
        observer.brush_event(msg)
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/passive-ecm-contract-packets.jsonl")
        .read_text()
        .splitlines()[0]
    )["packet"]
    packet = copy.deepcopy(fixture)
    packet.update({k: BINDING[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")})
    packet.update(sequence=0, sim_time_sec=0.05, paused=False, captured_at_unix_ns=time.time_ns())
    packet["body"]["world_pose"][:2] = [0.05, 0.05]
    observer.contact_seen = {t: time.monotonic() for t in observer.support_topics}
    retained = []
    observer.plan_audit.emit = lambda kind, payload, **kw: retained.append((kind, payload))
    wire = json.dumps(packet)
    observer.physics_event(SimpleNamespace(data=wire))
    import hashlib

    source = next(payload for kind, payload in retained if kind == "physics_snapshot_received")
    assert source["raw_packet_utf8"] == wire
    assert source["packet"] == packet
    assert source["packet_sha256"] == hashlib.sha256(wire.encode()).hexdigest()
    observer.tick()
    first = json.loads(observer.publishers["/rosclaw_sim/observation"][-1].data)
    assert first["observation_complete"] and first["cleaning_enabled"]
    assert first["time_sec"] == first["occupancy"]["sim_time_sec"] == 0.05
    assert first["occupancy"]["occupied_cells"] == list(range(16))
    accounting = OccupancyAccounting(
        CoverageVerifier(**grid),
        run_id="run",
        mission_id="mission",
        geometry_hash=observer.physics_projector.geometry_hash,
    )
    accounting.observe_sample(first)
    assert not accounting.verifier.visits
    actor.sim_time = 0.16
    actor.tick()
    observer.brush_event(actor.publishers["/rosclaw_sim/brush_events"][-1])
    packet.update(sequence=1, sim_time_sec=0.15, captured_at_unix_ns=time.time_ns())
    packet["obstacles"][0]["world_pose"][0] = 2
    if fault == "sequence":
        packet["sequence"] = 2
    elif fault == "geometry":
        packet["obstacles"][0]["collision_geometry"][0].update(radius=0.3, enclosing_radius_m=0.4)
    elif fault == "stale":
        packet["captured_at_unix_ns"] -= 300_000_000
    observer.physics_event(SimpleNamespace(data=json.dumps(packet)))
    observer.tick()
    second = json.loads(observer.publishers["/rosclaw_sim/observation"][-1].data)
    if fault is None:
        assert second["observation_complete"] and second["occupancy"]["occupied_cells"] == []
        accounting.observe_sample(second)
        assert set(accounting.verifier.visits) == {10}  # only actual new revisit after withdrawal
    else:
        assert not second["observation_complete"] and second["physics_source_fault"]
        with pytest.raises(ValueError):
            accounting.observe_sample(second)
        assert not accounting.verifier.visits
