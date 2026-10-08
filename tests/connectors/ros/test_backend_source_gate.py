"""Joint-source logical contracts; pose decoder double, no actual runtime."""

import base64
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from tests.connectors.ros.test_backend_probe_evidence import specification
from tests.connectors.ros.test_native_contact_components import packet_named, policy


def sources():
    robot = packet_named("native_actual_contact_entity_names")
    robot_policy = policy(robot)
    robot_policy.update(
        schema_version="rosclaw.native_contact_policy.v3",
        component_gz_topic="/rosclaw_sim/contact_components",
        sampling_semantics="ALL_POSTUPDATE_PHYSICS_STEPS",
    )
    spec = specification()
    probe = copy.deepcopy(robot)
    probe.update(body_model_name="instrument_body", body_model_entity_id=40)
    row = probe["contact_sources"][0]
    row.update(
        sensor_entity_id=48,
        sensor_name="probe_touch",
        link_entity_id=44,
        link_name="probe_link",
        gz_topic="/qualified/probe_contact",
        collision_entity_ids=[46],
    )
    collision = "instrument_body::probe_link::probe_collision"
    probe["collision_contacts"][0].update(collision_entity_id=46, collision_name=collision)
    probe["collision_contacts"][0]["contacts"][0].update(
        collision1_entity_id=46, collision1_name=collision
    )
    native = spec["native_policy"]
    base = native["contact_policy"]
    base.update(
        model_name="instrument_body",
        pose_topic="/instrument/pose",
        contacts={"/instrument/contact": [collision]},
        support_topics=["/instrument/contact"],
    )
    native["native_sources"] = {
        "/instrument/contact": {
            "gz_topic": "/qualified/probe_contact",
            "sensor_name": "probe_touch",
            "link_name": "probe_link",
        }
    }
    return robot, robot_policy, probe, spec


def retained(raw, kind):
    return {
        "original_source_base64": base64.b64encode(raw).decode(),
        "original_source_sha256": hashlib.sha256(raw).hexdigest(),
        "original_size_bytes": len(raw),
        "source_bytes_complete": True,
        "source_type": kind,
    }


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("backend_source_gate")
    replay = importlib.import_module("closed_backend_probe")

    def synthetic_pose(raw, *args):
        packet = json.loads(raw)
        return packet["sim"], packet["pose"]

    monkeypatch.setattr(replay, "decode_ros_pose", synthetic_pose)
    return module


class Fixture:
    def __init__(self, module):
        self.robot, self.robot_policy, self.probe, self.spec = sources()
        self.gate = module.BackendObservationGate(self.robot_policy, self.spec, "synthetic_frame")
        self.i = 0
        self.origin = 1_791_504_000_000_000_000

    def frame(self, z=0.05, touching=True):
        self.i += 1
        sim, wall = self.i * 0.05, 100 + self.i * 0.05
        robot = copy.deepcopy(self.robot)
        robot.update(
            sequence=self.i,
            iterations=self.i,
            physics_step_dt_sec=0.05,
            sim_time_sec=sim,
            captured_at_unix_ns=self.origin + round(sim * 1e9),
        )
        self.gate.robot.pose(sim, wall, [0, 0, 0, 1, 0, 0, 0])
        self.gate.robot.observe(
            json.dumps(robot).encode(),
            received_monotonic_sec=wall + 0.001,
            received_unix_ns=robot["captured_at_unix_ns"] + 1_000_000,
        )
        probe = copy.deepcopy(self.probe)
        probe.update(
            sequence=self.i,
            iterations=self.i * 50,
            sim_time_sec=sim,
            captured_at_unix_ns=self.origin + round(sim * 1e9),
            body_world_pose=[6, 0, z, 1, 0, 0, 0],
        )
        if not touching:
            probe["collision_contacts"][0]["contacts"] = []
        pose = json.dumps({"sim": sim, "pose": probe["body_world_pose"]}).encode()
        self.gate.probe.apply(
            "backend_probe_pose",
            {**retained(pose, "tf2_msgs/msg/TFMessage"), "received_monotonic_sec": wall},
        )
        self.gate.probe.apply(
            "backend_probe_components",
            {
                **retained(json.dumps(probe).encode(), "gazebo_ecm_contact_sensor_data_json"),
                "received_monotonic_sec": wall + 0.001,
                "received_unix_ns": probe["captured_at_unix_ns"] + 1_000_000,
            },
        )
        return self.gate.snapshot(wall + 0.002)

    def cycle(self):
        for _ in range(12):
            self.frame()
        lift = importlib.import_module("probe_lift_evidence")
        wall = 100 + self.i * 0.05
        self.gate.probe.apply(
            "backend_probe_lift_ack",
            {
                "request": retained(lift.lift_request(self.spec), "gz.msgs.Pose_protobuf_text"),
                "response": retained(b"data: true", "gz.msgs.Boolean_protobuf_text"),
                "returncode": 0,
                "received_monotonic_sec": wall + 0.003,
                "received_unix_ns": self.origin + round(self.i * 0.05 * 1e9) + 3_000_000,
            },
        )
        for i in range(12):
            self.frame(z=10 - i * 0.05, touching=False)
        for _ in range(12):
            result = self.frame()
        return result


def test_disjoint_actual_sources_need_measured_cycle_and_do_not_grant_authority(module):
    f = Fixture(module)
    assert not f.gate.snapshot(100)["live_source_constraint_satisfied"]
    result = f.cycle()
    assert result["live_source_constraint_satisfied"]
    assert result["backend_health_admitted"] is False and result["authorization"] is False
    assert (
        result["physical_acceptance"] == "NOT_VERIFIED"
        and result["runtime_actor_integration"] == "NOT_IMPLEMENTED"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "different_world_id",
        "aliased_entity_id",
        "pose_loss",
        "regressed_gate_clock",
        "positive_robot_contact",
    ],
)
def test_lost_or_conflicting_actual_source_scope_latches_constraint_closed(module, fault):
    f = Fixture(module)
    assert f.cycle()["live_source_constraint_satisfied"]
    wall = 100 + f.i * 0.05 + 0.003
    if fault == "different_world_id":
        f.gate.probe.tracker.actual_source_identity["world_entity_id"] = 2
    elif fault == "aliased_entity_id":
        f.gate.probe.tracker.actual_source_identity["entity_ids"].add(
            f.gate.robot.actual_source_identity["body_model_entity_id"]
        )
    elif fault == "pose_loss":
        wall += 0.31
    elif fault == "regressed_gate_clock":
        wall -= 0.01
    else:
        f.gate.robot.tracker.collision_count = 1
    result = f.gate.snapshot(wall)
    assert not result["live_source_constraint_satisfied"] and result["source_fault"]
    assert not f.gate.snapshot(wall + 0.01)["live_source_constraint_satisfied"]


@pytest.mark.parametrize(
    "fault",
    [
        "different_run",
        "different_world",
        "aliased_model",
        "aliased_component_topic",
        "aliased_pose_topic",
        "aliased_contact_topic",
    ],
)
def test_cross_run_world_or_endpoint_policy_aliases_refused(module, fault):
    robot, robot_policy, probe, spec = sources()
    base = spec["native_policy"]["contact_policy"]
    if fault == "different_run":
        base["run_id"] = "different"
    elif fault == "different_world":
        spec["native_policy"]["world_name"] = "different"
    elif fault == "aliased_model":
        base["model_name"] = robot_policy["contact_policy"]["model_name"]
    elif fault == "aliased_component_topic":
        spec["native_policy"]["component_topic"] = robot_policy["component_topic"]
    elif fault == "aliased_pose_topic":
        base["pose_topic"] = robot_policy["contact_policy"]["pose_topic"]
    else:
        base["contacts"] = robot_policy["contact_policy"]["contacts"]
        base["support_topics"] = robot_policy["contact_policy"]["support_topics"]
    with pytest.raises(ValueError):
        module.BackendObservationGate(robot_policy, spec, "synthetic_frame")
