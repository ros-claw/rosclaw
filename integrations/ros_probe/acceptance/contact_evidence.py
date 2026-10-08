"""Independent contact evidence with explicit source mapping; no motion authority."""

import math
import re

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def contact_policy(policy):
    required = {
        "schema_version",
        "run_id",
        "body_snapshot_hash",
        "attachment_hash",
        "producer_id",
        "model_name",
        "pose_topic",
        "contacts",
        "support_topics",
        "ground_collisions",
        "source_sdf_sha256",
        "source_bridge_sha256",
    }
    if (
        type(policy) is not dict
        or set(policy) != required
        or policy["schema_version"] != "rosclaw.independent_contact_policy.v1"
    ):
        raise ValueError("closed explicit independent contact policy required")
    for key in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id", "model_name"):
        if type(policy[key]) is not str or not 0 < len(policy[key]) <= 256:
            raise ValueError("bounded frozen independent contact identity required")
    for key in ("source_sdf_sha256", "source_bridge_sha256"):
        if type(policy[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", policy[key]):
            raise ValueError("exact prepared source hashes required")
    contacts = policy["contacts"]
    if type(contacts) is not dict or not 1 <= len(contacts) <= 256:
        raise ValueError("bounded complete contact source mapping required")
    names = []
    for topic, collisions in contacts.items():
        if (
            not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z0-9_]*)+", topic)
            or type(collisions) is not list
            or not 1 <= len(collisions) <= 256
        ):
            raise ValueError("explicit ROS source and collision names required")
        if any(
            type(v) is not str
            or not v.startswith(policy["model_name"] + "::")
            or not 0 < len(v) <= 512
            for v in collisions
        ):
            raise ValueError("contact sources must belong to the frozen Body model")
        names.extend(collisions)
    if len(names) > 512 or len(names) != len(set(names)):
        raise ValueError("each Body collision requires one explicit contact source")
    if not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z0-9_]*)+", policy["pose_topic"]):
        raise ValueError("explicit independent pose source required")
    support = policy["support_topics"]
    grounds = policy["ground_collisions"]
    if (
        type(support) is not list
        or not 1 <= len(support) <= 32
        or len(set(support)) != len(support)
        or set(support) - set(contacts)
    ):
        raise ValueError("explicit continuous support streams required")
    if (
        type(grounds) is not list
        or not 1 <= len(grounds) <= 32
        or len(set(grounds)) != len(grounds)
        or any(type(v) is not str or not 0 < len(v) <= 512 or v in names for v in grounds)
    ):
        raise ValueError("explicit ground collision names required")
    return policy


class IndependentContacts:
    """Require fresh messages from every declared stream; missing is UNKNOWN."""

    def __init__(self, policy):
        self.policy = contact_policy(policy)
        self.policy_hash = digest(policy)
        self.sim_time = self.pose_wall = None
        self.seen = {}
        self.stamps = {}
        self.active = {}
        self.collision_count = 0
        self.fault = None
        self.admitted = False
        self.previous_tick_sim = None

    def pose(self, sim_time, wall_time):
        try:
            self._times(sim_time, wall_time)
            if self.sim_time is not None and (
                sim_time <= self.sim_time or wall_time <= self.pose_wall
            ):
                raise ValueError("independent pose source regressed or repeated")
            self.sim_time, self.pose_wall = sim_time, wall_time
        except ValueError as exc:
            self.fault = self.fault or str(exc)
            raise

    @staticmethod
    def _times(sim_time, wall_time):
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in (sim_time, wall_time)
        ):
            raise ValueError("finite nonnegative original source times required")

    def contacts(self, topic, sim_time, wall_time, collisions):
        try:
            self._times(sim_time, wall_time)
            if (
                topic not in self.policy["contacts"]
                or type(collisions) is not list
                or len(collisions) > 4096
            ):
                raise ValueError("bounded declared actual contact stream required")
            if self.sim_time is None or not -0.1 <= self.sim_time - sim_time < 0.3:
                raise ValueError("contact source lacks a fresh independent SIM clock")
            if sim_time < self.stamps.get(topic, 0) or wall_time < self.seen.get(topic, 0):
                raise ValueError("actual contact source timestamp regressed")
            touching = False
            for pair in collisions:
                if (
                    type(pair) is not list
                    or len(pair) != 2
                    or any(type(v) is not str or not 0 < len(v) <= 512 for v in pair)
                ):
                    raise ValueError("actual bounded collision pair required")
                if not any(v in self.policy["contacts"][topic] for v in pair):
                    raise ValueError("contact belongs to another Body collision/source")
                if not any(v in self.policy["ground_collisions"] for v in pair):
                    touching = True
            if topic in self.policy["support_topics"] and not any(
                any(v in self.policy["ground_collisions"] for v in pair) for pair in collisions
            ):
                raise ValueError("declared continuous support lacks actual ground contact")
            if touching and not any(self.active.values()):
                self.collision_count += 1
            self.active[topic] = touching
            self.seen[topic], self.stamps[topic] = wall_time, sim_time
        except ValueError as exc:
            self.fault = self.fault or str(exc)
            raise

    def snapshot(self, wall_time):
        self._times(0, wall_time)
        complete = (
            self.fault is None
            and self.sim_time is not None
            and 0 <= wall_time - self.pose_wall < 0.3
            and all(
                t in self.seen
                and 0 <= wall_time - self.seen[t] < 0.3
                and -0.1 <= self.sim_time - self.stamps[t] < 0.3
                for t in self.policy["contacts"]
            )
        )
        if self.admitted and not complete:
            self.fault = self.fault or "independent contact/pose source lost after admission"
        if complete:
            self.admitted = True
        result = {
            "source": "independent_gazebo_contact_subscription",
            "evidence_role": "independent_contact_observation_not_task_acceptance",
            "contact_policy_hash": self.policy_hash,
            "source_admitted": self.admitted,
            "observation_complete": bool(complete and self.fault is None),
            "sim_time_sec": self.sim_time,
            "collision_count": self.collision_count,
            "active_contact_topics": sorted(t for t, active in self.active.items() if active),
            "source_fault": self.fault,
            "contact_source_stamps": dict(self.stamps),
            "contact_wall_ages_sec": {t: wall_time - stamp for t, stamp in self.seen.items()},
        }
        return result


def prepare_contact_policy(
    directory, *, support_topics, ground_collisions, pose_topic="/rosclaw_sim/ground_truth"
):
    """Map explicit prepared SDF sensors and bridge; never guess support streams."""
    import hashlib
    import json
    from pathlib import Path
    from xml.etree import ElementTree as ET

    import yaml

    root = Path(directory)
    inputs = {}
    for name in ("robot.sdf", "bridge.yaml", "brush_binding.json", "physics_binding.json"):
        path = root / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 2_000_000:
            raise ValueError("bounded regular prepared contact source required")
        inputs[name] = path.read_bytes()
    xml = inputs["robot.sdf"]
    if b"\x00" in xml or any(v in xml.upper() for v in (b"<!DOCTYPE", b"<!ENTITY")):
        raise ValueError("materialized UTF-8 contact source required")
    tree = ET.fromstring(xml.decode("utf-8"))
    models = tree.findall("model") if tree.tag == "sdf" else []
    if (
        len(models) != 1
        or models[0].find("model") is not None
        or models[0].find("include") is not None
    ):
        raise ValueError("single materialized Body SDF model required")
    model = models[0]
    binding = json.loads(inputs["brush_binding.json"])
    physics = json.loads(inputs["physics_binding.json"])
    if any(
        binding.get(k) != physics.get(k)
        for k in ("run_id", "body_snapshot_hash", "attachment_hash")
    ) or physics.get("body_model_name") != model.get("name"):
        raise ValueError("prepared Body, brush and physics contact source binding differs")
    bridge = yaml.safe_load(inputs["bridge.yaml"])
    if type(bridge) is not list or len(bridge) > 1024:
        raise ValueError("bounded actual prepared bridge required")
    mapped = {}
    links = model.findall("link")
    link_names = [link.get("name") for link in links]
    if not 1 <= len(links) <= 512 or len(link_names) != len(set(link_names)):
        raise ValueError("bounded unique materialized Body links required")
    all_collisions = set()
    gz_topics = set()
    for link in model.findall("link"):
        name = link.get("name")
        collision_names = [c.get("name") for c in link.findall("collision")]
        if (
            type(name) is not str
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or any(
                type(v) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v)
                for v in collision_names
            )
            or len(collision_names) != len(set(collision_names))
        ):
            raise ValueError("unique explicit source link/collision names required")
        all_collisions.update(f"{model.get('name')}::{name}::{c}" for c in collision_names)
        for sensor in link.findall("sensor"):
            if sensor.get("type") != "contact":
                continue
            topic = sensor.findtext("topic")
            refs = [c.text for c in sensor.findall("contact/collision")]
            if (
                not refs
                or any(v not in collision_names for v in refs)
                or topic in mapped
                or len(refs) != len(set(refs))
            ):
                raise ValueError("complete unambiguous actual SDF contact references required")
            mappings = [v for v in bridge if type(v) is dict and v.get("ros_topic_name") == topic]
            if (
                len(mappings) != 1
                or mappings[0].get("ros_type_name") != "ros_gz_interfaces/msg/Contacts"
                or mappings[0].get("gz_type_name") != "gz.msgs.Contacts"
                or mappings[0].get("direction") != "GZ_TO_ROS"
                or type(mappings[0].get("gz_topic_name")) is not str
            ):
                raise ValueError("one exact prepared Gazebo-to-ROS contact bridge required")
            gz_topic = mappings[0]["gz_topic_name"]
            scoped = f"/world/{physics['world_name']}/model/{model.get('name')}/link/{name}/sensor/{sensor.get('name')}/contact"
            if gz_topic in gz_topics or gz_topic not in {topic, scoped}:
                raise ValueError(
                    "exact distinct SDF or fully scoped Gazebo contact source required"
                )
            gz_topics.add(gz_topic)
            mapped[topic] = [f"{model.get('name')}::{name}::{c}" for c in refs]
    if {v for row in mapped.values() for v in row} != all_collisions:
        raise ValueError("every prepared Body collision requires an explicit contact sensor")
    poses = [v for v in bridge if type(v) is dict and v.get("ros_topic_name") == pose_topic]
    if (
        len(poses) != 1
        or poses[0].get("ros_type_name") != "tf2_msgs/msg/TFMessage"
        or poses[0].get("gz_type_name") != "gz.msgs.Pose_V"
        or poses[0].get("direction") != "GZ_TO_ROS"
    ):
        raise ValueError("one explicit independent Gazebo pose bridge required")
    publishers = [
        v for v in model.findall("plugin") if v.get("name") == "gz::sim::systems::PosePublisher"
    ]
    if (
        len(publishers) != 1
        or publishers[0].findtext("topic") != poses[0].get("gz_topic_name")
        or publishers[0].findtext("publish_model_pose") != "true"
        or publishers[0].findtext("use_pose_vector_msg") != "true"
    ):
        raise ValueError("actual declared independent model pose publisher required")
    return contact_policy(
        {
            "schema_version": "rosclaw.independent_contact_policy.v1",
            **{
                k: binding[k]
                for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
            },
            "model_name": model.get("name"),
            "pose_topic": poses[0]["ros_topic_name"],
            "contacts": mapped,
            "support_topics": support_topics,
            "ground_collisions": ground_collisions,
            "source_sdf_sha256": hashlib.sha256(xml).hexdigest(),
            "source_bridge_sha256": hashlib.sha256(inputs["bridge.yaml"]).hexdigest(),
        }
    )


def reopen_contact_policy(directory, policy):
    if prepare_contact_policy(
        directory,
        support_topics=policy["support_topics"],
        ground_collisions=policy["ground_collisions"],
        pose_topic=policy["pose_topic"],
    ) != contact_policy(policy):
        raise ValueError("frozen independent contact source or binding changed")
    return policy
