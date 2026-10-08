"""Qualified read-only native contact-cache evidence, never mission acceptance.

An empty initialized ContactSensorData is distinct from a silent ROS topic.
This decoder requires the full actual inventory and continuous measured support;
cache snapshots alone do not prove backend health or a completed cleaning task.
"""

import hashlib
import json
import math
import re

from contact_evidence import IndependentContacts, contact_policy

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest

IDENTITIES = ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
HEADER = {
    "schema_version",
    "source",
    "evidence_domain",
    "sdk_version",
    "component_semantics",
    "body_pose_component",
    *IDENTITIES,
    "world_name",
    "body_model_name",
    "world_entity_id",
    "sequence",
    "iterations",
    "sim_time_sec",
    "physics_step_dt_sec",
    "captured_at_unix_ns",
    "paused",
    "complete",
}
SOURCE_KEYS = {
    "sensor_entity_id",
    "sensor_name",
    "link_entity_id",
    "link_name",
    "gz_topic",
    "collision_entity_ids",
}
PAIR_KEYS = {
    "collision1_entity_id",
    "collision2_entity_id",
    "collision1_name",
    "collision2_name",
}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate native contact source field")
        result[key] = value
    return result


def _integer(value, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value < 2**64:
        raise ValueError("bounded actual native source integer required")
    return value


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite actual native source number required")
    return value


def native_policy(policy):
    keys = {
        "schema_version",
        "contact_policy",
        "world_name",
        "native_sources",
        "component_topic",
        "plugin_sha256",
        "backend_health_evidence_required",
    }
    if type(policy) is not dict:
        raise ValueError("closed frozen native contact policy required")
    version = policy.get("schema_version")
    if version in {"rosclaw.native_contact_policy.v2", "rosclaw.native_contact_policy.v3"}:
        keys.add("component_gz_topic")
    if version == "rosclaw.native_contact_policy.v3":
        keys.add("sampling_semantics")
    if set(policy) != keys or version not in {
        "rosclaw.native_contact_policy.v1",
        "rosclaw.native_contact_policy.v2",
        "rosclaw.native_contact_policy.v3",
    }:
        raise ValueError("closed frozen native contact policy required")
    if version == "rosclaw.native_contact_policy.v3" and (
        policy["sampling_semantics"] != "ALL_POSTUPDATE_PHYSICS_STEPS"
        or policy["component_gz_topic"] != "/rosclaw_sim/contact_components"
    ):
        raise ValueError("v3 robot contact evidence requires all actual physics steps")
    if (
        version == "rosclaw.native_contact_policy.v1"
        and policy["component_topic"] != "/rosclaw_sim/contact_components"
    ):
        raise ValueError("v1 native component topic binding differs")
    if (
        type(policy["component_topic"]) is not str
        or len(policy["component_topic"]) > 1024
        or not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z0-9_]*)+", policy["component_topic"])
        or policy.get("component_gz_topic", "/rosclaw_sim/contact_components")
        not in {"/rosclaw_sim/contact_components", "/rosclaw_sim/backend_probe_components"}
    ):
        raise ValueError("explicit ROS observation bridge to simulator-owned producer required")
    base = contact_policy(policy["contact_policy"])
    if (
        type(policy["world_name"]) is not str
        or not 0 < len(policy["world_name"]) <= 256
        or type(policy["plugin_sha256"]) is not str
        or len(policy["plugin_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in policy["plugin_sha256"])
        or policy["backend_health_evidence_required"] is not True
    ):
        raise ValueError("explicit qualified native producer and health boundary required")
    if (
        policy["component_topic"] == base["pose_topic"]
        or policy["component_topic"] in base["contacts"]
    ):
        raise ValueError("distinct native component, pose and contact source roles required")
    sources = policy["native_sources"]
    if type(sources) is not dict or set(sources) != set(base["contacts"]):
        raise ValueError("every declared contact stream requires exact native source binding")
    used = set()
    for topic, source in sources.items():
        if type(source) is not dict or set(source) != {"gz_topic", "sensor_name", "link_name"}:
            raise ValueError("closed native source identity required")
        if any(type(v) is not str or not 0 < len(v) <= 1024 for v in source.values()):
            raise ValueError("bounded actual native source names required")
        if not source["gz_topic"].startswith("/") or source["gz_topic"] in used:
            raise ValueError("distinct qualified native contact topics required")
        used.add(source["gz_topic"])
        if any(
            not name.startswith(base["model_name"] + "::" + source["link_name"] + "::")
            for name in base["contacts"][topic]
        ):
            raise ValueError("declared native link differs from frozen collision scope")
    return policy


def decode_native_packet(raw, policy):
    """Validate original producer UTF-8 bytes without synthesizing sensor data."""
    policy = native_policy(policy)
    if type(raw) is not bytes or not 0 < len(raw) <= 262144:
        raise ValueError("bounded complete original native contact bytes required")
    try:
        packet = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("original native contact packet is not valid UTF-8 JSON") from exc
    if type(packet) is not dict or not set(packet) >= HEADER:
        raise ValueError("closed original native contact header required")
    base = policy["contact_policy"]
    expected = {
        "schema_version": "rosclaw.gazebo_postupdate_contacts.v2",
        "source": "gazebo_ecm_contact_sensor_data",
        "evidence_domain": "GAZEBO_PHYSICS",
        "sdk_version": "8.15.0",
        "component_semantics": "PHYSICS_UPDATE_CONTACT_CACHE",
        "body_pose_component": "PHYSICS_UPDATED_DIRECT_WORLD_MODEL_POSE",
        **{k: base[k] for k in IDENTITIES},
        "world_name": policy["world_name"],
        "body_model_name": base["model_name"],
    }
    if any(packet[k] != v for k, v in expected.items()):
        raise ValueError("native source version or frozen Body binding differs")
    if packet["complete"] is not True:
        raise ValueError(
            "native contact source incomplete: " + str(packet.get("fault", "UNKNOWN"))[:1024]
        )
    if set(packet) != HEADER | {
        "body_model_entity_id",
        "body_world_pose",
        "contact_sources",
        "collision_contacts",
    }:
        raise ValueError("closed complete native contact packet required")
    for key in ("world_entity_id", "body_model_entity_id", "iterations", "captured_at_unix_ns"):
        _integer(packet[key], positive=True)
    _integer(packet["sequence"])
    if packet["world_entity_id"] == packet["body_model_entity_id"]:
        raise ValueError("actual Body and World entities must differ")
    if (
        packet["paused"] is not False
        or _finite(packet["sim_time_sec"]) <= 0
        or not 0 < _finite(packet["physics_step_dt_sec"]) <= 1
    ):
        raise ValueError("unpaused advancing native physics source required")
    pose = packet["body_world_pose"]
    if type(pose) is not list or len(pose) != 7:
        raise ValueError("actual native Body world pose required")
    for value in pose:
        _finite(value)
    if abs(sum(v * v for v in pose[3:]) - 1) > 1e-6:
        raise ValueError("normalized actual native Body quaternion required")
    rows, inventory = packet["collision_contacts"], packet["contact_sources"]
    if (
        type(rows) is not list
        or not 1 <= len(rows) <= 256
        or type(inventory) is not list
        or not 1 <= len(inventory) <= 256
    ):
        raise ValueError("bounded full actual native contact inventory required")
    collision_names, pairs = {}, {}
    all_pairs = 0
    for row in rows:
        if type(row) is not dict or set(row) != {
            "collision_entity_id",
            "collision_name",
            "contacts",
        }:
            raise ValueError("closed actual collision contact row required")
        entity = _integer(row["collision_entity_id"], positive=True)
        name = row["collision_name"]
        if (
            type(name) is not str
            or not 0 < len(name) <= 512
            or entity in collision_names
            or name in collision_names.values()
        ):
            raise ValueError("unique actual collision identity required")
        collision_names[entity] = name
        contacts = row["contacts"]
        if type(contacts) is not list or len(contacts) > 4096:
            raise ValueError("bounded native contact pairs required")
        pairs[entity] = []
        for pair in contacts:
            all_pairs += 1
            if all_pairs > 4096 or type(pair) is not dict or set(pair) != PAIR_KEYS:
                raise ValueError("closed bounded actual native pair required")
            a, b = (
                _integer(pair[k], positive=True)
                for k in ("collision1_entity_id", "collision2_entity_id")
            )
            names = [pair["collision1_name"], pair["collision2_name"]]
            if (
                a == b
                or entity not in (a, b)
                or any(type(v) is not str or not 0 < len(v) <= 512 for v in names)
            ):
                raise ValueError("actual native pair belongs to another collision")
            if names[0 if a == entity else 1] != name:
                raise ValueError("actual native pair identity disagrees with collision row")
            pairs[entity].append(names)
    if set(collision_names.values()) != {v for values in base["contacts"].values() for v in values}:
        raise ValueError("native packet does not cover exactly every frozen Body collision")
    mapped, covered, sensors = {}, set(), set()
    topics = {v["gz_topic"]: k for k, v in policy["native_sources"].items()}
    for source in inventory:
        if type(source) is not dict or set(source) != SOURCE_KEYS:
            raise ValueError("closed actual native sensor row required")
        sensor = _integer(source["sensor_entity_id"], positive=True)
        _integer(source["link_entity_id"], positive=True)
        if type(source["gz_topic"]) is not str:
            raise ValueError("actual native topic must be bounded text")
        topic = topics.get(source["gz_topic"])
        if topic is None or topic in mapped or sensor in sensors:
            raise ValueError("native sensor topic is foreign or duplicated")
        sensors.add(sensor)
        frozen = policy["native_sources"][topic]
        if any(source[k] != frozen[k] for k in ("sensor_name", "link_name")):
            raise ValueError("actual native sensor/link differs from frozen source")
        ids = source["collision_entity_ids"]
        if type(ids) is not list or not 1 <= len(ids) <= 256:
            raise ValueError("bounded native collision references required")
        for entity in ids:
            _integer(entity, positive=True)
            if entity not in collision_names or entity in covered:
                raise ValueError("native collision reference missing or duplicated")
            covered.add(entity)
        if {collision_names[v] for v in ids} != set(base["contacts"][topic]):
            raise ValueError("actual sensor references differ from frozen Body collision mapping")
        mapped[topic] = [pair for entity in ids for pair in pairs[entity]]
    if set(mapped) != set(base["contacts"]) or covered != set(collision_names):
        raise ValueError("native contact inventory is incomplete")
    return packet, mapped


class NativeContactEvidence:
    """Fail closed on raw source faults, discontinuity and source loss.

    Source admission requires actual independent pose plus support contact. It
    does not grant backend health admission; a separate runtime gate must prove
    that the pinned physics backend is updating its cache.
    """

    def __init__(self, policy):
        self.policy = native_policy(policy)
        self.tracker = IndependentContacts(policy["contact_policy"])
        self.last = self.inventory_hash = self.world_pose = None
        self.actual_source_identity = None

    def pose(self, sim_time, wall_time, world_pose):
        try:
            if type(world_pose) is not list or len(world_pose) != 7:
                raise ValueError("complete independent world pose required")
            for value in world_pose:
                _finite(value)
            if abs(sum(v * v for v in world_pose[3:]) - 1) > 1e-6:
                raise ValueError("normalized independent world quaternion required")
            self.tracker.pose(sim_time, wall_time)
            self.world_pose = list(world_pose)
        except ValueError as exc:
            self.tracker.fault = self.tracker.fault or str(exc)
            raise

    def observe(self, raw, *, received_monotonic_sec, received_unix_ns):
        try:
            packet, mapped = decode_native_packet(raw, self.policy)
            if self.world_pose is None:
                raise ValueError("native contacts require original independent Body world pose")
            pose = packet["body_world_pose"]
            distance = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(pose[:3], self.world_pose[:3], strict=True))
            )
            dot = abs(sum(a * b for a, b in zip(pose[3:], self.world_pose[3:], strict=True)))
            if distance > 0.05 or dot < math.cos(0.1 / 2):
                raise ValueError(
                    "native contact pose differs from fresh independent Body world pose"
                )
            if (
                type(received_unix_ns) is not int
                or not 0 <= received_unix_ns - packet["captured_at_unix_ns"] < 300_000_000
            ):
                raise ValueError("native PostUpdate wall source stale or future")
            current = (
                packet["sequence"],
                packet["iterations"],
                packet["sim_time_sec"],
                packet["captured_at_unix_ns"],
            )
            if self.last is not None and any(
                a <= b for a, b in zip(current, self.last, strict=True)
            ):
                raise ValueError(
                    "native contact producer sequence, iteration or source time repeated/regressed"
                )
            if (
                self.last is not None
                and self.policy.get("sampling_semantics") == "ALL_POSTUPDATE_PHYSICS_STEPS"
                and (
                    current[0] != self.last[0] + 1
                    or current[1] != self.last[1] + 1
                    or abs(current[2] - self.last[2] - packet["physics_step_dt_sec"]) > 1e-9
                )
            ):
                raise ValueError(
                    "all-step robot contact source has missing physics steps or sequence"
                )
            identity = digest(
                {
                    k: packet[k]
                    for k in ("world_entity_id", "body_model_entity_id", "contact_sources")
                }
            )
            if self.inventory_hash is not None and identity != self.inventory_hash:
                raise ValueError("actual native contact inventory changed after source observation")
            for topic, pairs in mapped.items():
                self.tracker.contacts(topic, packet["sim_time_sec"], received_monotonic_sec, pairs)
            self.last, self.inventory_hash = current, identity
            self.actual_source_identity = {
                "world_entity_id": packet["world_entity_id"],
                "body_model_entity_id": packet["body_model_entity_id"],
                "entity_ids": {packet["body_model_entity_id"]}
                | {
                    entity
                    for row in packet["contact_sources"]
                    for entity in (
                        row["sensor_entity_id"],
                        row["link_entity_id"],
                        *row["collision_entity_ids"],
                    )
                },
            }
            return {
                "original_source_sha256": hashlib.sha256(raw).hexdigest(),
                "original_size_bytes": len(raw),
                "sequence": packet["sequence"],
                "iterations": packet["iterations"],
                "native_policy_hash": digest(self.policy),
            }
        except ValueError as exc:
            self.tracker.fault = self.tracker.fault or str(exc)
            raise

    def snapshot(self, wall_time):
        result = self.tracker.snapshot(wall_time)
        result.update(
            source="independent_gazebo_contact_component_subscription",
            component_semantics="PHYSICS_UPDATE_CONTACT_CACHE",
            native_policy_hash=digest(self.policy),
            backend_health_admitted=False,
            physical_acceptance="NOT_VERIFIED",
        )
        return result


def prepare_native_policy(
    directory,
    *,
    plugin_path,
    support_topics,
    ground_collisions,
    pose_topic,
    component_topic="/rosclaw_sim/contact_components",
    component_gz_topic="/rosclaw_sim/contact_components",
    all_physics_steps=False,
):
    """Freeze supplied SIM source bindings; this does not admit a running source."""
    from pathlib import Path
    from xml.etree import ElementTree as ET

    import yaml
    from contact_evidence import prepare_contact_policy

    directory = Path(directory)
    if type(all_physics_steps) is not bool or (
        all_physics_steps and component_gz_topic != "/rosclaw_sim/contact_components"
    ):
        raise ValueError("explicit all-step robot source declaration required")
    plugin_path = Path(plugin_path)
    if (
        plugin_path.is_symlink()
        or not plugin_path.is_file()
        or not plugin_path.resolve().is_relative_to(directory.resolve())
        or not 0 < plugin_path.stat().st_size <= 100_000_000
    ):
        raise ValueError("owned bounded regular native producer library required")
    base = prepare_contact_policy(
        directory,
        support_topics=support_topics,
        ground_collisions=ground_collisions,
        pose_topic=pose_topic,
    )
    physics = json.loads((directory / "physics_binding.json").read_bytes())
    model = ET.fromstring((directory / "robot.sdf").read_bytes()).find("model")
    bridge = yaml.safe_load((directory / "bridge.yaml").read_bytes())
    native_sources = {}
    for link in model.findall("link"):
        for sensor in link.findall("sensor"):
            if sensor.get("type") != "contact":
                continue
            topic = sensor.findtext("topic")
            contact_topic = sensor.findtext("contact/topic", "__default_topic__")
            resolved = (
                f"/world/{physics['world_name']}/model/{model.get('name')}/link/{link.get('name')}/sensor/{sensor.get('name')}/contact"
                if contact_topic == "__default_topic__"
                else contact_topic
            )
            mappings = [row for row in bridge if row.get("ros_topic_name") == topic]
            if len(mappings) != 1 or mappings[0].get("gz_topic_name") != resolved:
                raise ValueError(
                    "prepared bridge differs from qualified native contact/topic resolver"
                )
            native_sources[topic] = {
                "gz_topic": resolved,
                "sensor_name": sensor.get("name"),
                "link_name": link.get("name"),
            }
    components = [row for row in bridge if row.get("ros_topic_name") == component_topic]
    if len(components) != 1 or components[0] != {
        "ros_topic_name": component_topic,
        "gz_topic_name": component_gz_topic,
        "ros_type_name": "std_msgs/msg/String",
        "gz_type_name": "gz.msgs.StringMsg",
        "direction": "GZ_TO_ROS",
    }:
        raise ValueError("one exact native component producer bridge required")
    fixed = component_topic == component_gz_topic == "/rosclaw_sim/contact_components"
    version = (
        "rosclaw.native_contact_policy.v3"
        if all_physics_steps
        else "rosclaw.native_contact_policy.v1"
        if fixed
        else "rosclaw.native_contact_policy.v2"
    )
    return native_policy(
        {
            "schema_version": version,
            **(
                {}
                if fixed and not all_physics_steps
                else {"component_gz_topic": component_gz_topic}
            ),
            "contact_policy": base,
            "world_name": physics["world_name"],
            "native_sources": native_sources,
            "component_topic": component_topic,
            "plugin_sha256": hashlib.sha256(plugin_path.read_bytes()).hexdigest(),
            "backend_health_evidence_required": True,
            **({"sampling_semantics": "ALL_POSTUPDATE_PHYSICS_STEPS"} if all_physics_steps else {}),
        }
    )


def reopen_native_policy(directory, policy, *, plugin_path):
    base = native_policy(policy)["contact_policy"]
    current = prepare_native_policy(
        directory,
        plugin_path=plugin_path,
        support_topics=base["support_topics"],
        ground_collisions=base["ground_collisions"],
        pose_topic=base["pose_topic"],
        component_topic=policy["component_topic"],
        component_gz_topic=policy.get("component_gz_topic", "/rosclaw_sim/contact_components"),
        all_physics_steps=policy.get("sampling_semantics") == "ALL_POSTUPDATE_PHYSICS_STEPS",
    )
    if current != policy:
        raise ValueError("frozen native component source or producer library changed")
    return policy
