"""Frozen generic SIM transport proposal, never live admission or authority.

No model template or namespace inference. Physical source and contact-sensor
placement still need independent checks against the loaded simulator.
"""

import json
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.context.sim_native_fixture import prepare_sim_native_fixture
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.intelligence.system_model import RosSystemModel
from rosclaw.connectors.ros.mission.sim_endpoints import absolute_endpoint

TOPIC_TYPES = {
    "observation": "std_msgs/msg/String",
    "physics": "std_msgs/msg/String",
    "brush_events": "std_msgs/msg/String",
    "cleaning_state": "std_msgs/msg/Bool",
    "localization": "geometry_msgs/msg/PoseWithCovarianceStamped",
    "map": "nav_msgs/msg/OccupancyGrid",
    "nav_velocity": "geometry_msgs/msg/Twist",
    "drive_velocity": "geometry_msgs/msg/TwistStamped",
}


def prepare_sim_runtime_policy(root, admission, model, declaration, *, now=None):
    """Reopen Body/URDF, match actual typed Graph, retain explicit scene policy."""
    now = now or datetime.now(UTC)
    root = Path(root)
    native = prepare_sim_native_fixture(root, admission)
    config, body = native["execution_config"], native["body"]
    if (
        model.snapshot_hash != model.compute_snapshot_hash()
        or model.snapshot_hash != admission["source_snapshot_hash"]
        or model.completeness.get("graph") is not True
        or not -100 <= model.age_ms(now) <= 5000
    ):
        raise ValueError("fresh intact matching actual Graph required")
    keys = {
        "source",
        "approved",
        "evidence_domain",
        "world_name",
        "body_model_name",
        "model_base_identity_approved",
        "model_base_identity_source",
        "map_world_identity_approved",
        "world_to_map_xyyaw",
        "topics",
        "collision_streams",
        "support_contact_topics",
        "ground_model_names",
        "controller_watchdog_approved",
    }
    if type(declaration) is not dict or set(declaration) != keys:
        raise ValueError("closed explicit generic SIM runtime policy required")
    policy = deepcopy(declaration)
    if (
        policy["source"] != "simulator_operator_fixture_policy"
        or policy["approved"] is not True
        or policy["evidence_domain"] != "SIMULATION"
        or policy["model_base_identity_approved"] is not True
        or policy["model_base_identity_source"] != "simulator_operator_fixture_policy"
        or policy["map_world_identity_approved"] is not True
        or type(policy["world_to_map_xyyaw"]) is not list
        or any(type(v) not in (int, float) for v in policy["world_to_map_xyyaw"])
        or policy["world_to_map_xyyaw"] != [0, 0, 0]
        or policy["controller_watchdog_approved"] is not True
    ):
        raise ValueError(
            "explicit SIM model/base and map/world identity and watchdog policy required"
        )
    for key in ("world_name", "body_model_name"):
        if type(policy[key]) is not str or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]{0,63}", policy[key]
        ):
            raise ValueError("bounded explicit simulator model/world name required")
    topics = policy["topics"]
    if type(topics) is not dict or set(topics) != set(TOPIC_TYPES):
        raise ValueError("complete explicit typed runtime topic roles required")
    topics = {k: absolute_endpoint(v) for k, v in topics.items()}
    if len(set(topics.values())) != len(topics):
        raise ValueError("runtime topic roles must be distinct")
    effective = BodyResolver(workspace=root / "home").get_effective_body(recompile_if_stale=False)
    bindings = effective.provider_interfaces["ros_capability_bindings"]
    if (
        topics["observation"] != config["observation_topic"]
        or topics["map"] != bindings["mapping.occupancy_map"]["name"]
        or topics["localization"] != bindings["localization.pose_estimate"]["name"]
        or topics["nav_velocity"] != bindings["safety.collision_monitor"]["output_topic"]
        or topics["cleaning_state"] != bindings["cleaning.enable"]["state_topic"]
    ):
        raise ValueError("runtime topic differs from actual compiled Body binding")
    map_candidates = [t for t in model.graph.get("topics", []) if t.get("name") == topics["map"]]
    frame = model.observations.get("message_frames", {}).get(topics["map"], {})
    if len(map_candidates) != 1 or frame.get("frame_id") != body["map_frame"]:
        raise ValueError("runtime map must match actual compiled map frame")
    for role, kind in TOPIC_TYPES.items():
        _observed_topic(model, topics[role], kind, now, fresh=True)
    raw = (root / "robot.urdf").read_bytes()
    robot = ET.fromstring(raw)
    collisions = {
        (link.get("name"), i)
        for link in robot.findall("link")
        for i, _ in enumerate(link.findall("collision"))
    }
    streams = policy["collision_streams"]
    if type(streams) is not list or not 1 <= len(streams) <= 512:
        raise ValueError("bounded complete body collision stream mapping required")
    seen, contact_topics = set(), []
    for item in streams:
        if (
            type(item) is not dict
            or set(item) != {"link", "collision_index", "topic"}
            or type(item["link"]) is not str
            or type(item["collision_index"]) is not int
        ):
            raise ValueError("typed source URDF collision stream mapping required")
        identity = (item["link"], item["collision_index"])
        topic = absolute_endpoint(item["topic"])
        if identity in seen or topic in contact_topics or topic in topics.values():
            raise ValueError("collision streams must be distinct and cover each URDF collision")
        seen.add(identity)
        contact_topics.append(topic)
        _observed_topic(model, topic, "ros_gz_interfaces/msg/Contacts", now, fresh=False)
    if seen != collisions:
        raise ValueError("every actual captured URDF collision requires an explicit stream")
    support = policy["support_contact_topics"]
    if (
        type(support) is not list
        or not support
        or any(type(t) is not str for t in support)
        or len(support) != len(set(support))
        or any(t not in contact_topics for t in support)
    ):
        raise ValueError("explicit continuously observed support contact streams required")
    for topic in support:
        _observed_topic(model, topic, "ros_gz_interfaces/msg/Contacts", now, fresh=True)
    grounds = policy["ground_model_names"]
    if (
        type(grounds) is not list
        or not 1 <= len(grounds) <= 32
        or any(
            type(n) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", n)
            for n in grounds
        )
        or len(grounds) != len(set(grounds))
        or policy["body_model_name"] in grounds
    ):
        raise ValueError("explicit distinct ground model identities required")
    result = {
        "schema_version": "rosclaw.sim_runtime_policy.v1",
        "status": "PREPARED_NOT_SOURCE_ADMITTED",
        "physical_acceptance": "NOT_RUN",
        "usable_for_real_execution": False,
        "actions_dispatched": False,
        "run_id": admission["run_id"],
        "mission_id": admission["mission_id"],
        "body_snapshot_hash": body["effective_body_hash"],
        "source_snapshot_hash": model.snapshot_hash,
        "execution_proposal_hash": admission["artifact_hash"],
        "source_urdf_sha256": effective.provider_interfaces["sim_fixture_evidence"][
            "source_urdf_sha256"
        ],
        "body": body,
        "endpoints": config["endpoints"],
        "grid": config["grid"],
        "policy": policy,
        "contact_topics": contact_topics,
        "requires_actual_component_geometry_and_contact_mapping_admission": True,
    }
    result["artifact_hash"] = digest(result)
    return result


def _observed_topic(model, name, kind, now, *, fresh):
    matches = [t for t in model.graph.get("topics", []) if t.get("name") == name]
    signals = [s for s in model.signals if s.topic == name]
    if (
        len(matches) != 1
        or matches[0].get("msg_type") != kind
        or len(signals) != 1
        or signals[0].publisher_count != 1
        or not -100 <= (now - signals[0].captured_at).total_seconds() * 1000 <= 5000
        or (
            fresh
            and (
                signals[0].last_message_age_ms is None
                or not 0 <= signals[0].last_message_age_ms < 300
            )
        )
    ):
        raise ValueError("unique typed actual runtime stream required: " + name)


def load_frozen_sim_runtime_policy(root):
    """Reopen archived source policy; live evidence must be checked separately.

    Absence is compatible only with the legacy known-fixture path. A generic
    execution marker refuses fallback even when its policy file is missing.
    """
    root = Path(root)
    path = root / "sim_runtime_policy.json"
    if not path.exists():
        config_path = root / "execution_config.json"
        if config_path.exists() and "generic_execution_proposal" in _read_json(config_path):
            raise ValueError("generic fixture requires its frozen runtime policy")
        return None
    saved = _read_json(path)
    admission = _read_json(root / "generic_execution_proposal.json")
    model = RosSystemModel.from_dict(_read_json(root / "snapshot.json"))
    # This checks the original source capture, not freshness of live ROS now.
    rebuilt = prepare_sim_runtime_policy(
        root, admission, model, saved.get("policy"), now=model.captured_at
    )
    if saved != rebuilt or (root / "run_id.txt").read_text().strip() != rebuilt["run_id"]:
        raise ValueError("frozen runtime source policy or run identity changed")
    brush = _read_json(root / "brush_binding.json")
    evidence = (
        BodyResolver(workspace=root / "home")
        .get_effective_body(recompile_if_stale=False)
        .provider_interfaces["sim_fixture_evidence"]
    )
    if (
        set(brush) != {"run_id", "body_snapshot_hash", "attachment_hash", "producer_id"}
        or brush["run_id"] != rebuilt["run_id"]
        or brush["body_snapshot_hash"] != rebuilt["body_snapshot_hash"]
        or brush["attachment_hash"] != evidence["attachment_hash"]
        or type(brush["producer_id"]) is not str
        or not 1 <= len(brush["producer_id"]) <= 256
    ):
        raise ValueError("generic runtime and actual compiled brush binding differ")
    binding = _read_json(root / "physics_binding.json")
    expected = {
        "run_id": rebuilt["run_id"],
        "mission_id": rebuilt["mission_id"],
        "body_snapshot_hash": rebuilt["body_snapshot_hash"],
        "attachment_hash": brush["attachment_hash"],
        "world_name": rebuilt["policy"]["world_name"],
        "body_model_name": rebuilt["policy"]["body_model_name"],
        "grid": rebuilt["grid"],
        "runtime_policy_hash": rebuilt["artifact_hash"],
        "source_urdf_sha256": rebuilt["source_urdf_sha256"],
        "maximum_body_planar_radius_m": rebuilt["body"]["physical_radius_m"],
        "body_reference_link": rebuilt["body"]["base_frame"],
        "model_base_identity_approved": True,
        "model_base_identity_source": "simulator_operator_fixture_policy",
        "world_to_map_xyyaw": [0, 0, 0],
        "map_world_identity_approved": True,
        "frame_transform_source": "simulator_operator_fixture_policy",
    }
    if any(type(binding.get(k)) is not type(v) or binding[k] != v for k, v in expected.items()):
        raise ValueError("generic physics policy differs from frozen runtime/Body sources")
    return rebuilt


def _read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("bounded frozen generic source file required")
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("typed frozen generic source file required")
    return value
