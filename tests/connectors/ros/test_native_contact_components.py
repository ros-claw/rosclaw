"""Real SDK-produced ECS bytes and fail-closed native contact decoding contracts."""

import copy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CASES = [
    json.loads(line)
    for line in (Path(__file__).parent / "fixtures/passive-native-contact-contract-packets.jsonl")
    .read_text()
    .splitlines()
]


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "integrations/ros_probe/acceptance"))
    return importlib.import_module("native_contact_evidence")


def policy(packet):
    return {
        "schema_version": "rosclaw.native_contact_policy.v1",
        "world_name": "fixture_world",
        "component_topic": "/rosclaw_sim/contact_components",
        "plugin_sha256": "a" * 64,
        "backend_health_evidence_required": True,
        "native_sources": {
            "/actual/contact": {
                "sensor_name": "native_touch",
                "link_name": "actual_base",
                "gz_topic": packet["contact_sources"][0]["gz_topic"],
            }
        },
        "contact_policy": {
            "schema_version": "rosclaw.independent_contact_policy.v1",
            **{
                k: packet[k]
                for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
            },
            "model_name": "anonymous_body",
            "pose_topic": "/actual/pose",
            "contacts": {"/actual/contact": ["anonymous_body::actual_base::actual_collision"]},
            "support_topics": ["/actual/contact"],
            "ground_collisions": ["support_plane::floor_link::floor_collision"],
            "source_sdf_sha256": "b" * 64,
            "source_bridge_sha256": "c" * 64,
        },
    }


def packet_named(name):
    return copy.deepcopy(next(row["packet"] for row in CASES if row["case"] == name))


@pytest.mark.parametrize("row", CASES, ids=lambda row: row["case"])
def test_original_official_sdk_packets_decode_or_refuse(evidence, row):
    packet = row["packet"]
    frozen = policy(
        packet if row["expected_complete"] else packet_named("native_actual_contact_entity_names")
    )
    line = next(
        line
        for line in (
            Path(__file__).parent / "fixtures/passive-native-contact-contract-packets.jsonl"
        )
        .read_bytes()
        .splitlines()
        if json.loads(line)["case"] == row["case"]
    )
    raw = line.split(b'"packet":', 1)[1][:-1]
    if row["expected_complete"]:
        decoded, pairs = evidence.decode_native_packet(raw, frozen)
        assert decoded == packet
        assert set(pairs) == {"/actual/contact"}
    else:
        with pytest.raises(ValueError):
            evidence.decode_native_packet(raw, frozen)


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "sdk",
        "old_schema",
        "pose_component",
        "body",
        "missing_row",
        "duplicate_row",
        "wrong_pair",
        "topic",
        "sensor",
        "quaternion",
        "bool_sequence",
        "duplicate_json",
        "nan",
        "utf8",
        "oversized",
    ],
)
def test_corrupted_raw_bytes_are_not_admitted(evidence, fault):
    packet = packet_named("native_actual_contact_entity_names")
    frozen = policy(packet)
    if fault == "extra":
        packet["untrusted"] = True
    elif fault == "sdk":
        packet["sdk_version"] = "8.16.0"
    elif fault == "old_schema":
        packet["schema_version"] = "rosclaw.gazebo_postupdate_contacts.v1"
    elif fault == "pose_component":
        packet["body_pose_component"] = "OPTIONAL_WORLD_POSE"
    elif fault == "body":
        packet["body_snapshot_hash"] = "wrong"
    elif fault == "missing_row":
        packet["collision_contacts"] = []
    elif fault == "duplicate_row":
        packet["collision_contacts"] *= 2
    elif fault == "wrong_pair":
        packet["collision_contacts"][0]["contacts"][0]["collision1_entity_id"] = 999
    elif fault == "topic":
        packet["contact_sources"][0]["gz_topic"] = "/foreign"
    elif fault == "sensor":
        packet["contact_sources"][0]["sensor_name"] = "foreign"
    elif fault == "quaternion":
        packet["body_world_pose"][3] = 0
    elif fault == "bool_sequence":
        packet["sequence"] = True
    raw = json.dumps(packet).encode()
    if fault == "duplicate_json":
        raw = raw[:-1] + b',"sequence":0}'
    elif fault == "nan":
        raw = raw.replace(b'"sim_time_sec": 0.1', b'"sim_time_sec": NaN')
    elif fault == "utf8":
        raw = b"\xff"
    elif fault == "oversized":
        raw = b" " * 262145
    with pytest.raises(ValueError):
        evidence.decode_native_packet(raw, frozen)


def observe(evidence, packet, frozen=None):
    tracker = evidence.NativeContactEvidence(frozen or policy(packet))
    tracker.pose(packet["sim_time_sec"], 10.0, packet["body_world_pose"])
    tracker.observe(
        json.dumps(packet).encode(),
        received_monotonic_sec=10.01,
        received_unix_ns=packet["captured_at_unix_ns"] + 10_000_000,
    )
    return tracker


def test_actual_ground_contact_admits_observation_but_not_backend_or_task(evidence):
    packet = packet_named("native_actual_contact_entity_names")
    tracker = observe(evidence, packet)
    result = tracker.snapshot(10.02)
    assert result["observation_complete"] is True
    assert result["collision_count"] == 0
    assert result["backend_health_admitted"] is False
    assert result["physical_acceptance"] == "NOT_VERIFIED"
    assert tracker.snapshot(10.31)["observation_complete"] is False
    assert tracker.tracker.fault


def test_initialized_empty_component_is_not_continuous_support_proof(evidence):
    packet = packet_named("native_empty_component_not_ros_silence")
    with pytest.raises(ValueError, match="support"):
        observe(evidence, packet)


@pytest.mark.parametrize(
    "fault",
    [
        "repeat",
        "sequence",
        "iteration",
        "sim",
        "unix",
        "inventory",
        "stale",
        "future",
        "lost_support",
    ],
)
def test_source_continuity_fault_latches_after_admission(evidence, fault):
    first = packet_named("native_actual_contact_entity_names")
    tracker = observe(evidence, first)
    assert tracker.snapshot(10.02)["observation_complete"]
    second = copy.deepcopy(first)
    second.update(
        sequence=1,
        iterations=150,
        sim_time_sec=0.15,
        captured_at_unix_ns=first["captured_at_unix_ns"] + 50_000_000,
    )
    tracker.pose(0.15, 10.05, first["body_world_pose"])
    if fault == "repeat":
        second = first
    elif fault == "sequence":
        second["sequence"] = 0
    elif fault == "iteration":
        second["iterations"] = first["iterations"]
    elif fault == "sim":
        second["sim_time_sec"] = 0.1
    elif fault == "unix":
        second["captured_at_unix_ns"] = first["captured_at_unix_ns"]
    elif fault == "inventory":
        second["contact_sources"][0]["sensor_entity_id"] = 999
    elif fault == "lost_support":
        second["collision_contacts"][0]["contacts"] = []
    wall = second["captured_at_unix_ns"] + (
        300_000_000 if fault == "stale" else -1 if fault == "future" else 10_000_000
    )
    with pytest.raises(ValueError):
        tracker.observe(
            json.dumps(second).encode(), received_monotonic_sec=10.06, received_unix_ns=wall
        )
    assert tracker.snapshot(10.07)["observation_complete"] is False
    assert tracker.tracker.fault


@pytest.mark.parametrize("pose", [[0.06, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]])
def test_original_independent_world_pose_mismatch_is_rejected(evidence, pose):
    packet = packet_named("native_actual_contact_entity_names")
    tracker = evidence.NativeContactEvidence(policy(packet))
    tracker.pose(0.1, 10.0, pose)
    with pytest.raises(ValueError, match="pose differs"):
        tracker.observe(
            json.dumps(packet).encode(),
            received_monotonic_sec=10.01,
            received_unix_ns=packet["captured_at_unix_ns"] + 1,
        )
    assert tracker.tracker.fault


@pytest.fixture
def prepared_native(tmp_path):
    import xml.etree.ElementTree as ET

    import yaml

    model = ET.Element("model", name="anonymous_body")
    for name in ("support", "body"):
        link = ET.SubElement(model, "link", name=name)
        ET.SubElement(link, "collision", name="solid")
        sensor = ET.SubElement(link, "sensor", name="touch", type="contact")
        ET.SubElement(sensor, "topic").text = "/contacts/" + name
        contact = ET.SubElement(sensor, "contact")
        ET.SubElement(contact, "collision").text = "solid"
        ET.SubElement(contact, "topic").text = "/contacts/" + name
    plugin = ET.SubElement(model, "plugin", name="gz::sim::systems::PosePublisher")
    for key, text in {
        "topic": "/actual_pose",
        "publish_model_pose": "true",
        "use_pose_vector_msg": "true",
    }.items():
        ET.SubElement(plugin, key).text = text
    sdf = ET.Element("sdf", version="1.9")
    sdf.append(model)
    (tmp_path / "robot.sdf").write_bytes(ET.tostring(sdf))
    bridge = [
        {
            "ros_topic_name": "/contacts/" + name,
            "gz_topic_name": "/contacts/" + name,
            "ros_type_name": "ros_gz_interfaces/msg/Contacts",
            "gz_type_name": "gz.msgs.Contacts",
            "direction": "GZ_TO_ROS",
        }
        for name in ("support", "body")
    ]
    bridge.extend(
        [
            {
                "ros_topic_name": "/actual_pose",
                "gz_topic_name": "/actual_pose",
                "ros_type_name": "tf2_msgs/msg/TFMessage",
                "gz_type_name": "gz.msgs.Pose_V",
                "direction": "GZ_TO_ROS",
            },
            {
                "ros_topic_name": "/rosclaw_sim/contact_components",
                "gz_topic_name": "/rosclaw_sim/contact_components",
                "ros_type_name": "std_msgs/msg/String",
                "gz_type_name": "gz.msgs.StringMsg",
                "direction": "GZ_TO_ROS",
            },
        ]
    )
    (tmp_path / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    source = packet_named("native_actual_contact_entity_names")
    binding = {
        k: source[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
    }
    (tmp_path / "brush_binding.json").write_text(json.dumps(binding))
    (tmp_path / "physics_binding.json").write_text(
        json.dumps({**binding, "body_model_name": "anonymous_body", "world_name": "fixture_world"})
    )
    # This is a synthetic source-file integrity fixture, not a loadable plugin.
    (tmp_path / "source-fixture.so").write_bytes(b"synthetic producer-file integrity bytes")
    return tmp_path


def prepare(evidence, path):
    return evidence.prepare_native_policy(
        path,
        plugin_path=path / "source-fixture.so",
        support_topics=["/contacts/support"],
        ground_collisions=["floor::link::collision"],
        pose_topic="/actual_pose",
    )


def test_prepared_source_freeze_reopens_without_physical_admission(evidence, prepared_native):
    frozen = prepare(evidence, prepared_native)
    assert frozen["backend_health_evidence_required"] is True
    assert (
        evidence.reopen_native_policy(
            prepared_native, frozen, plugin_path=prepared_native / "source-fixture.so"
        )
        == frozen
    )
    (prepared_native / "source-fixture.so").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        evidence.reopen_native_policy(
            prepared_native, frozen, plugin_path=prepared_native / "source-fixture.so"
        )


@pytest.mark.parametrize(
    "fault",
    [
        "nested_missing",
        "nested_wrong",
        "component_missing",
        "component_type",
        "component_duplicate",
        "plugin_symlink",
        "bridge_modified",
        "binding_modified",
    ],
)
def test_prepared_native_topic_or_binding_conflicts_refuse(evidence, prepared_native, fault):
    import xml.etree.ElementTree as ET

    import yaml

    frozen = prepare(evidence, prepared_native)
    if fault in {"nested_missing", "nested_wrong"}:
        tree = ET.parse(prepared_native / "robot.sdf")
        contact = tree.find("model/link/sensor/contact")
        if fault == "nested_missing":
            contact.remove(contact.find("topic"))
        else:
            contact.find("topic").text = "/wrong"
        tree.write(prepared_native / "robot.sdf")
    elif fault in {"component_missing", "component_type", "component_duplicate"}:
        bridge = yaml.safe_load((prepared_native / "bridge.yaml").read_bytes())
        if fault == "component_missing":
            bridge.pop()
        elif fault == "component_type":
            bridge[-1]["gz_type_name"] = "gz.msgs.Contacts"
        else:
            bridge.append(bridge[-1])
        (prepared_native / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    elif fault == "plugin_symlink":
        plugin = prepared_native / "source-fixture.so"
        plugin.rename(prepared_native / "original")
        plugin.symlink_to(prepared_native / "original")
    elif fault == "bridge_modified":
        bridge = prepared_native / "bridge.yaml"
        bridge.write_bytes(bridge.read_bytes() + b"\n")
    else:
        binding = prepared_native / "physics_binding.json"
        item = json.loads(binding.read_bytes())
        item["run_id"] = "foreign"
        binding.write_text(json.dumps(item))
    with pytest.raises(ValueError):
        evidence.reopen_native_policy(
            prepared_native, frozen, plugin_path=prepared_native / "source-fixture.so"
        )


def test_distinct_ros_and_explicit_native_gazebo_topics_are_preserved(evidence, prepared_native):
    import xml.etree.ElementTree as ET

    import yaml

    tree = ET.parse(prepared_native / "robot.sdf")
    tree.find("model/link/sensor/contact/topic").text = "/qualified/native_support"
    tree.write(prepared_native / "robot.sdf")
    bridge = yaml.safe_load((prepared_native / "bridge.yaml").read_bytes())
    bridge[0]["gz_topic_name"] = "/qualified/native_support"
    (prepared_native / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    frozen = prepare(evidence, prepared_native)
    assert frozen["native_sources"]["/contacts/support"]["gz_topic"] == "/qualified/native_support"
    assert (
        evidence.reopen_native_policy(
            prepared_native, frozen, plugin_path=prepared_native / "source-fixture.so"
        )
        == frozen
    )


@pytest.mark.parametrize(
    "ros_topic,gz_topic",
    [
        ("/unseen/observations/components", "/rosclaw_sim/contact_components"),
        ("/instrument/observations/components", "/rosclaw_sim/backend_probe_components"),
    ],
)
def test_explicit_namespace_and_separate_probe_producer_reopen(
    evidence, prepared_native, ros_topic, gz_topic
):
    import yaml

    bridge = yaml.safe_load((prepared_native / "bridge.yaml").read_bytes())
    bridge[-1].update(ros_topic_name=ros_topic, gz_topic_name=gz_topic)
    (prepared_native / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    native = evidence.prepare_native_policy(
        prepared_native,
        plugin_path=prepared_native / "source-fixture.so",
        support_topics=["/contacts/support"],
        ground_collisions=["floor::link::collision"],
        pose_topic="/actual_pose",
        component_topic=ros_topic,
        component_gz_topic=gz_topic,
    )
    assert native["schema_version"] == "rosclaw.native_contact_policy.v2"
    assert native["component_topic"] == ros_topic and native["component_gz_topic"] == gz_topic
    assert (
        evidence.reopen_native_policy(
            prepared_native, native, plugin_path=prepared_native / "source-fixture.so"
        )
        == native
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("component_gz_topic", "/cmd_vel"),
        ("component_topic", "relative"),
        ("component_topic", "/actual/pose"),
        ("component_topic", "/actual/contact"),
        ("schema_version", "unknown"),
    ],
)
def test_probe_or_namespace_policy_cannot_alias_other_source_roles(evidence, field, value):
    native = policy(packet_named("native_actual_contact_entity_names"))
    native.update(
        schema_version="rosclaw.native_contact_policy.v2",
        component_gz_topic="/rosclaw_sim/backend_probe_components",
    )
    native[field] = value
    with pytest.raises(ValueError):
        evidence.native_policy(native)
