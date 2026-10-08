"""Independent source contracts; synthetic packets are not physical acceptance."""

import importlib.util
import json
from pathlib import Path

import pytest

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"
spec = importlib.util.spec_from_file_location("independent_contacts", ROOT / "contact_evidence.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def policy():
    return {
        "schema_version": "rosclaw.independent_contact_policy.v1",
        "run_id": "run",
        "body_snapshot_hash": "body",
        "attachment_hash": "brush",
        "producer_id": "actor",
        "model_name": "anonymous_body",
        "pose_topic": "/independent_pose",
        "contacts": {
            "/contacts/support": ["anonymous_body::support::solid"],
            "/contacts/body": ["anonymous_body::base::solid"],
        },
        "support_topics": ["/contacts/support"],
        "ground_collisions": ["floor::ground::solid"],
        "source_sdf_sha256": "a" * 64,
        "source_bridge_sha256": "b" * 64,
    }


def feed(tracker, sim, wall, touching=False):
    tracker.pose(sim, wall)
    tracker.contacts(
        "/contacts/support", sim, wall, [["anonymous_body::support::solid", "floor::ground::solid"]]
    )
    tracker.contacts(
        "/contacts/body",
        sim,
        wall,
        [["anonymous_body::base::solid", "obstacle::base::solid"]] if touching else [],
    )
    return tracker.snapshot(wall)


def test_independent_streams_require_actual_startup_and_count_observed_contacts():
    tracker = m.IndependentContacts(policy())
    initial = tracker.snapshot(0)
    assert not initial["source_admitted"] and not initial["observation_complete"]
    tracker.pose(1, 1)
    tracker.contacts(
        "/contacts/support", 1, 1, [["anonymous_body::support::solid", "floor::ground::solid"]]
    )
    assert not tracker.snapshot(1)["observation_complete"]  # No missing-body-topic zero inference.
    tracker.contacts("/contacts/body", 1, 1, [])
    assert tracker.snapshot(1)["observation_complete"]
    for sim, touch, count in [(1.1, True, 1), (1.2, True, 1), (1.3, False, 1), (1.4, True, 2)]:
        result = feed(tracker, sim, sim, touching=touch)
        assert result["observation_complete"] and result["collision_count"] == count
    assert result["contact_policy_hash"] == digest(policy())


@pytest.mark.parametrize(
    "fault",
    [
        "topic",
        "foreign",
        "support_absent",
        "fake_ground",
        "stamp",
        "future",
        "nan",
        "pair",
        "oversize",
        "pose_repeat",
        "pose_reverse",
        "source_loss",
    ],
)
def test_independent_source_faults_latch_without_recovery_credit(fault):
    tracker = m.IndependentContacts(policy())
    feed(tracker, 1, 1)
    args = ["/contacts/body", 1, 1.01, []]
    if fault == "topic":
        args[0] = "/other"
    elif fault == "foreign":
        args[3] = [["other::base::solid", "floor::ground::solid"]]
    elif fault == "support_absent":
        args[0] = "/contacts/support"
    elif fault == "fake_ground":
        args[:1] = ["/contacts/support"]
        args[3] = [["anonymous_body::support::solid", "floor::evil::solid"]]
    elif fault == "stamp":
        args[1] = 0.9
    elif fault == "future":
        args[1] = 1.2
    elif fault == "nan":
        args[1] = float("nan")
    elif fault == "pair":
        args[3] = [["anonymous_body::base::solid"]]
    elif fault == "oversize":
        args[3] = [[]] * 4097
    if fault == "source_loss":
        assert not tracker.snapshot(1.31)["observation_complete"]
    elif fault.startswith("pose_"):
        with pytest.raises(ValueError):
            tracker.pose(1 if fault == "pose_repeat" else 0.9, 1.1)
    else:
        with pytest.raises(ValueError):
            tracker.contacts(*args)
    assert tracker.fault
    assert not feed(tracker, 2, 2)["observation_complete"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("support_topics", []),
        ("support_topics", ["/unknown"]),
        ("ground_collisions", []),
        ("ground_collisions", ["anonymous_body::base::solid"]),
        ("source_sdf_sha256", "old"),
        ("pose_topic", "/0bad"),
        ("contacts", {}),
        ("producer_id", ""),
    ],
)
def test_policy_refuses_missing_or_ambiguous_source_declarations(field, value):
    declaration = policy()
    declaration[field] = value
    with pytest.raises(ValueError):
        m.contact_policy(declaration)


@pytest.fixture
def prepared(tmp_path):
    sdf = '<sdf version="1.9"><model name="anonymous_body"><link name="support"><collision name="solid"/><sensor name="touch" type="contact"><topic>/contacts/support</topic><contact><collision>solid</collision></contact></sensor></link><link name="base"><collision name="solid"/><sensor name="touch" type="contact"><topic>/contacts/body</topic><contact><collision>solid</collision></contact></sensor></link></model></sdf>'
    sdf = sdf.replace(
        "</model>",
        '<plugin name="gz::sim::systems::PosePublisher"><topic>/actual_pose</topic><publish_model_pose>true</publish_model_pose><use_pose_vector_msg>true</use_pose_vector_msg></plugin></model>',
    )
    (tmp_path / "robot.sdf").write_text(sdf)
    import yaml

    bridge = [
        {
            "ros_topic_name": topic,
            "gz_topic_name": topic,
            "ros_type_name": "ros_gz_interfaces/msg/Contacts",
            "gz_type_name": "gz.msgs.Contacts",
            "direction": "GZ_TO_ROS",
        }
        for topic in ("/contacts/support", "/contacts/body")
    ]
    bridge.append(
        {
            "ros_topic_name": "/rosclaw_sim/ground_truth",
            "gz_topic_name": "/actual_pose",
            "gz_type_name": "gz.msgs.Pose_V",
            "ros_type_name": "tf2_msgs/msg/TFMessage",
            "direction": "GZ_TO_ROS",
        }
    )
    (tmp_path / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    binding = {
        k: policy()[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
    }
    (tmp_path / "brush_binding.json").write_text(json.dumps(binding))
    (tmp_path / "physics_binding.json").write_text(
        json.dumps({**binding, "body_model_name": "anonymous_body", "world_name": "fixture_world"})
    )
    return tmp_path


def test_actual_prepared_source_reopen_is_exact_and_does_not_change_source_bytes(prepared):
    original = {p.name: p.read_bytes() for p in prepared.iterdir()}
    declaration = m.prepare_contact_policy(
        prepared, support_topics=["/contacts/support"], ground_collisions=["floor::ground::solid"]
    )
    assert declaration["contacts"] == policy()["contacts"]
    assert m.reopen_contact_policy(prepared, declaration) == declaration
    assert original == {p.name: p.read_bytes() for p in prepared.iterdir()}
    (prepared / "robot.sdf").write_text((prepared / "robot.sdf").read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        m.reopen_contact_policy(prepared, declaration)


@pytest.mark.parametrize(
    "fault",
    [
        "binding",
        "bridge",
        "missing_sensor",
        "foreign_collision",
        "source_escape",
        "entity",
        "duplicate_link",
        "foreign_bridge",
        "missing_pose_publisher",
        "wrong_pose_type",
    ],
)
def test_unresolved_prepared_contact_sources_are_not_silently_admitted(prepared, fault):
    if fault == "binding":
        (prepared / "brush_binding.json").write_text("{}")
    elif fault == "bridge":
        (prepared / "bridge.yaml").write_text("[]")
    elif fault in {"foreign_bridge", "wrong_pose_type"}:
        import yaml

        p = prepared / "bridge.yaml"
        mappings = yaml.safe_load(p.read_text())
        if fault == "foreign_bridge":
            mappings[0]["gz_topic_name"] = "/foreign"
        else:
            mappings[-1]["gz_type_name"] = "gz.msgs.Foreign"
        p.write_text(yaml.safe_dump(mappings))
    elif fault == "source_escape":
        (prepared / "robot.sdf").rename(prepared / "outside.sdf")
        (prepared / "robot.sdf").symlink_to(prepared / "outside.sdf")
    else:
        p = prepared / "robot.sdf"
        text = p.read_text()
        if fault == "missing_sensor":
            text = text.replace('type="contact"', 'type="unresolved"')
        elif fault == "foreign_collision":
            text = text.replace("<collision>solid</collision>", "<collision>foreign</collision>")
        elif fault == "entity":
            text = '<!DOCTYPE sdf [<!ENTITY e "value">]>' + text
        elif fault == "missing_pose_publisher":
            text = text.replace("gz::sim::systems::PosePublisher", "unresolved")
        elif fault == "duplicate_link":
            text = text.replace('<link name="base">', '<link name="support">')
        p.write_text(text)
    with pytest.raises(ValueError):
        m.prepare_contact_policy(
            prepared,
            support_topics=["/contacts/support"],
            ground_collisions=["floor::ground::solid"],
        )


def test_explicit_namespaced_independent_pose_source_reopens_without_name_guessing(prepared):
    import yaml

    p = prepared / "bridge.yaml"
    mappings = yaml.safe_load(p.read_text())
    mappings[-1]["ros_topic_name"] = "/sandbox/independent_pose"
    p.write_text(yaml.safe_dump(mappings))
    declaration = m.prepare_contact_policy(
        prepared,
        support_topics=["/contacts/support"],
        ground_collisions=["floor::ground::solid"],
        pose_topic="/sandbox/independent_pose",
    )
    assert declaration["pose_topic"] == "/sandbox/independent_pose"
    assert m.reopen_contact_policy(prepared, declaration) == declaration
