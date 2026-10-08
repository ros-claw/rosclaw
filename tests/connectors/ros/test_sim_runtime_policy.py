"""Real compiler/reopen on synthetic Graph; no chosen heldout or physical run."""

import hashlib
import json
from copy import deepcopy
from datetime import timedelta

import pytest
import yaml

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.context.sim_execution_config import prepare_sim_execution_config
from rosclaw.connectors.ros.context.sim_runtime_policy import (
    _observed_topic,
    load_frozen_sim_runtime_policy,
    prepare_sim_runtime_policy,
)
from rosclaw.connectors.ros.intelligence.system_model import Signal
from tests.connectors.ros import test_sim_execution_config as execution
from tests.connectors.ros.test_sim_execution_interfaces import NOW, execution_example


@pytest.fixture
def runtime(request, monkeypatch, tmp_path):
    namespace = "/different/robot"
    topics = {
        "observation": namespace + "/independent",
        "physics": namespace + "/actual_components",
        "brush_events": namespace + "/brush_events",
        "cleaning_state": namespace + "/brush_active",
        "localization": namespace + "/measured_pose",
        "map": namespace + "/floor",
        "nav_velocity": namespace + "/guarded",
        "drive_velocity": namespace + "/driver_input",
        "independent_pose": namespace + "/measured_world_pose",
    }

    def augmented(*args):
        model, data, attachment, policy = execution_example(*args)
        data = data.replace(b"<collision>", b'<collision name="chassis_collision">')
        for description in model.observations["urdf_descriptions"].values():
            description.update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        policy["source_urdf_sha256"] = hashlib.sha256(data).hexdigest()
        for name, kind in [
            (topics["physics"], "std_msgs/msg/String"),
            (topics["brush_events"], "std_msgs/msg/String"),
            (topics["drive_velocity"], "geometry_msgs/msg/TwistStamped"),
            (topics["independent_pose"], "tf2_msgs/msg/TFMessage"),
            (namespace + "/ground_contact", "ros_gz_interfaces/msg/Contacts"),
        ]:
            model.graph["topics"].append({"name": name, "msg_type": kind})
            model.signals.append(
                Signal(
                    topic=name,
                    source="synthetic",
                    captured_at=NOW,
                    publisher_count=1,
                    last_message_age_ms=10,
                )
            )
        model.seal()
        policy["source_snapshot_hash"] = model.snapshot_hash
        return model, data, attachment, policy

    monkeypatch.setattr(execution, "execution_example", augmented)
    workspace, model, data, kwargs = request.getfixturevalue("sources")
    root = tmp_path / "fixture"
    root.mkdir()
    workspace.rename(root / "home")
    (root / "robot.urdf").write_bytes(data)
    (root / "bridge.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "ros_topic_name": namespace + "/ground_contact",
                    "gz_topic_name": "/actual/contact",
                    "ros_type_name": "ros_gz_interfaces/msg/Contacts",
                    "gz_type_name": "gz.msgs.Contacts",
                    "direction": "GZ_TO_ROS",
                }
            ]
        )
    )
    admission = prepare_sim_execution_config(root / "home", model, data, **kwargs)
    declaration = {
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "world_name": "renamed_world",
        "body_model_name": "unselected_synthetic_model",
        "model_base_identity_approved": True,
        "model_base_identity_source": "simulator_operator_fixture_policy",
        "map_world_identity_approved": True,
        "world_to_map_xyyaw": [0, 0, 0],
        "controller_watchdog_approved": True,
        "topics": topics,
        "collision_streams": [
            {
                "link": "chassis",
                "collision_index": 0,
                "topic": namespace + "/ground_contact",
                "collision_name": "chassis_collision",
                "sensor_name": "body_contact",
                "gz_topic": "/actual/contact",
            }
        ],
        "support_contact_topics": [namespace + "/ground_contact"],
        "ground_model_names": ["renamed_ground"],
    }
    return root, admission, model, declaration


# Reuse source fixture while adding the runtime streams before Body compilation.
sources = execution.sources


def test_namespaced_runtime_reopens_actual_compiled_sources_without_profile(runtime):
    root, admission, model, policy = runtime
    before = deepcopy(policy)
    result = prepare_sim_runtime_policy(root, admission, model, policy, now=NOW)
    assert policy == before
    assert result["body"]["base_frame"] == "chassis"
    assert result["policy"]["body_model_name"] == "unselected_synthetic_model"
    assert len(result["contact_topics"]) == 1  # no two-wheel/model-name assumption
    assert result["endpoints"]["hold"] == "/different/robot/hold"
    assert result["physical_acceptance"] == "NOT_RUN"
    assert not result["actions_dispatched"] and not result["usable_for_real_execution"]
    assert result["requires_actual_component_geometry_and_contact_mapping_admission"]
    assert not (root / "sim_runtime_policy.json").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "old",
        "unsealed",
        "source",
        "real",
        "base_identity",
        "map_identity",
        "transform",
        "watchdog",
        "model",
        "missing_role",
        "relative",
        "duplicate_roles",
        "wrong_body_topic",
        "wrong_type",
        "publisher",
        "stale_signal",
        "missing_collision",
        "foreign_collision",
        "duplicate_collision",
        "support",
        "ground",
        "extra",
        "urdf",
    ],
)
def test_runtime_refuses_missing_stale_guessed_or_substituted_contract(runtime, fault):
    root, admission, model, policy = runtime
    now = NOW
    if fault == "old":
        now += timedelta(seconds=6)
    elif fault == "unsealed":
        model.graph["topics"].pop()
    elif fault == "source":
        admission["source_snapshot_hash"] = "foreign"
    elif fault == "real":
        policy["evidence_domain"] = "REAL"
    elif fault == "base_identity":
        policy["model_base_identity_approved"] = False
    elif fault == "map_identity":
        policy["map_world_identity_approved"] = False
    elif fault == "transform":
        policy["world_to_map_xyyaw"] = [1, 0, 0]
    elif fault == "watchdog":
        policy["controller_watchdog_approved"] = False
    elif fault == "model":
        policy["body_model_name"] = "bad/model"
    elif fault == "missing_role":
        policy["topics"].pop("physics")
    elif fault == "relative":
        policy["topics"]["physics"] = "relative"
    elif fault == "duplicate_roles":
        policy["topics"]["physics"] = policy["topics"]["brush_events"]
    elif fault == "wrong_body_topic":
        policy["topics"]["nav_velocity"] = "/foreign"
    elif fault in ("wrong_type", "publisher", "stale_signal"):
        if fault == "wrong_type":
            model.graph["topics"][-1]["msg_type"] = "std_msgs/msg/String"
        elif fault == "publisher":
            model.signals[-1].publisher_count = 2
        else:
            model.signals[-1].last_message_age_ms = 300
        model.seal()
    elif fault == "missing_collision":
        policy["collision_streams"] = []
    elif fault == "foreign_collision":
        policy["collision_streams"][0]["link"] = "foreign"
    elif fault == "duplicate_collision":
        policy["collision_streams"].append(deepcopy(policy["collision_streams"][0]))
    elif fault == "support":
        policy["support_contact_topics"] = []
    elif fault == "ground":
        policy["ground_model_names"] = [policy["body_model_name"]]
    elif fault == "extra":
        policy["guess_namespace"] = "/other"
    elif fault == "urdf":
        (root / "robot.urdf").write_bytes(b"foreign")
    with pytest.raises(ValueError):
        prepare_sim_runtime_policy(root, admission, model, policy, now=now)
    assert not (root / "sim_runtime_policy.json").exists()


@pytest.fixture
def frozen_runtime(runtime):
    root, admission, model, declaration = runtime
    policy = prepare_sim_runtime_policy(root, admission, model, declaration, now=NOW)
    evidence = (
        BodyResolver(workspace=root / "home")
        .get_effective_body(recompile_if_stale=False)
        .provider_interfaces["sim_fixture_evidence"]
    )
    brush = {
        "run_id": policy["run_id"],
        "body_snapshot_hash": policy["body_snapshot_hash"],
        "attachment_hash": evidence["attachment_hash"],
        "producer_id": "synthetic_producer",
    }
    binding = {
        **{k: brush[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")},
        "mission_id": policy["mission_id"],
        "grid": policy["grid"],
        "world_name": declaration["world_name"],
        "body_model_name": declaration["body_model_name"],
        "runtime_policy_hash": policy["artifact_hash"],
        "source_urdf_sha256": policy["source_urdf_sha256"],
        "maximum_body_planar_radius_m": policy["body"]["physical_radius_m"],
        "body_reference_link": policy["body"]["base_frame"],
        "body_contact_mapping": policy["body_contact_mapping"],
        "model_base_identity_approved": True,
        "model_base_identity_source": "simulator_operator_fixture_policy",
        "world_to_map_xyyaw": [0, 0, 0],
        "map_world_identity_approved": True,
        "frame_transform_source": "simulator_operator_fixture_policy",
        "obstacle_names": ["synthetic_obstacle"],
        "scene_model_names": [
            declaration["body_model_name"],
            "renamed_ground",
            "synthetic_obstacle",
        ],
    }
    for name, value in (
        ("sim_runtime_policy.json", policy),
        ("generic_execution_proposal.json", admission),
        ("snapshot.json", model.to_dict()),
        ("brush_binding.json", brush),
        ("physics_binding.json", binding),
    ):
        (root / name).write_text(json.dumps(value))
    (root / "run_id.txt").write_text(policy["run_id"])
    return root, policy


def test_runtime_loader_reopens_all_archived_sources_without_live_claim(frozen_runtime):
    root, policy = frozen_runtime
    assert load_frozen_sim_runtime_policy(root) == policy
    assert policy["physical_acceptance"] == "NOT_RUN"


@pytest.mark.parametrize(
    "file,key,value",
    [
        ("sim_runtime_policy.json", "artifact_hash", "foreign"),
        ("brush_binding.json", "attachment_hash", "foreign"),
        ("physics_binding.json", "runtime_policy_hash", "foreign"),
        ("physics_binding.json", "maximum_body_planar_radius_m", 0.01),
        ("physics_binding.json", "model_base_identity_approved", 1),
        ("physics_binding.json", "grid", {}),
    ],
)
def test_runtime_loader_refuses_source_substitutions(frozen_runtime, file, key, value):
    root, _ = frozen_runtime
    path = root / file
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_frozen_sim_runtime_policy(root)


def test_runtime_loader_refuses_missing_generic_policy_and_retains_legacy(tmp_path):
    assert load_frozen_sim_runtime_policy(tmp_path) is None
    (tmp_path / "execution_config.json").write_text('{"generic_execution_proposal":{}}')
    with pytest.raises(ValueError, match="requires its frozen"):
        load_frozen_sim_runtime_policy(tmp_path)


@pytest.mark.parametrize("fault", ["wrong_type", "publisher", "stale_signal", "duplicate"])
def test_observed_runtime_stream_checks_actual_types_counts_ages_and_uniqueness(runtime, fault):
    _, _, model, declaration = runtime
    topic = declaration["support_contact_topics"][0]
    if fault == "wrong_type":
        model.graph["topics"][-1]["msg_type"] = "std_msgs/msg/String"
    elif fault == "publisher":
        model.signals[-1].publisher_count = 2
    elif fault == "stale_signal":
        model.signals[-1].last_message_age_ms = 300
    else:
        model.graph["topics"].append(deepcopy(model.graph["topics"][-1]))
    with pytest.raises(ValueError, match="unique typed actual runtime stream"):
        _observed_topic(model, topic, "ros_gz_interfaces/msg/Contacts", NOW, fresh=True)


def test_actual_generic_observer_constructor_uses_frozen_names_without_drive_authority(
    frozen_runtime, monkeypatch
):
    from tests.connectors.ros.test_split_sim_actuator import FakePath, load_node

    root, policy = frozen_runtime
    brush = json.loads((root / "brush_binding.json").read_text())
    monkeypatch.setattr(
        FakePath,
        "extra",
        {
            "/evidence/physics_binding.json": (root / "physics_binding.json").read_text(),
        },
    )
    node = load_node(
        "witness.py",
        "Witness",
        monkeypatch,
        dynamic=True,
        runtime_policy=load_frozen_sim_runtime_policy(root),
        brush_binding=brush,
    )
    assert set(node.publishers) == {policy["policy"]["topics"]["observation"]}
    assert not node.services and node.velocity is None
    assert node.profile.simulation_model == "unselected_synthetic_model"
    assert node.base_frame == "chassis"
    assert node.support_topics == ["/different/robot/ground_contact"]
    assert set(node.subscriptions) == {
        policy["policy"]["topics"][k] for k in ("physics", "brush_events", "localization", "map")
    } | set(policy["contact_topics"])
    assert "/nav_cmd_vel" not in node.subscriptions


def test_actual_generic_observer_refuses_legacy_pose_source(frozen_runtime, monkeypatch):
    from tests.connectors.ros.test_split_sim_actuator import load_node

    root, policy = frozen_runtime
    brush = json.loads((root / "brush_binding.json").read_text())
    with pytest.raises(ValueError, match="split actuator and actual component"):
        load_node(
            "witness.py",
            "Witness",
            monkeypatch,
            dynamic=False,
            runtime_policy=policy,
            brush_binding=brush,
        )


def test_actual_generic_contact_handler_excludes_only_explicit_ground_model(
    frozen_runtime, monkeypatch
):
    from types import SimpleNamespace

    from tests.connectors.ros.test_split_sim_actuator import FakePath, load_node

    root, policy = frozen_runtime
    brush = json.loads((root / "brush_binding.json").read_text())
    monkeypatch.setattr(
        FakePath,
        "extra",
        {
            "/evidence/physics_binding.json": (root / "physics_binding.json").read_text(),
        },
    )
    node = load_node(
        "witness.py",
        "Witness",
        monkeypatch,
        dynamic=True,
        runtime_policy=policy,
        brush_binding=brush,
    )
    topic = policy["contact_topics"][0]

    def message(name):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
            contacts=[
                SimpleNamespace(
                    collision1=SimpleNamespace(
                        name="unselected_synthetic_model::chassis::chassis_collision"
                    ),
                    collision2=SimpleNamespace(name=name),
                )
            ],
        )

    node.contacts(topic, message("renamed_ground::surface::collision"))
    assert node.physics_collision_count == 0
    node.contacts(topic, message("foreign_renamed_ground::wall::collision"))
    assert node.physics_collision_count == 1


def test_actual_generic_actuator_uses_declared_services_and_lease_stop(frozen_runtime, monkeypatch):
    import time
    from types import SimpleNamespace

    from tests.connectors.ros.test_split_sim_actuator import FakePath, load_node

    root, policy = frozen_runtime
    brush = json.loads((root / "brush_binding.json").read_text())
    monkeypatch.setattr(
        FakePath,
        "extra",
        {
            "/evidence/brush_binding.json": json.dumps(brush),
            "/evidence/run_id.txt": brush["run_id"],
        },
    )
    actor = load_node("sim_actuator.py", "SimulationActuator", monkeypatch, runtime_policy=policy)
    assert set(actor.services) == {policy["endpoints"][k] for k in ("lease", "hold", "cleaning")}
    assert set(actor.subscriptions) == {policy["policy"]["topics"]["nav_velocity"]}
    assert set(actor.publishers) == {
        policy["policy"]["topics"][k] for k in ("drive_velocity", "cleaning_state", "brush_events")
    }
    assert not actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace()).success
    actor.tick()
    actor.heartbeat(SimpleNamespace(data=True), SimpleNamespace())
    actor.sim_time = 1.1
    assert actor.set_cleaning(SimpleNamespace(data=True), SimpleNamespace()).success
    actor.lease = time.monotonic() - 1
    actor.sim_time = 1.2
    actor.tick()
    assert not actor.cleaning
    assert actor.publishers[policy["policy"]["topics"]["drive_velocity"]]


@pytest.mark.parametrize(
    "case,expected",
    [
        ("actual_collision_components", "v3"),
        ("v2_actual_body_collision_with_joint", "v3"),
        ("v3_actual_body_reference_identity", "v4"),
        ("v4_actual_contact_sensor_collision_mapping", "exceeds"),
    ],
)
def test_actual_generic_observer_refuses_missing_or_oversized_component_body_before_credit(
    frozen_runtime, monkeypatch, case, expected
):
    import time
    from pathlib import Path
    from types import SimpleNamespace

    from tests.connectors.ros.test_split_sim_actuator import FakePath, load_node

    root, policy = frozen_runtime
    brush = json.loads((root / "brush_binding.json").read_text())
    monkeypatch.setattr(
        FakePath,
        "extra",
        {
            "/evidence/physics_binding.json": (root / "physics_binding.json").read_text(),
        },
    )
    observer = load_node(
        "witness.py",
        "Witness",
        monkeypatch,
        dynamic=True,
        runtime_policy=policy,
        brush_binding=brush,
    )
    rows = [
        json.loads(line)
        for line in (
            Path(__file__).parent / "fixtures/passive-ecm-contact-v4-contract-packets.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    packet = deepcopy(next(row["packet"] for row in rows if row["case"] == case))
    packet.update({k: brush[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")})
    packet.update(
        world_name=policy["policy"]["world_name"], paused=False, captured_at_unix_ns=time.time_ns()
    )
    packet["body"]["model_name"] = policy["policy"]["body_model_name"]
    if "reference_link" in packet["body"]:
        packet["body"]["reference_link"]["name"] = policy["body"]["base_frame"]
    if "contact_sources" in packet["body"]:
        for row in packet["body"]["contact_sources"]:
            row.update(policy["body_contact_mapping"][0])
    packet["obstacles"][0]["model_name"] = "synthetic_obstacle"
    packet["scene_models"] = [
        {
            "model_name": policy["policy"]["body_model_name"],
            "entity_id": packet["body"]["entity_id"],
        },
        {"model_name": "synthetic_obstacle", "entity_id": packet["obstacles"][0]["entity_id"]},
        {"model_name": "renamed_ground", "entity_id": 1000},
    ]
    observer.physics_event(SimpleNamespace(data=json.dumps(packet)))
    assert observer.physics_fault and expected in observer.physics_fault
    assert observer.physics_projector is None and observer.pose is None
    assert not observer.physics_queue
    assert not observer.publishers[policy["policy"]["topics"]["observation"]]


@pytest.mark.parametrize(
    "fault",
    [
        "collision_name",
        "sensor_name",
        "gz_topic",
        "bridge_topic",
        "bridge_direction",
        "duplicate_bridge",
        "missing_bridge",
    ],
)
def test_actual_contact_bridge_and_captured_collision_mapping_must_agree(runtime, fault):
    root, admission, model, policy = runtime
    if fault in {"collision_name", "sensor_name", "gz_topic"}:
        policy["collision_streams"][0][fault] = "foreign" if fault != "gz_topic" else "/foreign"
    else:
        bridge = yaml.safe_load((root / "bridge.yaml").read_text())
        if fault == "bridge_topic":
            bridge[0]["gz_topic_name"] = "/foreign"
        elif fault == "bridge_direction":
            bridge[0]["direction"] = "ROS_TO_GZ"
        elif fault == "duplicate_bridge":
            bridge.append(deepcopy(bridge[0]))
        else:
            bridge.clear()
        (root / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    # A different legitimate sensor name alone is a proposal, and is checked
    # against actual components at live admission, never certified by Graph.
    if fault == "sensor_name":
        policy["collision_streams"][0][fault] = "invalid/name"
    with pytest.raises(ValueError):
        prepare_sim_runtime_policy(root, admission, model, policy, now=NOW)


def test_runtime_reopen_refuses_changed_contact_bridge_bytes(frozen_runtime):
    root, _ = frozen_runtime
    with (root / "bridge.yaml").open("a") as stream:
        stream.write("\n# source bytes changed\n")
    with pytest.raises(ValueError):
        load_frozen_sim_runtime_policy(root)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_collision",
        "missing_header",
        "old",
        "future",
        "reversed",
        "empty_support",
        "boolean_stamp",
    ],
)
def test_actual_generic_contact_fault_is_latched_on_wrong_identity_or_time(
    frozen_runtime, monkeypatch, fault
):
    from types import SimpleNamespace

    from tests.connectors.ros.test_split_sim_actuator import FakePath, load_node

    root, policy = frozen_runtime
    brush = json.loads((root / "brush_binding.json").read_text())
    monkeypatch.setattr(
        FakePath,
        "extra",
        {
            "/evidence/physics_binding.json": (root / "physics_binding.json").read_text(),
        },
    )
    node = load_node(
        "witness.py",
        "Witness",
        monkeypatch,
        dynamic=True,
        runtime_policy=policy,
        brush_binding=brush,
    )
    topic = policy["contact_topics"][0]
    good = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
        contacts=[
            SimpleNamespace(
                collision1=SimpleNamespace(
                    name="unselected_synthetic_model::chassis::chassis_collision"
                ),
                collision2=SimpleNamespace(name="renamed_ground::surface::collision"),
            )
        ],
    )
    node.contacts(topic, good)
    assert node.contact_fault is None
    bad = deepcopy(good)
    if fault == "wrong_collision":
        bad.contacts[0].collision1.name = "unselected_synthetic_model::chassis::foreign"
    elif fault == "missing_header":
        del bad.header
    elif fault == "old":
        bad.header.stamp.sec = 0
    elif fault == "future":
        bad.header.stamp.sec = 2
    elif fault == "reversed":
        bad.header.stamp.sec, bad.header.stamp.nanosec = 0, 999_999_999
    elif fault == "empty_support":
        bad.contacts.clear()
    else:
        bad.header.stamp.sec = True
    node.contacts(topic, bad)
    latched = node.contact_fault
    assert latched and node.physics_collision_count == 0
    node.contacts(topic, good)
    assert node.contact_fault == latched


def test_actual_generic_component_contact_ids_cannot_change_after_initial_admission(
    frozen_runtime, monkeypatch
):
    import math
    import time
    from types import SimpleNamespace

    from tests.connectors.ros.test_physics_contact_components_v4 import packet as sdk_packet
    from tests.connectors.ros.test_split_sim_actuator import FakePath, load_node

    root, policy = frozen_runtime
    brush = json.loads((root / "brush_binding.json").read_text())
    monkeypatch.setattr(
        FakePath,
        "extra",
        {
            "/evidence/physics_binding.json": (root / "physics_binding.json").read_text(),
        },
    )
    node = load_node(
        "witness.py",
        "Witness",
        monkeypatch,
        dynamic=True,
        runtime_policy=policy,
        brush_binding=brush,
    )
    packet = sdk_packet()  # SDK source with explicit synthetic geometry/policy substitution.
    packet.update({k: brush[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")})
    packet.update(
        world_name=policy["policy"]["world_name"], paused=False, captured_at_unix_ns=time.time_ns()
    )
    packet["body"]["model_name"] = policy["policy"]["body_model_name"]
    packet["body"]["reference_link"]["name"] = policy["body"]["base_frame"]
    packet["body"]["contact_sources"][0].update(policy["body_contact_mapping"][0])
    geometry = packet["body"]["collision_geometry"][0]
    geometry["size"] = [0.4, 0.2, 0.1]
    geometry["enclosing_radius_m"] = math.sqrt(0.2**2 + 0.1**2 + 0.05**2)
    packet["obstacles"][0]["model_name"] = "synthetic_obstacle"
    packet["scene_models"] = [
        {
            "model_name": policy["policy"]["body_model_name"],
            "entity_id": packet["body"]["entity_id"],
        },
        {"model_name": "synthetic_obstacle", "entity_id": packet["obstacles"][0]["entity_id"]},
        {"model_name": "renamed_ground", "entity_id": 1000},
    ]
    node.physics_event(SimpleNamespace(data=json.dumps(packet)))
    assert node.physics_fault is None and len(node.body_contact_mapping_hash) == 64
    packet.update(sequence=1, sim_time_sec=0.2, captured_at_unix_ns=time.time_ns())
    packet["body"]["contact_sources"][0]["sensor_entity_id"] += 100
    node.physics_event(SimpleNamespace(data=json.dumps(packet)))
    assert "contact component identities changed" in node.physics_fault
