"""Prepare disjoint robot/probe contact producers in an exclusive SIM world.

This assembles source files only. No simulator, Node, scene service, actuator,
permission or Native task is started. Actual source ownership/admission and the
complete guarded launcher remain independent requirements.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from backend_observer_replay import BackendObserverReplay
from backend_probe_fixture import prepare_probe_fixture, validate_probe_declaration
from backend_probe_world import bounded_source, prepare_probe_world, xml_source
from native_contact_evidence import prepare_native_policy
from probe_scene_geometry import decode_scene_json


def observation_bridge_rows(raw):
    rows = yaml.safe_load(raw)
    if type(rows) is not list or len(rows) > 1023:
        raise ValueError("bounded declared scene bridge array required")
    result = []
    for row in rows:
        if type(row) is not dict or row.get("direction") != "GZ_TO_ROS" or "service_name" in row:
            raise ValueError("source bundle accepts observation-only typed bridge roles")
        row = row.copy()
        shorthand = row.pop("topic_name", None)
        if shorthand is not None:
            if "ros_topic_name" in row or "gz_topic_name" in row:
                raise ValueError("bridge topic_name is mutually exclusive with explicit roles")
            row["ros_topic_name"] = row["gz_topic_name"] = shorthand
        else:
            row.setdefault("ros_topic_name", row.get("gz_topic_name"))
            row.setdefault("gz_topic_name", row.get("ros_topic_name"))
        if any(
            type(row.get(k)) is not str or not 0 < len(row[k]) <= 512
            for k in (
                "ros_topic_name",
                "gz_topic_name",
                "ros_type_name",
                "gz_type_name",
            )
        ):
            raise ValueError("complete typed source bridge names required")
        result.append(row)
    return result


def disambiguate_world_visuals(world):
    def physical_bytes():
        copy = deepcopy(world)
        for link in copy.findall(".//link"):
            for visual in link.findall("visual"):
                link.remove(visual)
        return ET.canonicalize(ET.tostring(copy, encoding="unicode")).encode()

    before = physical_bytes()
    renamed = []
    for link in world.findall(".//link"):
        reserved = {e.get("name") for e in link if e.tag != "visual" and e.get("name")}
        for index, visual in enumerate(link.findall("visual")):
            old = visual.get("name")
            if old in reserved:
                if any(
                    value == old or value.endswith("::" + old)
                    for e in world.iter()
                    for key, value in e.attrib.items()
                    if key in {"relative_to", "attached_to"}
                ):
                    raise ValueError("ambiguous visual frame reference cannot be renamed safely")
                new = f"rosclaw_observer_visual_{index}_{old}"
                while new in reserved:
                    new += "_"
                visual.set("name", new)
                renamed.append({"link": link.get("name"), "original": old, "candidate": new})
            reserved.add(visual.get("name"))
    if physical_bytes() != before:
        raise ValueError("visual disambiguation changed nonvisual physical source")
    return renamed, hashlib.sha256(before).hexdigest()


def prepare_backend_world(
    output,
    *,
    scene_directory,
    declaration,
    contact_library,
    robot_support_topics,
    robot_ground_collisions,
    robot_pose_topic,
    robot_pose_frame,
    probe_pose_frame,
    instrument_service_binary=None,
    instrument_service_binary_sha256=None,
):
    d = validate_probe_declaration(declaration)
    source, output = Path(scene_directory), Path(output)
    if (instrument_service_binary is None) != (instrument_service_binary_sha256 is None):
        raise ValueError("complete original instrument service source binary/SHA required")
    service_binary = None
    if instrument_service_binary is not None:
        service_binary = bounded_source(instrument_service_binary, 100_000_000)
        if (
            not service_binary.startswith(b"\x7fELF")
            or hashlib.sha256(service_binary).hexdigest() != instrument_service_binary_sha256
        ):
            raise ValueError("actual frozen instrument service source binary SHA differs")
    inputs = {
        name: bounded_source(source / name)
        for name in (
            "world.sdf",
            "robot.sdf",
            "bridge.yaml",
            "physics_binding.json",
            "brush_binding.json",
        )
    }
    # Known fixture renderers keep their passive pose bridge in a separate
    # source file. Include it explicitly without changing either original.
    if (source / "truth_bridge.yaml").exists():
        inputs["truth_bridge.yaml"] = bounded_source(source / "truth_bridge.yaml")
    binding = decode_scene_json(inputs["physics_binding.json"])
    brush = decode_scene_json(inputs["brush_binding.json"])
    if (
        binding["run_id"] != d["run_id"]
        or binding["world_name"] != d["world_name"]
        or binding["body_model_name"] != d["robot_model_name"]
        or binding["grid"]["cleaning_polygon"] != d["cleaning_polygon"]
        or any(binding[k] != brush[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash"))
    ):
        raise ValueError("frozen Body/brush/scene/run/region declarations differ")
    robot = xml_source(inputs["robot.sdf"])
    models = robot.findall("model")
    if len(models) != 1 or models[0].get("name") != d["robot_model_name"]:
        raise ValueError("exact single materialized actual robot model source required")
    # Existing contact producers could duplicate or alias these source roles.
    world = xml_source(inputs["world.sdf"]).find("world")
    if world is None or any(
        p.get("name") == "rosclaw::PassiveContacts" for p in world.findall("plugin")
    ):
        raise ValueError("exclusive world contact producer source roles required")
    library = bounded_source(contact_library, 20_000_000)
    if not library.startswith(b"\x7fELF"):
        raise ValueError("explicit compiled native contact library bytes required")
    bridge = observation_bridge_rows(inputs["bridge.yaml"])
    if "truth_bridge.yaml" in inputs:
        bridge += observation_bridge_rows(inputs["truth_bridge.yaml"])
    if len(bridge) > 1023:
        raise ValueError("bounded combined observation bridge required")
    for role in ("ros_topic_name", "gz_topic_name"):
        names = [row[role].lstrip("/") for row in bridge]
        if len(set(names)) != len(names):
            raise ValueError("source bridge roles alias after conservative topic spelling check")
    component = {
        "ros_topic_name": "/rosclaw_sim/contact_components",
        "gz_topic_name": "/rosclaw_sim/contact_components",
        "ros_type_name": "std_msgs/msg/String",
        "gz_type_name": "gz.msgs.StringMsg",
        "direction": "GZ_TO_ROS",
    }
    matches = [
        row
        for row in bridge
        if row.get("ros_topic_name") == component["ros_topic_name"]
        or row.get("gz_topic_name") == component["gz_topic_name"]
    ]
    if matches and matches != [component]:
        raise ValueError("unambiguous exact robot native contact bridge required")
    if not matches:
        bridge.append(component)
    output.mkdir(parents=True, exist_ok=False)
    robot_directory = output / "robot-source"
    robot_directory.mkdir()
    for name, raw in inputs.items():
        (robot_directory / name).write_bytes(raw)
    (robot_directory / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    (robot_directory / "librosclaw_passive_contacts.so").write_bytes(library)
    robot_policy = prepare_native_policy(
        robot_directory,
        plugin_path=robot_directory / "librosclaw_passive_contacts.so",
        support_topics=robot_support_topics,
        ground_collisions=robot_ground_collisions,
        pose_topic=robot_pose_topic,
        all_physics_steps=True,
    )
    (robot_directory / "native-policy.json").write_text(json.dumps(robot_policy, indent=2) + "\n")
    instrument = output / "instrument-source"
    prepare_probe_fixture(instrument, d)
    (instrument / "librosclaw_passive_contacts.so").write_bytes(library)
    if service_binary is not None:
        service_path = instrument / "owned_instrument_service"
        service_path.write_bytes(service_binary)
        service_path.chmod(0o500)
    native = prepare_native_policy(
        instrument,
        plugin_path=instrument / "librosclaw_passive_contacts.so",
        support_topics=[d["contact_topic"]],
        ground_collisions=[d["ground_collision_name"]],
        pose_topic=d["pose_topic"],
        component_topic=d["component_topic"],
        component_gz_topic="/rosclaw_sim/backend_probe_components",
    )
    probe_policy = {
        "schema_version": "rosclaw.backend_cache_probe_policy.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "native_policy": native,
        **{k: d[k] for k in ("probe_xy", "sphere_radius_m", "ground_z_m", "lift_z_m")},
        "stable_samples": 5,
        "stable_sim_span_sec": 0.2,
        "refresh_sim_sec": 10,
        "refresh_wall_sec": 20,
    }
    (instrument / "probe-policy.json").write_text(json.dumps(probe_policy, indent=2) + "\n")
    candidate = output / "world-source"
    prepare_probe_world(
        candidate,
        scene_directory=robot_directory,
        instrument_directory=instrument,
        policy=probe_policy,
        plugin_path=instrument / "librosclaw_passive_contacts.so",
    )
    tree = ET.parse(candidate / "world.sdf")
    plugin = ET.SubElement(
        tree.find("world"),
        "plugin",
        filename="librosclaw_passive_contacts.so",
        name="rosclaw::PassiveContacts",
    )
    values = {
        **{k: brush[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")},
        "world_name": d["world_name"],
        "body_model_name": d["robot_model_name"],
        "topic": "/rosclaw_sim/contact_components",
    }
    for key, value in values.items():
        ET.SubElement(plugin, key).text = value
    renamed_visuals, physical_source_sha = disambiguate_world_visuals(tree.find("world"))
    tree.write(candidate / "world.sdf", encoding="utf-8", xml_declaration=True)
    (candidate / "probe-declaration.json").write_text(json.dumps(d, indent=2) + "\n")
    scene_binding = decode_scene_json((candidate / "physics_binding.json").read_bytes())
    engine = BackendObserverReplay(
        robot_policy,
        probe_policy,
        robot_pose_frame=robot_pose_frame,
        probe_pose_frame=probe_pose_frame,
        scene_binding=scene_binding,
        probe_declaration=d,
        instrument_service_binary_sha256=instrument_service_binary_sha256,
    )
    observer = output / "observer-source"
    observer.mkdir()
    config = {
        "run_id": brush["run_id"],
        "body_snapshot_hash": brush["body_snapshot_hash"],
        "constraint_policy_hash": engine.gate.policy_hash,
    }
    (observer / "backend_actor_constraint.json").write_text(json.dumps(config) + "\n")
    result = {
        "schema_version": "rosclaw.backend_world_source_bundle.v1",
        "status": "SOURCE_PREPARED_NOT_RUNTIME_ADMISSION",
        "source_hashes": {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()},
        "source_contact_library_sha256": hashlib.sha256(library).hexdigest(),
        "instrument_service_binary_sha256": instrument_service_binary_sha256,
        "candidate_visual_name_repairs": renamed_visuals,
        "nonvisual_world_source_sha256_before_and_after_visual_repair": physical_source_sha,
        "output_hashes": {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "constraint_policy_hash": engine.gate.policy_hash,
        "robot_sampling": "ALL_POSTUPDATE_PHYSICS_STEPS",
        "probe_sampling_hz": 20,
        "actual_world_source_ownership_admitted": False,
        "plugin_search_path": "REQUIRES_EXPLICIT_OWNED_WORLD_AND_ORIGINAL_PHYSICS_LIBRARY_PATH",
        "original_probe_world_manifest": "SUBSTAGE_ONLY_ROBOT_PLUGIN_ADDED_BY_FINAL_BUNDLE",
        "controller_IPC": "NOT_JOINED",
        "qualified_Native_launcher": "NOT_JOINED",
        "backend_health_admitted": False,
        "physical_acceptance": "NOT_RUN",
        "authorization": False,
    }
    (output / "backend-world-bundle.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
