"""Prepare an exclusive world-source candidate containing a SIM instrument.

No simulator or scene service is started. Runtime admission still requires
actual Body clearance, observed component sources and a closed probe replay.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from backend_probe_evidence import probe_policy
from backend_probe_fixture import validate_probe_declaration
from native_contact_evidence import reopen_native_policy

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.verification.occupancy_geometry import seal_obstacle_geometry


def bounded_source(path, limit=2_000_000):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("bounded regular owned source file required")
    return path.read_bytes()


def xml_source(raw):
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper() or b"\x00" in raw:
        raise ValueError("plain bounded UTF-8 source SDF required")
    root = ET.fromstring(raw.decode("utf-8"))
    if root.tag != "sdf" or any("{" in e.tag for e in root.iter()):
        raise ValueError("plain source SDF document required")
    return root


def support_plane(world, declaration):
    names = declaration["ground_collision_name"].split("::")
    if len(names) != 3:
        raise ValueError("explicit direct-model support collision identity required")
    models = [m for m in world.findall("model") if m.get("name") == names[0]]
    if len(models) != 1 or models[0].findtext("static") != "true":
        raise ValueError("one actual static support plane source required")
    model = models[0]
    links = [link for link in model.findall("link") if link.get("name") == names[1]]
    if len(links) != 1:
        raise ValueError("unambiguous actual support link required")
    collisions = [c for c in links[0].findall("collision") if c.get("name") == names[2]]
    if len(collisions) != 1:
        raise ValueError("unambiguous actual support collision required")
    collision = collisions[0]
    # Complex frames, rotated planes and terrain need another explicit adapter.
    for element in (model, links[0], collision):
        poses = element.findall("pose")
        if len(poses) > 1 or (
            poses and (poses[0].attrib or (poses[0].text or "").split() != ["0"] * 6)
        ):
            raise ValueError("support source pose unresolved; no guessed ground transform")
    geometry = collision.find("geometry")
    plane = geometry.find("plane") if geometry is not None else None
    if (
        geometry is None
        or len(geometry) != 1
        or plane is None
        or plane.findtext("normal") != "0 0 1"
    ):
        raise ValueError("explicit horizontal source support plane required")
    try:
        size = [float(v) for v in plane.findtext("size", "").split()]
    except ValueError as exc:
        raise ValueError("bounded actual support plane size required") from exc
    if len(size) != 2 or any(not 0 < v <= 100 for v in size) or declaration["ground_z_m"] != 0:
        raise ValueError("bounded resolved actual support plane required")
    r = declaration["sphere_radius_m"]
    if any(abs(v) + r + 0.1 >= s / 2 for v, s in zip(declaration["probe_xy"], size, strict=True)):
        raise ValueError("isolated probe must remain within declared support footprint")


def prepare_probe_world(output, *, scene_directory, instrument_directory, policy, plugin_path):
    """Write new source artifacts; leave original frozen scene bytes unchanged."""
    probe_policy(policy)
    scene, instrument, output = map(Path, (scene_directory, instrument_directory, output))
    sources = {
        "world.sdf": bounded_source(scene / "world.sdf"),
        "bridge.yaml": bounded_source(scene / "bridge.yaml"),
        "physics_binding.json": bounded_source(scene / "physics_binding.json"),
        "instrument.sdf": bounded_source(instrument / "robot.sdf"),
        "instrument_bridge.yaml": bounded_source(instrument / "bridge.yaml"),
        "probe-fixture.json": bounded_source(instrument / "probe-fixture.json"),
    }
    fixture = json.loads(sources["probe-fixture.json"])
    declaration = validate_probe_declaration(fixture["declaration"])
    binding = json.loads(sources["physics_binding.json"])
    native = policy["native_policy"]
    reopen_native_policy(instrument, native, plugin_path=plugin_path)
    transform = binding.get("world_to_map_xyyaw")
    if (
        type(transform) is not list
        or len(transform) != 3
        or any(type(v) not in (int, float) or v != 0 for v in transform)
        or binding.get("frame_transform_source") != "simulator_operator_fixture_policy"
        or binding.get("map_world_identity_approved") is not True
    ):
        raise ValueError(
            "explicit approved world/map identity required; transformed polygon adapter not admitted"
        )
    for name, raw in (
        ("robot.sdf", sources["instrument.sdf"]),
        ("bridge.yaml", sources["instrument_bridge.yaml"]),
    ):
        if fixture["files"][name]["sha256"] != hashlib.sha256(raw).hexdigest() or fixture["files"][
            name
        ]["size_bytes"] != len(raw):
            raise ValueError("instrument original source bytes differ")
    if (
        declaration["run_id"] != binding["run_id"]
        or declaration["world_name"] != binding["world_name"]
        or declaration["robot_model_name"] != binding["body_model_name"]
        or declaration["probe_model_name"] != native["contact_policy"]["model_name"]
        or any(
            policy[k] != declaration[k]
            for k in ("probe_xy", "sphere_radius_m", "ground_z_m", "lift_z_m")
        )
        or declaration["cleaning_polygon"] != binding["grid"]["cleaning_polygon"]
    ):
        raise ValueError("instrument and frozen world/robot/run/region bindings differ")
    original = xml_source(sources["world.sdf"])
    worlds = original.findall("world")
    if len(worlds) != 1 or worlds[0].get("name") != binding["world_name"]:
        raise ValueError("one explicitly bound actual world source required")
    world = worlds[0]
    if world.findall("include") or any(
        m.find("include") is not None or m.find("model") is not None for m in world.findall("model")
    ):
        raise ValueError("unresolved whole-world model sources refused")
    names = [m.get("name") for m in world.findall("model")]
    expected = set(binding["scene_model_names"]) - {binding["body_model_name"]}
    if len(set(names)) != len(names) or set(names) != expected:
        raise ValueError("pre-spawn scene source inventory differs from frozen binding")
    name = declaration["probe_model_name"]
    if name in binding["scene_model_names"] or len(binding["obstacle_names"]) >= 32:
        raise ValueError("exclusive bounded instrument scene identity required")
    support_plane(world, declaration)
    required = {
        "gz::sim::systems::Physics",
        "gz::sim::systems::Contact",
        "gz::sim::systems::UserCommands",
    }
    if not required.issubset({p.get("name") for p in world.findall("plugin")}):
        raise ValueError("explicit Physics, Contact and owned scene service sources required")
    if any(sum(p.get("name") == name for p in world.findall("plugin")) != 1 for name in required):
        raise ValueError("ambiguous duplicate required simulation system source")
    passive = [p for p in world.findall("plugin") if p.get("name") == "rosclaw::PassivePhysics"]
    if len(passive) != 1 or passive[0].findtext("body_model_name") != binding["body_model_name"]:
        raise ValueError("one source-bound passive occupancy plugin required")
    if any(
        passive[0].findtext(key) != binding[key]
        for key in ("run_id", "body_snapshot_hash", "attachment_hash")
    ):
        raise ValueError("passive occupancy plugin source identities differ")
    declared = [
        element.text
        for key in ("static_model", "obstacle_model")
        for element in passive[0].findall(key)
    ]
    if len(set(declared)) != len(declared) or set(declared) != set(names):
        raise ValueError("passive source scene declarations differ from actual model sources")
    ground_name = declaration["ground_collision_name"].split("::")[0]
    other_names = tuple(model_name for model_name in names if model_name != ground_name)
    if other_names:
        radii = dict(
            seal_obstacle_geometry(sources["world.sdf"], obstacle_names=other_names).model_radii
        )
        for model in world.findall("model"):
            model_name = model.get("name")
            if model_name == ground_name:
                continue
            poses = model.findall("pose")
            if len(poses) > 1 or (poses and poses[0].attrib):
                raise ValueError("scene model source placement unresolved")
            pose = [float(v) for v in model.findtext("pose", "0 0 0 0 0 0").split()]
            if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
                raise ValueError("finite actual scene model source pose required")
            if (
                math.dist(pose[:2], declaration["probe_xy"])
                <= radii[model_name] + declaration["sphere_radius_m"] + 0.2
            ):
                raise ValueError("instrument source placement lacks full scene collision clearance")
    # The sphere is an observed obstacle, never a falsely static or hidden model.
    ET.SubElement(passive[0], "obstacle_model").text = name
    probe_root = xml_source(sources["instrument.sdf"])
    models = probe_root.findall("model")
    if len(models) != 1 or models[0].get("name") != name:
        raise ValueError("exact explicit instrument model source required")
    world.append(copy.deepcopy(models[0]))
    plugin = ET.SubElement(
        world, "plugin", filename="librosclaw_passive_contacts.so", name="rosclaw::PassiveContacts"
    )
    for key, value in fixture["binding"].items():
        ET.SubElement(plugin, key).text = value
    ET.SubElement(plugin, "topic").text = "/rosclaw_sim/backend_probe_components"
    library = bounded_source(plugin_path, 20_000_000)
    if (
        not library.startswith(b"\x7fELF")
        or hashlib.sha256(library).hexdigest() != native["plugin_sha256"]
    ):
        raise ValueError("source-pinned compiled native contact library required")
    old_bridge, probe_bridge = [
        yaml.safe_load(sources[k]) for k in ("bridge.yaml", "instrument_bridge.yaml")
    ]
    if type(old_bridge) is not list or type(probe_bridge) is not list:
        raise ValueError("explicit bridge source arrays required")
    for role in ("ros_topic_name", "gz_topic_name"):
        topics = [row[role] for row in old_bridge + probe_bridge]
        if len(set(topics)) != len(topics):
            raise ValueError("instrument observer endpoints alias existing scene roles")
    merged_binding = copy.deepcopy(binding)
    merged_binding["obstacle_names"].append(name)
    merged_binding["scene_model_names"].append(name)
    world_raw = ET.tostring(original, encoding="utf-8", xml_declaration=True)
    seal_obstacle_geometry(world_raw, obstacle_names=tuple(merged_binding["obstacle_names"]))
    outputs = {
        "world.sdf": world_raw,
        "bridge.yaml": yaml.safe_dump(old_bridge + probe_bridge).encode(),
        "physics_binding.json": json.dumps(merged_binding, indent=2).encode(),
        "librosclaw_passive_contacts.so": library,
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename, raw in outputs.items():
        (output / filename).write_bytes(raw)
    result = {
        "schema_version": "rosclaw.backend_probe_world_candidate.v1",
        "status": "SOURCE_PREPARED_RUNTIME_ADMISSION_NOT_IMPLEMENTED",
        "probe_role": "DYNAMIC_OBSERVED_INSTRUMENT_OBSTACLE_OUTSIDE_WORK_REGION",
        "source_hashes": {key: hashlib.sha256(raw).hexdigest() for key, raw in sources.items()},
        "output_hashes": {key: hashlib.sha256(raw).hexdigest() for key, raw in outputs.items()},
        "probe_policy_hash": digest(policy),
        "actual_body_clearance_admitted": False,
        "actual_ground_contact_admitted": False,
        "plugin_search_path": "REQUIRES_EXPLICIT_OWNED_RUNTIME_SYSTEM_PLUGIN_PATH",
        "backend_health_admitted": False,
        "physical_acceptance": "NOT_RUN",
        "authorization": False,
    }
    (output / "probe-world-source.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
