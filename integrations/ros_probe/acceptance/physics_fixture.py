"""Prepare an explicit known SIM scene; preparation is not physical evidence.

No ROS/Gazebo imports or actuator calls. Only the fixture launcher invokes this
before its owned simulator starts. Actual geometry is admitted independently
from PostUpdate component packets, never from the generated SDF below.
"""

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
from rosclaw.connectors.ros.verification.occupancy_geometry import seal_obstacle_geometry


def prepare_physics(output, *, config_path, library_path, brush_binding, profile):
    """Validate everything before writing an exclusive frozen scene admission."""
    plugin_path = output / "librosclaw_passive_physics.so"
    destinations = (plugin_path, output / "physics_binding.json", output / "physics_fixture.json")
    if any(p.exists() for p in destinations):
        raise ValueError("physics fixture preparation requires a fresh exclusive output")
    config_bytes = Path(config_path).read_bytes()
    if len(config_bytes) > 2_000_000:
        raise ValueError("physics fixture configuration exceeds bound")
    config = json.loads(config_bytes)
    if (
        type(config) is not dict
        or set(config) != {"schema_version", "binding", "obstacles", "plugin_sha256"}
        or config["schema_version"] != "rosclaw.sim_physics_fixture.v1"
    ):
        raise ValueError("explicit versioned physics fixture configuration required")
    binding = config["binding"]
    keys = {
        "run_id",
        "body_snapshot_hash",
        "attachment_hash",
        "mission_id",
        "grid",
        "world_name",
        "body_model_name",
        "obstacle_names",
        "scene_model_names",
        "world_to_map_xyyaw",
        "frame_transform_source",
        "map_world_identity_approved",
    }
    if type(binding) is not dict or set(binding) != keys:
        raise ValueError("closed frozen physics binding required")
    if any(
        binding[k] != brush_binding[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")
    ) or any(
        type(binding[k]) is not str or not 0 < len(binding[k]) <= 256
        for k in ("run_id", "body_snapshot_hash", "attachment_hash", "mission_id")
    ):
        raise ValueError("physics and brush source identities differ")
    if (
        binding["world_name"] != "ros_expert"
        or binding["body_model_name"] != profile.simulation_model
        or binding["world_to_map_xyyaw"] != [0, 0, 0]
        or binding["frame_transform_source"] != "simulator_operator_fixture_policy"
        or binding["map_world_identity_approved"] is not True
    ):
        raise ValueError("explicit known-fixture world/map identity required")
    if (
        type(binding["grid"]) is not dict
        or set(binding["grid"])
        != {
            "width",
            "height",
            "resolution",
            "origin",
            "frame_id",
            "accessible_cells",
            "cleaning_polygon",
        }
        or type(binding["grid"]["accessible_cells"]) is not list
        or len(binding["grid"]["accessible_cells"]) > 4096
    ):
        raise ValueError("bounded frozen fixture grid required")
    grid = CoverageVerifier(**binding["grid"])
    if (grid.width, grid.height, grid.resolution, tuple(grid.origin), grid.frame_id) != (
        64,
        64,
        0.05,
        (-1.6, -1.6),
        "map",
    ) or [list(p) for p in grid.polygon] != profile.cleaning_polygon:
        raise ValueError("physics grid or brush differs from the known fixture")
    # The denominator is frozen before obstacle introduction, never trimmed by
    # obstacle occupancy. Require the same original map/Body computation.
    from rosclaw.connectors.ros.verification.reachable import cleanable_cells

    occupancy = [
        100 if abs((x + 0.5) * 0.05 - 1.6) >= 1.5 or abs((y + 0.5) * 0.05 - 1.6) >= 1.5 else 0
        for y in range(64)
        for x in range(64)
    ]
    expected = cleanable_cells(
        width=64,
        height=64,
        resolution=0.05,
        occupancy=occupancy,
        start_cell=32 * 64 + 32,
        robot_radius=profile.physical_radius_m,
        cleaning_radius=profile.cleaner_half_width_m,
    )
    if grid.accessible != set(expected):
        raise ValueError("physics denominator must equal the original static fixture")
    obstacles = config["obstacles"]
    if type(obstacles) is not list or not 1 <= len(obstacles) <= 32:
        raise ValueError("one to 32 explicit obstacle primitives required")
    world_tree = ET.parse(output / "world.sdf")
    world = world_tree.getroot().find("world")
    static_names = {m.get("name") for m in world.findall("model")}
    if static_names != {"floor", "east", "west", "north", "south"}:
        raise ValueError("unknown fixture world model set")
    names = []
    for obstacle in obstacles:
        if type(obstacle) is not dict or set(obstacle) != {"name", "pose", "box_size"}:
            raise ValueError("explicit box obstacle definition required")
        name, pose, size = obstacle["name"], obstacle["pose"], obstacle["box_size"]
        if (
            type(name) is not str
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name)
            or name in static_names | {profile.simulation_model}
            or name in names
        ):
            raise ValueError("unique bounded obstacle model identity required")
        if (
            type(pose) is not list
            or len(pose) != 6
            or type(size) is not list
            or len(size) != 3
            or any(type(v) not in (int, float) or not -100 <= v <= 100 for v in pose)
            or any(type(v) not in (int, float) or not 0 < v <= 10 for v in size)
        ):
            raise ValueError("bounded finite obstacle pose and dimensions required")
        # New obstacles must start outside the room; later introduction is a
        # separate fixture-owned scenario event, never a pre-mission blockage.
        radius = math.sqrt(sum((v / 2) ** 2 for v in size))
        if math.hypot(pose[0], pose[1]) - radius <= math.sqrt(2) * 1.6:
            raise ValueError("obstacle must initially be parked outside the fixed room")
        names.append(name)
        model = ET.SubElement(world, "model", name=name)
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = " ".join(map(str, pose))
        link = ET.SubElement(model, "link", name="obstacle")
        for tag in ("collision", "visual"):
            shape = ET.SubElement(link, tag, name="box")
            geometry = ET.SubElement(shape, "geometry")
            box = ET.SubElement(geometry, "box")
            ET.SubElement(box, "size").text = " ".join(map(str, size))
    if (
        binding["obstacle_names"] != names
        or type(binding["scene_model_names"]) is not list
        or len(binding["scene_model_names"]) != len(static_names) + len(names) + 1
        or set(binding["scene_model_names"])
        != static_names | set(names) | {profile.simulation_model}
    ):
        raise ValueError("closed scene identities differ from generated fixture")
    seal_obstacle_geometry(ET.tostring(world_tree.getroot()), obstacle_names=tuple(names))
    if not 0 < Path(library_path).stat().st_size <= 20_000_000:
        raise ValueError("bounded compiled passive plugin required")
    library = Path(library_path).read_bytes()
    sha = hashlib.sha256(library).hexdigest()
    if (
        not library.startswith(b"\x7fELF")
        or len(library) > 20_000_000
        or sha != config["plugin_sha256"]
    ):
        raise ValueError("pinned compiled passive plugin integrity mismatch")
    plugin = ET.SubElement(
        world, "plugin", filename=str(plugin_path), name="rosclaw::PassivePhysics"
    )
    for key in ("run_id", "body_snapshot_hash", "attachment_hash", "body_model_name"):
        ET.SubElement(plugin, key).text = binding[key]
    for name in sorted(static_names):
        ET.SubElement(plugin, "static_model").text = name
    for name in names:
        ET.SubElement(plugin, "obstacle_model").text = name
    bridge = yaml.safe_load((output / "bridge.yaml").read_text())
    bridge.append(
        {
            "ros_topic_name": "/rosclaw_sim/physics_snapshot",
            "gz_topic_name": "/rosclaw_sim/physics_snapshot",
            "ros_type_name": "std_msgs/msg/String",
            "gz_type_name": "gz.msgs.StringMsg",
            "direction": "GZ_TO_ROS",
            "publisher_queue": 6,
            "subscriber_queue": 6,
        }
    )
    with plugin_path.open("xb") as destination:
        destination.write(library)
    for path, value in zip(destinations[1:], (binding, config), strict=True):
        with path.open("x") as destination:
            destination.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
    world_tree.write(output / "world.sdf")
    (output / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    return {"plugin_sha256": sha, "physical_acceptance": "NOT_RUN", "binding": binding}


def admit_dynamic_daemon_config(output, config, *, mission_id, profile):
    """Bind compiled Body/actual map to the prepared observer's source seal.

    This does not make stale readiness live: rosclawd separately waits for a
    fresh complete observation, and accounts every paired packet on dispatch.
    The readiness file is produced by the fixture-owned passive observer, not
    by an action argument. Same-UID fixtures prove no DDS authentication.
    """
    binding = json.loads((output / "physics_binding.json").read_text())
    ready = json.loads((output / "physics_ready.json").read_text())
    fixture = json.loads((output / "physics_fixture.json").read_text())
    brush = json.loads((output / "brush_binding.json").read_text())
    if (
        type(brush) is not dict
        or set(brush) != {"run_id", "body_snapshot_hash", "attachment_hash", "producer_id"}
        or any(type(v) is not str or not v for v in brush.values())
        or type(binding) is not dict
        or type(ready) is not dict
        or type(fixture) is not dict
    ):
        raise ValueError("complete prepared dynamic source files required")
    if (
        ready.get("binding") != binding
        or fixture.get("binding") != binding
        or ready.get("evidence_role") != "actual_component_source_admission_not_mission_acceptance"
        or binding.get("body_snapshot_hash") != config["body_snapshot_hash"]
        or binding.get("body_model_name") != profile.simulation_model
        or binding.get("mission_id") != mission_id
        or binding.get("grid") != config["grid"]
        or binding.get("run_id") != (output / "run_id.txt").read_text().strip()
        or any(
            binding.get(k) != brush[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")
        )
    ):
        raise ValueError("compiled Body/map/mission differs from frozen dynamic source")
    if any(
        type(ready.get(k)) is not str or not re.fullmatch(r"[0-9a-f]{64}", ready[k])
        for k in ("geometry_hash", "initial_packet_sha256")
    ):
        raise ValueError("actual component geometry and initial packet hashes required")
    if hashlib.sha256(
        (output / "librosclaw_passive_physics.so").read_bytes()
    ).hexdigest() != fixture.get("plugin_sha256"):
        raise ValueError("prepared passive plugin bytes changed")
    return {
        **config,
        "brush_binding": brush,
        "physical_radius_m": profile.physical_radius_m,
        "occupancy_binding": {"run_id": binding["run_id"], "geometry_hash": ready["geometry_hash"]},
        "dynamic_fixture_admission": {
            "mission_id": mission_id,
            "initial_packet_sha256": ready["initial_packet_sha256"],
            "source_role": ready["evidence_role"],
            "physical_acceptance": "NOT_RUN",
        },
    }
