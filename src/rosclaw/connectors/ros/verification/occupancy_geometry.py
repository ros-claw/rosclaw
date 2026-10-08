"""Bounded conservative collision projection from a sealed Gazebo SDF.

Pure observer computation: no actuator, ROS publisher or simulator mutation.
One complete ground-truth packet must include all declared obstacle models.
Collision shapes are enclosed by model-centered discs, so projected occupied
cells can overestimate collision occupancy but cannot expose covered geometry.
Runtime source authentication and Native physical acceptance remain separate.
"""

import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
from rosclaw.connectors.ros.verification.occupancy import OccupancySnapshot, coverage_grid_hash


def _numbers(text, count):
    values = tuple(float(s) for s in (text or "").split())
    if len(values) != count or not all(math.isfinite(v) and abs(v) <= 1_000_000 for v in values):
        raise ValueError("finite SDF geometry coordinates required")
    return values


def _offset(element):
    poses = element.findall("pose")
    if len(poses) > 1:
        raise ValueError("ambiguous collision pose")
    pose = poses[0] if poses else None
    if pose is None:
        return 0.0
    if pose.attrib:
        raise ValueError("relative_to or alternate SDF pose semantics unsupported")
    values = _numbers(pose.text, 6)
    return math.sqrt(sum(v * v for v in values[:3]))


@dataclass(frozen=True)
class ObstacleGeometry:
    sdf_sha256: str
    model_radii: tuple[tuple[str, float], ...]

    def artifact_hash(self):
        return digest(self.__dict__)


def seal_obstacle_geometry(sdf_bytes, *, obstacle_names):
    """Read collision primitives, never visual geometry or a caller radius."""
    if type(sdf_bytes) is not bytes or not 0 < len(sdf_bytes) <= 2_000_000:
        raise ValueError("bounded SDF bytes required")
    upper = sdf_bytes.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper or b"\x00" in sdf_bytes:
        raise ValueError("SDF entities or non-UTF8 XML unsupported")
    if (
        type(obstacle_names) is not tuple
        or not 0 < len(obstacle_names) <= 32
        or any(type(n) is not str or not n for n in obstacle_names)
        or len(set(obstacle_names)) != len(obstacle_names)
    ):
        raise ValueError("unique bounded obstacle model names required")
    try:
        document = ET.fromstring(sdf_bytes.decode("utf-8"))
    except (ET.ParseError, UnicodeDecodeError) as exc:
        raise ValueError("invalid SDF") from exc
    if document.tag != "sdf" or any("{" in e.tag for e in document.iter()):
        raise ValueError("plain SDF document required")
    models = list(document.findall("model"))
    for world in document.findall("world"):
        models.extend(world.findall("model"))
    radii = []
    for name in obstacle_names:
        matching = [m for m in models if m.get("name") == name]
        if len(matching) != 1:
            raise ValueError("obstacle model missing or ambiguous")
        model = matching[0]
        if model.find("include") is not None or model.find("model") is not None:
            raise ValueError("nested or included collision geometry unresolved")
        if model.find("joint") is not None:
            raise ValueError("articulated obstacle geometry unsupported")
        links = model.findall("link")
        if not 0 < len(links) <= 64:
            raise ValueError("obstacle collision links missing or unbounded")
        radius = 0.0
        count = 0
        for link in links:
            link_offset = _offset(link)
            for collision in link.findall("collision"):
                count += 1
                if count > 256:
                    raise ValueError("obstacle collision geometry unbounded")
                geometries = collision.findall("geometry")
                geometry = geometries[0] if len(geometries) == 1 else None
                if geometry is None or len(geometry) != 1:
                    raise ValueError("unique collision primitive required")
                shape = geometry[0]
                expected = {"box": ["size"], "sphere": ["radius"], "cylinder": ["radius", "length"]}
                if shape.tag in expected and sorted(e.tag for e in shape) != sorted(
                    expected[shape.tag]
                ):
                    raise ValueError("ambiguous or unresolved collision primitive")
                if shape.tag == "box":
                    size = _numbers(shape.findtext("size"), 3)
                    if min(size) <= 0:
                        raise ValueError("positive collision dimensions required")
                    extent = math.sqrt(sum((v / 2) ** 2 for v in size))
                elif shape.tag in ("sphere", "cylinder"):
                    r = _numbers(shape.findtext("radius"), 1)[0]
                    length = (
                        _numbers(shape.findtext("length"), 1)[0] if shape.tag == "cylinder" else 0
                    )
                    if r <= 0 or length < 0 or (shape.tag == "cylinder" and length == 0):
                        raise ValueError("positive collision dimensions required")
                    extent = math.hypot(r, length / 2)
                else:
                    raise ValueError("unsupported collision primitive; cannot infer free occupancy")
                radius = max(radius, link_offset + _offset(collision) + extent)
        if count == 0 or not 0 < radius <= 100:
            raise ValueError("obstacle collision geometry missing or excessive")
        radii.append((name, radius))
    return ObstacleGeometry(hashlib.sha256(sdf_bytes).hexdigest(), tuple(sorted(radii)))


@dataclass(frozen=True)
class PhysicsGeometry:
    component_geometry_hash: str
    model_radii: tuple[tuple[str, float], ...]
    source: str = "gazebo_ecm_postupdate"

    def artifact_hash(self):
        return digest(self.__dict__)


@dataclass(frozen=True)
class ModelPose:
    model_name: str
    x: float
    y: float
    sim_time_sec: float


class OccupancyProjector:
    """Validate one whole packet before returning any occupancy credit input."""

    def __init__(self, verifier, geometry):
        if not isinstance(verifier, CoverageVerifier) or not isinstance(
            geometry, (ObstacleGeometry, PhysicsGeometry)
        ):
            raise TypeError("sealed collision geometry and fixed coverage grid required")
        if verifier.width * verifier.height > 1_000_000:
            raise ValueError("occupancy projection grid unbounded")
        self.verifier, self.geometry = verifier, geometry
        self.denominator = frozenset(verifier.accessible)
        self.geometry_hash = geometry.artifact_hash()
        self.grid_hash = coverage_grid_hash(verifier)

    def project(
        self,
        poses,
        *,
        run_id,
        mission_id,
        sequence,
        frame_id,
        sim_time_sec,
        ground_truth_age_sec,
        complete,
    ):
        if complete is not True or frame_id != self.verifier.frame_id:
            raise ValueError("complete ground truth in frozen coverage frame required")
        if type(poses) is not tuple or len(poses) != len(self.geometry.model_radii):
            raise ValueError("every declared obstacle must appear in one complete pose packet")
        if type(sequence) is not int or sequence < 0:
            raise ValueError("ground truth sequence invalid")
        if any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in [sim_time_sec, ground_truth_age_sec]
        ):
            raise ValueError("finite ground truth times required")
        if sim_time_sec < 0 or not 0 <= ground_truth_age_sec < 0.3:
            raise ValueError("fresh nonnegative ground truth required")
        if not all(type(s) is str and s for s in [run_id, mission_id]):
            raise ValueError("run and mission binding required")
        by_name = {}
        for pose in poses:
            if not isinstance(pose, ModelPose) or pose.model_name in by_name:
                raise ValueError("unique obstacle model poses required")
            if any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in [pose.x, pose.y, pose.sim_time_sec]
            ):
                raise ValueError("finite obstacle pose required")
            if pose.sim_time_sec != sim_time_sec:
                raise ValueError("all obstacle poses must share the cleaning packet SIM time")
            by_name[pose.model_name] = pose
        if set(by_name) != {n for n, _ in self.geometry.model_radii}:
            raise ValueError("obstacle geometry and ground truth model set mismatch")
        if frozenset(self.verifier.accessible) != self.denominator:
            raise ValueError("fixed denominator changed")
        if coverage_grid_hash(self.verifier) != self.grid_hash:
            raise ValueError("coverage grid or brush geometry changed")
        occupied = set()
        projected_cell_budget = 100_000
        grid = self.verifier
        padding = grid.resolution / math.sqrt(2)
        for name, radius in self.geometry.model_radii:
            pose = by_name[name]
            reach = radius + padding
            min_col = max(0, math.floor((pose.x - reach - grid.origin[0]) / grid.resolution))
            max_col = min(
                grid.width - 1, math.floor((pose.x + reach - grid.origin[0]) / grid.resolution)
            )
            min_row = max(0, math.floor((pose.y - reach - grid.origin[1]) / grid.resolution))
            max_row = min(
                grid.height - 1, math.floor((pose.y + reach - grid.origin[1]) / grid.resolution)
            )
            projected_cell_budget -= max(0, max_row - min_row + 1) * max(0, max_col - min_col + 1)
            if projected_cell_budget < 0:
                raise ValueError("occupancy projection work budget exceeded; no partial snapshot")
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    cell = row * grid.width + col
                    x = grid.origin[0] + (col + 0.5) * grid.resolution
                    y = grid.origin[1] + (row + 0.5) * grid.resolution
                    if cell in self.denominator and math.hypot(x - pose.x, y - pose.y) <= reach:
                        occupied.add(cell)
        return OccupancySnapshot(
            run_id,
            mission_id,
            frame_id,
            sim_time_sec,
            sequence,
            tuple(sorted(occupied)),
            self.geometry_hash,
            "independent_gazebo_model_geometry",
            True,
            ground_truth_age_sec,
        )


def parse_physics_packet(
    raw,
    *,
    run_id,
    body_snapshot_hash,
    attachment_hash,
    world_name,
    body_model_name,
    obstacle_names,
    scene_model_names,
    received_at_unix_ns=None,
    maximum_body_planar_radius_m=None,
    required_body_reference_link=None,
):
    """Validate complete simulator-side components before deriving any geometry.

    Source authorization remains the configured passive SIM bridge and daemon;
    this parser is not a DDS authentication mechanism. No caller radius is used.
    """
    if any(
        type(v) is not str or not v
        for v in (run_id, body_snapshot_hash, attachment_hash, world_name, body_model_name)
    ):
        raise ValueError("frozen physics source identities required")
    if type(raw) is not bytes or not 0 < len(raw) <= 2_000_000:
        raise ValueError("bounded physics packet bytes required")
    try:
        packet = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid physics packet JSON") from exc
    if type(packet) is not dict or packet.get("complete") is not True:
        raise ValueError("complete physical component packet required")
    if (
        packet.get("schema_version")
        not in (
            "rosclaw.gazebo_postupdate_observation.v1",
            "rosclaw.gazebo_postupdate_observation.v2",
            "rosclaw.gazebo_postupdate_observation.v3",
        )
        or packet.get("source") != "gazebo_ecm_postupdate"
        or packet.get("evidence_domain") != "GAZEBO_PHYSICS"
        or any(
            packet.get(k) != v
            for k, v in (
                ("run_id", run_id),
                ("body_snapshot_hash", body_snapshot_hash),
                ("attachment_hash", attachment_hash),
                ("world_name", world_name),
            )
        )
    ):
        raise ValueError("physics source/run/Body/attachment/world binding mismatch")
    if (
        type(obstacle_names) is not tuple
        or not 0 < len(obstacle_names) <= 32
        or any(type(n) is not str or not n for n in obstacle_names)
        or len(set(obstacle_names)) != len(obstacle_names)
        or type(scene_model_names) is not frozenset
        or not 0 < len(scene_model_names) <= 64
        or any(type(n) is not str or not n for n in scene_model_names)
        or body_model_name in obstacle_names
        or not {body_model_name, *obstacle_names} <= scene_model_names
    ):
        raise ValueError("explicit bounded closed scene identities required")
    for key in ("sequence", "physics_iteration", "captured_at_unix_ns"):
        if type(packet.get(key)) is not int or not 0 <= packet[key] < 2**64:
            raise ValueError("bounded unsigned source sequence/iteration/capture required")
    stamp = packet.get("sim_time_sec")
    if type(stamp) not in (float, int) or not 0 <= stamp <= (2**63 - 1) / 1e9:
        raise ValueError("nonnegative finite physical SIM time required")
    if type(packet.get("paused")) is not bool:
        raise ValueError("explicit physical pause state required")
    received = time.time_ns() if received_at_unix_ns is None else received_at_unix_ns
    if type(received) is not int or not 0 <= received - packet["captured_at_unix_ns"] < 300_000_000:
        raise ValueError("physical capture stale or future-dated")
    scene = packet.get("scene_models")
    if type(scene) is not list or len(scene) != len(scene_model_names):
        raise ValueError("complete actual scene model set required")
    models, ids = {}, set()
    for row in scene:
        if (
            type(row) is not dict
            or set(row) != {"model_name", "entity_id"}
            or type(row["model_name"]) is not str
            or row["model_name"] not in scene_model_names
            or row["model_name"] in models
            or type(row["entity_id"]) is not int
            or not 0 < row["entity_id"] < 2**64
            or row["entity_id"] in ids
        ):
            raise ValueError("ambiguous actual model identity")
        models[row["model_name"]] = row["entity_id"]
        ids.add(row["entity_id"])

    def pose(values):
        if (
            type(values) is not list
            or len(values) != 7
            or any(type(v) not in (float, int) or not -1_000_000 <= v <= 1_000_000 for v in values)
            or not math.isclose(sum(v * v for v in values[3:]), 1, abs_tol=1e-6)
        ):
            raise ValueError("finite normalized actual physical pose required")
        return values

    body = packet.get("body")
    reference_geometry = packet["schema_version"] == "rosclaw.gazebo_postupdate_observation.v3"
    body_geometry = (
        reference_geometry or packet["schema_version"] == "rosclaw.gazebo_postupdate_observation.v2"
    )
    if (
        type(body) is not dict
        or set(body)
        != {"model_name", "entity_id", "world_pose"}
        | ({"collision_geometry"} if body_geometry else set())
        | ({"reference_link"} if reference_geometry else set())
        or type(body["entity_id"]) is not int
        or body["model_name"] != body_model_name
        or body["entity_id"] != models[body_model_name]
    ):
        raise ValueError("actual body model identity missing or different")
    pose(body["world_pose"])
    if required_body_reference_link is not None and (
        type(required_body_reference_link) is not str
        or not 1 <= len(required_body_reference_link) <= 256
        or not reference_geometry
    ):
        raise ValueError("actual v3 Body reference link identity required")
    if reference_geometry:
        reference = body["reference_link"]
        if (
            type(reference) is not dict
            or set(reference) != {"name", "entity_id", "model_relative_pose", "world_pose"}
            or type(reference["name"]) is not str
            or not 1 <= len(reference["name"]) <= 256
            or (
                required_body_reference_link is not None
                and reference["name"] != required_body_reference_link
            )
            or type(reference["entity_id"]) is not int
            or not 0 < reference["entity_id"] < 2**64
            or reference["entity_id"] in ids
        ):
            raise ValueError("actual named Body reference link differs or is ambiguous")
        ids.add(reference["entity_id"])
        relative = pose(reference["model_relative_pose"])
        world = pose(reference["world_pose"])
        if (
            any(abs(v) > 1e-9 for v in [*relative[:3], *relative[4:]])
            or abs(abs(relative[3]) - 1) > 1e-9
        ):
            raise ValueError("actual model and Body reference link origins differ")
        sign = (
            1
            if sum(a * b for a, b in zip(world[3:], body["world_pose"][3:], strict=True)) >= 0
            else -1
        )
        if any(
            abs(a - b) > 1e-9 for a, b in zip(world[:3], body["world_pose"][:3], strict=True)
        ) or any(
            abs(a - sign * b) > 1e-9 for a, b in zip(world[3:], body["world_pose"][3:], strict=True)
        ):
            raise ValueError("actual Body reference world pose differs from model observation")
    obstacles = packet.get("obstacles")
    if type(obstacles) is not list or len(obstacles) != len(obstacle_names):
        raise ValueError("all declared obstacle components required")
    names, radii, geometry_rows, model_poses = set(), [], [], []
    collision_count = 0
    body_planar_radius = 0.0

    def rotate(quaternion, vector):
        x, y, z, w = quaternion
        a, b, c = vector
        tx, ty, tz = 2 * (y * c - z * b), 2 * (z * a - x * c), 2 * (x * b - y * a)
        return (
            a + w * tx + y * tz - z * ty,
            b + w * ty + z * tx - x * tz,
            c + w * tz + x * ty - y * tx,
        )

    for model in ([body] if body_geometry else []) + obstacles:
        is_body = body_geometry and model is body
        if (
            type(model) is not dict
            or set(model)
            != {"model_name", "entity_id", "world_pose", "collision_geometry"}
            | ({"reference_link"} if is_body and reference_geometry else set())
            or type(model["model_name"]) is not str
            or type(model["entity_id"]) is not int
            or model["model_name"] not in ((body_model_name,) if is_body else obstacle_names)
            or model["model_name"] in names
            or model["entity_id"] != models[model["model_name"]]
        ):
            raise ValueError("actual obstacle identity missing or ambiguous")
        names.add(model["model_name"])
        position = pose(model["world_pose"])
        if not is_body:
            model_poses.append(ModelPose(model["model_name"], position[0], position[1], stamp))
        collisions = model["collision_geometry"]
        if type(collisions) is not list or not 0 < len(collisions) <= 256:
            raise ValueError("bounded actual collision components required")
        maximum = 0
        for collision in collisions:
            collision_count += 1
            if collision_count > (512 if body_geometry else 256) or type(collision) is not dict:
                raise ValueError("whole packet collision budget exceeded")
            entity = collision.get("entity_id")
            if type(entity) is not int or not 0 < entity < 2**64 or entity in ids:
                raise ValueError("duplicate or invalid actual collision identity")
            ids.add(entity)
            relative = pose(collision.get("model_relative_pose"))
            kind = collision.get("kind")
            keys = {"entity_id", "kind", "model_relative_pose", "enclosing_radius_m"}
            expected_dimensions = {
                "box": {"size"},
                "sphere": {"radius"},
                "cylinder": {"radius", "length"},
            }
            if (
                type(kind) is not str
                or kind not in expected_dimensions
                or set(collision) != keys | expected_dimensions[kind]
            ):
                raise ValueError("unique explicit actual collision primitive required")
            if kind == "box":
                sizes = collision.get("size")
                if type(sizes) is not list or len(sizes) != 3:
                    raise ValueError("actual collision box dimensions required")
            elif kind in ("sphere", "cylinder"):
                sizes = [collision.get("radius")]
                if kind == "cylinder":
                    sizes.append(collision.get("length"))
            else:
                raise ValueError("unsupported actual collision primitive")
            if any(type(v) not in (float, int) or not 0 < v <= 200 for v in sizes):
                raise ValueError("positive finite collision dimensions required")
            extent = (
                math.sqrt(sum((v / 2) ** 2 for v in sizes))
                if kind == "box"
                else sizes[0]
                if kind == "sphere"
                else math.hypot(sizes[0], sizes[1] / 2)
            )
            radius = math.sqrt(sum(v * v for v in relative[:3])) + extent
            if not math.isfinite(radius) or not 0 < radius <= 100:
                raise ValueError("actual collision envelope unbounded")
            reported = collision.get("enclosing_radius_m")
            if (
                type(reported) not in (float, int)
                or not 0 < reported <= 100
                or not math.isclose(reported, radius, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError("reported envelope does not match actual components")
            maximum = max(maximum, math.nextafter(radius, math.inf))
            if is_body:
                half = (
                    [v / 2 for v in sizes]
                    if kind == "box"
                    else (
                        [sizes[0]] * 3 if kind == "sphere" else [sizes[0], sizes[0], sizes[1] / 2]
                    )
                )
                for corner in itertools.product(*[(-v, v) for v in half]):
                    rotated = rotate(relative[3:], corner)
                    translated = [a + b for a, b in zip(rotated, relative[:3], strict=True)]
                    actual = rotate(position[3:], translated)
                    body_planar_radius = max(body_planar_radius, math.hypot(actual[0], actual[1]))
        if is_body:
            continue
        radii.append((model["model_name"], maximum))
        geometry_rows.append(
            {k: model[k] for k in ("model_name", "entity_id", "collision_geometry")}
        )
    geometry_hash = digest(
        {
            "source": "gazebo_ecm_postupdate",
            "world_name": world_name,
            "scene_models": scene,
            "body_model": body_model_name,
            "obstacles": sorted(geometry_rows, key=lambda r: r["model_name"]),
        }
    )
    geometry = PhysicsGeometry(geometry_hash, tuple(sorted(radii)))
    result = {
        "packet": packet,
        "packet_sha256": hashlib.sha256(raw).hexdigest(),
        "geometry": geometry,
        "model_poses": tuple(model_poses),
        "body_world_pose": body["world_pose"],
        "ground_truth_age_sec": (received - packet["captured_at_unix_ns"]) / 1e9,
    }
    if body_geometry:
        result["body_collision_geometry"] = body["collision_geometry"]
        result["body_planar_radius_m"] = math.nextafter(body_planar_radius, math.inf)
    if reference_geometry:
        result["body_reference_link"] = body["reference_link"]
        result["body_model_reference_identity_verified_from_components"] = True
    if maximum_body_planar_radius_m is not None:
        if (
            type(maximum_body_planar_radius_m) not in (int, float)
            or not 0 < maximum_body_planar_radius_m <= 10
            or not body_geometry
            or result["body_planar_radius_m"]
            > math.nextafter(maximum_body_planar_radius_m, math.inf)
        ):
            raise ValueError("actual v2 Body geometry exceeds or lacks its frozen planar bound")
        result["body_geometry_within_frozen_bound"] = True
    return result
