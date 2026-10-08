"""Bounded conservative collision projection from a sealed Gazebo SDF.

Pure observer computation: no actuator, ROS publisher or simulator mutation.
One complete ground-truth packet must include all declared obstacle models.
Collision shapes are enclosed by model-centered discs, so projected occupied
cells can overestimate collision occupancy but cannot expose covered geometry.
Runtime source authentication and Native physical acceptance remain separate.
"""

import hashlib
import math
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
class ModelPose:
    model_name: str
    x: float
    y: float
    sim_time_sec: float


class OccupancyProjector:
    """Validate one whole packet before returning any occupancy credit input."""

    def __init__(self, verifier, geometry):
        if not isinstance(verifier, CoverageVerifier) or not isinstance(geometry, ObstacleGeometry):
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
