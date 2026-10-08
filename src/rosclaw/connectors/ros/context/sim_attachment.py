"""Explicit simulated cleaning declaration; no real cleaner inference."""

import math

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier, point_in_polygon
from rosclaw.connectors.ros.verification.reachable import cleanable_cells


def validate_sim_attachment(declaration):
    if (
        declaration.get("schema_version") != "rosclaw.sim_cleaning_attachment.v1"
        or declaration.get("evidence_domain") != "SIMULATION"
        or declaration.get("kind") != "SIMULATED_CLEANING"
        or not declaration.get("declaration_source")
    ):
        raise ValueError("explicit SIM cleaning attachment declaration required")
    polygon = declaration.get("cleaning_polygon", [])
    if not 3 <= len(polygon) <= 64:
        raise ValueError("SIM cleaning polygon must be bounded")
    verifier = CoverageVerifier(
        width=1, height=1, resolution=1, accessible_cells=[0], cleaning_polygon=polygon
    )
    polygon = [tuple(point) for point in verifier.polygon]
    if not point_in_polygon(0, 0, polygon):
        raise ValueError("offset-only cleaner needs a separate reachable-footprint derivation")
    distances = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        if not math.isfinite(length2) or length2 <= 0:
            raise ValueError("SIM cleaning edges must be finite and nondegenerate")
        t = max(0, min(1, -(a[0] * dx + a[1] * dy) / length2))
        distances.append(math.hypot(a[0] + t * dx, a[1] + t * dy))
    inscribed_radius = min(distances)
    if not math.isfinite(inscribed_radius) or inscribed_radius <= 0:
        raise ValueError("positive conservative cleaning inscribed radius required")
    payload = {
        "schema_version": "rosclaw.sim_cleaning_attachment.v1",
        "evidence_domain": "SIMULATION",
        "kind": "SIMULATED_CLEANING",
        "declaration_source": declaration["declaration_source"],
        "cleaning_polygon": [list(p) for p in polygon],
        "inscribed_radius_m": inscribed_radius,
        "usable_for_real_execution": False,
    }
    return {**payload, "attachment_hash": digest(payload)}


def derive_allowed_region_grid(
    measured_map, *, allowed_polygon, allowed_frame_id, attachment, physical_radius_m, start_pose
):
    """Explicit arbitrary simple task polygon, observed spawn and measured map.

    Occupancy outside the declared region is excluded before clearance and
    connectivity. This is a new Body/task preflight, never an adjustment to
    an existing mission denominator. Temporary obstacles must be subsequent
    time-paired masks, not modifications to this frozen initial grid.
    """
    if not 3 <= len(allowed_polygon) <= 64:
        raise ValueError("explicit bounded allowed task polygon required")
    CoverageVerifier(
        width=1, height=1, resolution=1, accessible_cells=[0], cleaning_polygon=allowed_polygon
    )
    cleaner = validate_sim_attachment(attachment)
    frame = measured_map["frame_id"]
    orientation = measured_map.get("origin_orientation")
    if (
        not isinstance(orientation, (list, tuple))
        or len(orientation) != 4
        or not all(type(v) in (int, float) and math.isfinite(v) for v in orientation)
        or any(abs(v) > 1e-12 for v in orientation[:3])
        or abs(abs(orientation[3]) - 1) > 1e-12
        or allowed_frame_id != frame
    ):
        raise ValueError("declared region frame and observed axis-aligned map origin required")
    if (
        start_pose.get("frame_id") != frame
        or start_pose.get("source") != "independent_gazebo_ground_truth_subscription"
        or start_pose.get("observation_complete") is not True
        or type(physical_radius_m) not in (int, float)
        or not math.isfinite(physical_radius_m)
        or physical_radius_m <= 0
    ):
        raise ValueError(
            "matching independent spawn frame and conservative physical radius required"
        )
    w, h, res = measured_map["width"], measured_map["height"], measured_map["resolution"]
    if (
        type(w) is not int
        or type(h) is not int
        or min(w, h) <= 0
        or w * h > 1_000_000
        or len(measured_map["occupancy"]) != w * h
        or any(type(v) is not int or not -1 <= v <= 100 for v in measured_map["occupancy"])
    ):
        raise ValueError("bounded complete integer occupancy grid required")
    ox, oy = measured_map["origin"]
    if (
        not all(
            type(v) in (int, float) and math.isfinite(v)
            for v in (ox, oy, res, start_pose["x"], start_pose["y"])
        )
        or res <= 0
    ):
        raise ValueError("finite observed map and spawn coordinates required")
    col = math.floor((start_pose["x"] - ox) / res)
    row = math.floor((start_pose["y"] - oy) / res)
    if not 0 <= col < w or not 0 <= row < h:
        raise ValueError("observed spawn lies outside the measured grid")
    occupancy = [
        value
        if point_in_polygon(ox + (i % w + 0.5) * res, oy + (i // w + 0.5) * res, allowed_polygon)
        else 100
        for i, value in enumerate(measured_map["occupancy"])
    ]
    kwargs = {
        "width": w,
        "height": h,
        "resolution": res,
        "occupancy": occupancy,
        "start_cell": row * w + col,
        "robot_radius": physical_radius_m,
    }
    denominator = cleanable_cells(**kwargs, cleaning_radius=cleaner["inscribed_radius_m"])
    centers = cleanable_cells(
        **kwargs, cleaning_radius=min(res / 100, cleaner["inscribed_radius_m"])
    )
    grid = {
        "width": w,
        "height": h,
        "resolution": res,
        "origin": [ox, oy],
        "frame_id": frame,
        "accessible_cells": denominator,
        "cleaning_polygon": cleaner["cleaning_polygon"],
    }
    return {
        "schema_version": "rosclaw.allowed_region_grid.v1",
        "evidence_role": "initial_fixed_denominator_preflight_not_physical_cleaning",
        "grid": grid,
        "grid_hash": digest(grid),
        "source_map_hash": digest(measured_map),
        "allowed_polygon_hash": digest(allowed_polygon),
        "attachment_hash": cleaner["attachment_hash"],
        "legal_center_cells": centers,
        "denominator_cells": len(denominator),
        "capabilities_granted": [],
        "usable_for_real_execution": False,
    }
