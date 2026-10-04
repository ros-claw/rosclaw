"""Rasterize the cleaning polygon, preserving temporarily blocked denominator.

Cell centers determine coverage. Trace gaps and disabled cleaning never sweep.
This computes coverage evidence, not an execution or mission success receipt.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if (
            abs(cross) < 1e-10
            and min(ax, bx) - 1e-10 <= x <= max(ax, bx) + 1e-10
            and min(ay, by) - 1e-10 <= y <= max(ay, by) + 1e-10
        ):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


@dataclass(frozen=True)
class CleaningPose:
    x: float
    y: float
    yaw: float
    time_sec: float
    cleaning_enabled: bool


class CoverageVerifier:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        resolution: float,
        accessible_cells: list[int],
        cleaning_polygon: list[tuple[float, float]],
        origin: tuple[float, float] = (0, 0),
        frame_id: str = "map",
        max_trace_gap_sec: float = 1.0,
        max_speed_mps: float = 2.0,
    ):
        numeric = [
            resolution,
            *origin,
            max_trace_gap_sec,
            max_speed_mps,
            *(coordinate for p in cleaning_polygon for coordinate in p),
        ]
        if not all(math.isfinite(n) for n in numeric):
            raise ValueError("coverage geometry must be finite")
        if width <= 0 or height <= 0 or resolution <= 0 or len(cleaning_polygon) < 3:
            raise ValueError("positive grid dimensions and a cleaning polygon are required")
        if max_trace_gap_sec <= 0 or max_speed_mps <= 0:
            raise ValueError("trace limits must be positive")
        self.width, self.height, self.resolution = width, height, resolution
        self.origin, self.frame_id = origin, frame_id
        self.accessible = set(accessible_cells)
        if not self.accessible or any(i < 0 or i >= width * height for i in self.accessible):
            raise ValueError("accessible cells must be nonempty and inside the grid")
        area = (
            abs(
                sum(
                    a[0] * b[1] - b[0] * a[1]
                    for a, b in zip(
                        cleaning_polygon, cleaning_polygon[1:] + cleaning_polygon[:1], strict=True
                    )
                )
            )
            / 2
        )
        if area <= 0:
            raise ValueError("cleaning polygon must have nonzero area")
        self.polygon = cleaning_polygon
        self.max_gap, self.max_speed = max_trace_gap_sec, max_speed_mps
        self.radius = max(math.hypot(x, y) for x, y in cleaning_polygon)
        self.visits: dict[int, int] = {}
        self.last_footprint: set[int] = set()
        self.temporary_blocked: set[int] = set()
        self.previous: CleaningPose | None = None
        self.gaps = 0

    def set_temporary_blocked(self, cells: list[int]) -> None:
        if not set(cells) <= self.accessible:
            raise ValueError("temporary blocks must be accessible cells")
        self.temporary_blocked = set(cells)

    def observe(self, pose: CleaningPose, *, frame_id: str) -> None:
        if frame_id != self.frame_id:
            raise ValueError("trajectory and grid frame mismatch")
        if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw, pose.time_sec)):
            raise ValueError("trajectory values must be finite")
        previous = self.previous
        if previous is not None and pose.time_sec <= previous.time_sec:
            raise ValueError("trajectory timestamps must strictly increase")
        self.previous = pose
        if not pose.cleaning_enabled:
            self.last_footprint.clear()
            return
        positions = [(pose.x, pose.y, pose.yaw)]
        if previous and previous.cleaning_enabled:
            distance = math.hypot(pose.x - previous.x, pose.y - previous.y)
            dt = pose.time_sec - previous.time_sec
            yaw_delta = (pose.yaw - previous.yaw + math.pi) % (2 * math.pi) - math.pi
            if dt <= self.max_gap and distance / dt <= self.max_speed:
                steps = max(
                    1, math.ceil((distance + abs(yaw_delta) * self.radius) / (self.resolution / 4))
                )
                positions = [
                    (
                        previous.x + (pose.x - previous.x) * i / steps,
                        previous.y + (pose.y - previous.y) * i / steps,
                        previous.yaw + yaw_delta * i / steps,
                    )
                    for i in range(steps + 1)
                ]
            else:
                self.gaps += 1
                self.last_footprint.clear()
        swept = set()
        for x, y, yaw in positions:
            footprint_cells: set[int] = set()
            cosine, sine = math.cos(yaw), math.sin(yaw)
            polygon = [
                (x + px * cosine - py * sine, y + px * sine + py * cosine)
                for px, py in self.polygon
            ]
            min_col = max(
                0, math.floor((min(p[0] for p in polygon) - self.origin[0]) / self.resolution)
            )
            max_col = min(
                self.width - 1,
                math.floor((max(p[0] for p in polygon) - self.origin[0]) / self.resolution),
            )
            min_row = max(
                0, math.floor((min(p[1] for p in polygon) - self.origin[1]) / self.resolution)
            )
            max_row = min(
                self.height - 1,
                math.floor((max(p[1] for p in polygon) - self.origin[1]) / self.resolution),
            )
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    cell = row * self.width + col
                    if (
                        cell in self.accessible
                        and cell not in self.temporary_blocked
                        and point_in_polygon(
                            self.origin[0] + (col + 0.5) * self.resolution,
                            self.origin[1] + (row + 0.5) * self.resolution,
                            polygon,
                        )
                    ):
                        footprint_cells.add(cell)
            swept.update(footprint_cells)
        for cell in swept - self.last_footprint:
            self.visits[cell] = self.visits.get(cell, 0) + 1
        self.last_footprint = footprint_cells

    def missed_regions(self, *, min_cells: int = 1) -> list[dict]:
        remaining = self.accessible - set(self.visits)
        regions = []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            queue, component = deque([start]), {start}
            while queue:
                cell = queue.popleft()
                row, col = divmod(cell, self.width)
                for nr, nc in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
                    candidate = nr * self.width + nc
                    if 0 <= nr < self.height and 0 <= nc < self.width and candidate in remaining:
                        remaining.remove(candidate)
                        component.add(candidate)
                        queue.append(candidate)
            if len(component) >= min_cells:
                regions.append(
                    {
                        "region_id": f"missed_{start}",
                        "cells": sorted(component),
                        "area_m2": len(component) * self.resolution**2,
                        "centroid": [
                            self.origin[0]
                            + sum(i % self.width + 0.5 for i in component)
                            / len(component)
                            * self.resolution,
                            self.origin[1]
                            + sum(i // self.width + 0.5 for i in component)
                            / len(component)
                            * self.resolution,
                        ],
                        "reason": "TEMP_BLOCKED"
                        if component & self.temporary_blocked
                        else "NOT_CLEANED",
                        "reachable": True,
                        "retry_count": 0,
                        "deferred": bool(component & self.temporary_blocked),
                    }
                )
        return regions

    def result(self) -> dict:
        total, covered = len(self.accessible), len(self.visits)
        return {
            "schema_version": "rosclaw.coverage_mask.v1",
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin": list(self.origin),
            "accessible_area_m2": total * self.resolution**2,
            "covered_area_m2": covered * self.resolution**2,
            "missed_area_m2": (total - covered) * self.resolution**2,
            "coverage_ratio": covered / total,
            "overlap_ratio": sum(v > 1 for v in self.visits.values()) / total,
            "trace_gaps": self.gaps,
            "missed_regions": self.missed_regions(),
            "mask": [
                1
                if i in self.visits
                else 4
                if i in self.temporary_blocked
                else 0
                if i in self.accessible
                else 2
                for i in range(self.width * self.height)
            ],
            "evidence_class": "computed_from_supplied_trace",
            "mission_success": None,
        }
