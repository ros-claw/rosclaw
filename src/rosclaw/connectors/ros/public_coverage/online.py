"""Persistent public-observation work, anchored in one continuous odom epoch.

This library proposes geometry; it never grants authority or awards Judge credit.
Only actual brush-on odometry contributes estimated cleaning. Global SE(2)
map corrections reproject the same odom evidence and outstanding obligations.
Odom resets or non-rigid SLAM deformation require a new independently registered
epoch; silently stitching those trajectories would fabricate cleaning evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coverage_ledger import CoverageLedger
from .coverage_paths import boundary_strokes
from .coverage_variants import room_sweeps
from .coverage_work import FailedWork, geometry_key, repair_options, select_phase
from .route_costs import RouteCosts


@dataclass(frozen=True)
class Transform2D:
    x: float
    y: float
    yaw: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.yaw)):
            raise ValueError("Finite public map-from-odom transform required")

    def apply(self, points):
        a = np.asarray(points, float).reshape(-1, 2)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return a @ np.array([[c, s], [-s, c]]) + [self.x, self.y]

    def inverse(self, points):
        a = np.asarray(points, float).reshape(-1, 2) - [self.x, self.y]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return a @ np.array([[c, -s], [s, c]])


class TransferUnavailableError(ValueError):
    """No finite, currently safe transfer prefix could be proposed."""


@dataclass(frozen=True)
class TransferProposal:
    """Geometry only: no admission, execution, freshness or arrival claim.

    Revalidate against current observations immediately before dispatch. The
    returned prefix retains every local/global constraint of the supplied grid.
    The full global route is deliberately not returned as an executable path.
    """

    prefix_xy: tuple[tuple[float, float], ...]
    requested_target_xy: tuple[float, float]
    selected_target_xy: tuple[float, float]
    intent_source: str
    intent_length_m: float
    target_shift_m: float


def propose_transfer(grid, robot_xy, target_xy, *, max_target_shift: float = 0.05):
    """Preserve available local detours; otherwise admit a <=0.60m prefix.

    The global proposal reuses this snapshot's exact occupancy, navigation,
    physical radius and tracking margin. It cannot clear a local unknown or
    lethal cell. Callers still own observation contracts, stopped/known-terminal
    barriers, canonical authorization, result reconciliation and finite retries.
    Work entry uses 5cm snapping by default; an explicit existing return
    contract may use up to 10cm. Tracking tolerance is independent of snapping.
    """
    from .coverage_paths import densify
    from .grid import Grid

    actual = np.asarray(robot_xy, float)
    requested = np.asarray(target_xy, float)
    if (
        actual.shape != (2,)
        or requested.shape != (2,)
        or not np.isfinite(actual).all()
        or not np.isfinite(requested).all()
        or not math.isfinite(max_target_shift)
        or not 0 <= max_target_shift <= 0.10
    ):
        raise TransferUnavailableError(
            "Finite public endpoints and original target-shift limit required"
        )
    if not grid.physical_pose_safe(actual):
        raise TransferUnavailableError("Actual public full footprint is not safe")
    target = grid.refresh_target(requested, max_shift=max_target_shift)
    route = RouteCosts(grid, actual).path(target) if target is not None else None
    source = "live_global_local_route"
    if route is None:
        occupancy, navigation, radius, margin = grid._global_intent_inputs
        intent = Grid(
            occupancy, actual, clearance=radius, tracking_margin=margin, navigation=navigation
        )
        target = intent.refresh_target(requested, max_shift=max_target_shift)
        route = RouteCosts(intent, actual).path(target) if target is not None else None
        source = "global_proposal_live_prefix_required"
    if target is None or route is None:
        raise TransferUnavailableError("No public full-footprint transfer intent")
    end = next((p for p in route[1:] if np.linalg.norm(p - actual) > 0.005), None)
    if end is None:
        raise TransferUnavailableError("No transfer translation required by this proposal")
    delta = end - actual
    distance = float(np.linalg.norm(delta))
    length = min(0.60, distance)
    while length > 0.005:
        candidate = densify(np.asarray([actual, actual + delta * (length / distance)]), step=0.025)
        if grid.motion_path_safe(candidate):
            return TransferProposal(
                prefix_xy=tuple((float(x), float(y)) for x, y in candidate),
                requested_target_xy=(float(requested[0]), float(requested[1])),
                selected_target_xy=(float(target[0]), float(target[1])),
                intent_source=source,
                intent_length_m=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
                target_shift_m=float(np.linalg.norm(target - requested)),
            )
        length -= 0.025
    raise TransferUnavailableError("No locally safe public transfer prefix")


class AnchoredCoverage:
    """Estimated cleaning and debt survive map translation/rotation and restart.

    Geometry, odom epoch ID and brush size must come from the configured Body
    and public SDK. A caller-provided trajectory is an estimate, not a verified
    physical outcome. No method returns whole-house task success.
    """

    def __init__(self, *, odom_epoch: str, brush_radius: float, resolution: float = 0.025):
        if not odom_epoch or not all(
            math.isfinite(v) and v > 0 for v in (brush_radius, resolution)
        ):
            raise ValueError("Explicit odom epoch and positive footprint required")
        self.epoch = odom_epoch
        self.anchor = CoverageLedger(resolution=resolution, brush_radius=brush_radius)
        self.last_sample: tuple[float, np.ndarray, bool] | None = None
        self.sample_count = 0
        self.evidence_samples: list[dict] = []
        self.retry = FailedWork(cooldown=30.0, maximum_attempts=3)
        self.discovered = False

    def sample(
        self,
        *,
        stamp: float,
        odom_xy,
        cleaner: bool,
        odom_epoch: str,
        odom_fresh: bool,
        cleaner_fresh: bool,
    ):
        point = np.asarray(odom_xy, float)
        if odom_epoch != self.epoch:
            raise ValueError("Odom epoch changed; explicit registration required")
        if point.shape != (2,) or not np.isfinite(point).all() or not math.isfinite(stamp):
            raise ValueError("Finite odometry required")
        if type(cleaner) is not bool:
            raise ValueError("Explicit independent cleaner state required")
        if self.last_sample is not None and stamp <= self.last_sample[0]:
            raise ValueError("Out-of-order odometry cannot earn cleaning credit")
        previous = self.last_sample
        valid = bool(odom_fresh and cleaner_fresh)
        # Stale/brush-off samples break interpolation. A subsequent valid
        # sample earns its disk only, never a bridge across the missing span.
        if valid and cleaner:
            self.anchor.cleaned.update(self.anchor._sweep(point, point))
            if previous is not None:
                old_stamp, old_point, old_cleaner = previous
                if (
                    old_cleaner
                    and stamp - old_stamp <= 1
                    and np.linalg.norm(point - old_point) <= 0.20
                ):
                    self.anchor.cleaned.update(self.anchor._sweep(old_point, point))
        self.last_sample = (stamp, point.copy(), cleaner and valid)
        self.sample_count += 1
        self.evidence_samples.append(
            {
                "stamp": stamp,
                "odom_xy": point.tolist(),
                "cleaner": cleaner,
                "odom_fresh": bool(odom_fresh),
                "cleaner_fresh": bool(cleaner_fresh),
            }
        )

    def _keys(self, points):
        return set(
            map(
                tuple,
                np.floor(np.asarray(points).reshape(-1, 2) / self.anchor.resolution).astype(int)[
                    :, ::-1
                ],
            )
        )

    def _points(self, keys):
        return (
            np.asarray(list(keys), float).reshape(-1, 2)[:, ::-1] + 0.5
        ) * self.anchor.resolution

    def observe(self, grid, transform: Transform2D, *, extra_paths=()):
        view = CoverageLedger(
            resolution=self.anchor.resolution, brush_radius=self.anchor.brush_radius
        )
        view.observe(grid, [], extra_paths=extra_paths)
        self.anchor.opportunities.update(
            self._keys(transform.inverse(self._points(view.opportunities)))
        )
        self.discovered = True
        # Never delete old debt merely because a later map is unknown,
        # occupied or disconnected. Classification is recomputed separately.
        return self.view(transform)

    def view(self, transform: Transform2D):
        view = CoverageLedger(
            resolution=self.anchor.resolution, brush_radius=self.anchor.brush_radius
        )
        view.opportunities = self._keys(transform.apply(self._points(self.anchor.opportunities)))
        view.cleaned = self._keys(transform.apply(self._points(self.anchor.cleaned)))
        return view

    def status(self, grid, transform: Transform2D):
        keys = self.anchor.opportunities
        points = transform.apply(self._points(keys))
        cells = np.floor((points - grid.origin) / grid.resolution).astype(int)
        counts = {
            "observed_unswept": 0,
            "actually_swept_estimate": 0,
            "temporarily_blocked": 0,
            "unobserved_again": 0,
            "pending_retry": 0,
        }
        for key, (x, y) in zip(keys, cells, strict=False):
            if key in self.anchor.cleaned:
                counts["actually_swept_estimate"] += 1
            elif not grid.inside((y, x)) or grid.data[y, x] < 0:
                counts["unobserved_again"] += 1
            elif grid.data[y, x] >= 65:
                counts["temporarily_blocked"] += 1
            elif any(
                np.linalg.norm(self._points([key])[0] - r["target"]) <= self.anchor.brush_radius
                for r in self.retry.records
            ):
                counts["pending_retry"] += 1
            else:
                counts["observed_unswept"] += 1
        return {
            "schema_version": "rosclaw.public_coverage_status.v1",
            "coordinate_anchor": "odom",
            "odom_epoch": self.epoch,
            "states": counts,
            "discovered_obligations": len(keys),
            "samples": self.sample_count,
            "unobserved_space_extent": "UNKNOWN",
            "whole_house_success": None,
            "independent_judge_score": None,
            "scope": "public estimated brush ledger; not independent physical verification",
            "registration_limit": "one continuous odom epoch; global rigid map corrections only",
        }

    def failed(self, path, robot_xy, transform: Transform2D, now: float):
        self.retry.record(transform.inverse(path), transform.inverse([robot_xy])[0], now)

    def next_work(
        self,
        grid,
        robot_xy,
        transform: Transform2D,
        *,
        now: float,
        maximum_length: float = 3.0,
        minimum_new_area_m2: float = 0.0,
        prepare_path=None,
        service=None,
    ):
        if not math.isfinite(maximum_length) or not 0 < maximum_length <= 3:
            raise ValueError("Bounded work length must be in (0, 3] metres")
        if not math.isfinite(minimum_new_area_m2) or minimum_new_area_m2 < 0:
            raise ValueError("Minimum estimated new brush area must be finite and nonnegative")
        minimum_cells = max(1, math.ceil(minimum_new_area_m2 / self.anchor.resolution**2))
        boundary = boundary_strokes(grid, service=service)
        view = self.observe(grid, transform, extra_paths=boundary)
        costs = RouteCosts(grid, robot_xy, service=service)
        phases = {
            "interior": room_sweeps(
                grid, robot_xy, spacing=min(0.25, self.anchor.brush_radius * 1.7)
            ),
            "boundary": boundary,
            "repair": repair_options(
                grid, view, robot_xy, transit_distances=costs.distance[tuple(grid.centers.T)]
            ),
        }
        owner = self

        class RetryView:
            def blocked(self, path, robot_xy, now):
                return owner.retry.blocked(
                    transform.inverse(path), transform.inverse([robot_xy])[0], now
                )

        selected = select_phase(
            phases,
            grid,
            view,
            robot_xy,
            {},
            maximum=maximum_length,
            minimum_cells=minimum_cells,
            retries=RetryView(),
            now=now,
            transit_cost=costs,
            bidirectional=True,
            prepare_path=prepare_path,
            service=service,
        )
        if selected is not None:
            phase, (path, key, version, gain) = selected
            return {
                "kind": phase,
                "path_xy": path.tolist(),
                "geometry_sha256": version,
                "expected_new_cells": gain,
                "execution_credit": False,
                "authority_required": "canonical Runtime or shared public SDK",
            }
        # Exhaustion of observed work is not proof that the house is discovered.
        targets = grid.frontier_targets(robot_xy)
        for target in targets:
            if not self.retry.blocked(
                transform.inverse([target]), transform.inverse([robot_xy])[0], now
            ):
                return {
                    "kind": "explore",
                    "path_xy": [target.tolist()],
                    "geometry_sha256": geometry_key(grid),
                    "execution_credit": False,
                }
        return {
            "kind": "NO_CURRENT_OBSERVED_WORK",
            "whole_house_success": None,
            "unobserved_space_extent": "UNKNOWN",
            "unfinished_obligations_retained": True,
            "deferred_work_minimum_new_area_m2": minimum_new_area_m2,
        }

    def save(self, path: Path):
        data = {
            "schema_version": "rosclaw.anchored_coverage.v1",
            "epoch": self.epoch,
            "resolution": self.anchor.resolution,
            "brush_radius": self.anchor.brush_radius,
            "opportunities": np.asarray(sorted(self.anchor.opportunities), int).tolist(),
            "cleaned": np.asarray(sorted(self.anchor.cleaned), int).tolist(),
            "evidence_samples": self.evidence_samples,
            "sample_count": self.sample_count,
            "discovered": self.discovered,
            "retry": [
                {**r, "target": r["target"].tolist(), "robot": r["robot"].tolist()}
                for r in self.retry.records
            ],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, sort_keys=True) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, *, odom_epoch: str):
        data = json.loads(path.read_text())
        if data["schema_version"] != "rosclaw.anchored_coverage.v1" or data["epoch"] != odom_epoch:
            raise ValueError("Unsupported ledger or odom epoch mismatch")
        ledger = cls(
            odom_epoch=odom_epoch, resolution=data["resolution"], brush_radius=data["brush_radius"]
        )
        ledger.anchor.opportunities = set(map(tuple, data["opportunities"]))
        ledger.anchor.cleaned = set(map(tuple, data["cleaned"]))
        ledger.sample_count = data["sample_count"]
        ledger.evidence_samples = data["evidence_samples"]
        ledger.discovered = data["discovered"]
        ledger.retry.records = [
            {**r, "target": np.asarray(r["target"]), "robot": np.asarray(r["robot"])}
            for r in data["retry"]
        ]
        # No interpolation across a process restart with unobserved brush state.
        return ledger
