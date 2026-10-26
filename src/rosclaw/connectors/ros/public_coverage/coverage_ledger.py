"""Persistent observed cleaning opportunities in a fixed solver-map lattice.

Unknown or temporarily disconnected regions do not erase previously discovered
work. Only explicit observed occupancy revises that floor classification.
"""

import numpy as np
from scipy.spatial import cKDTree


class CoverageLedger:
    def __init__(self, resolution=0.025, brush_radius=0.15):
        self.resolution = resolution
        self.brush_radius = brush_radius
        self.opportunities = set()
        self.cleaned = set()
        self.sample_count = 0
        self.geometry_key = None
        self.extra_centers = set()
        self.extra_geometry_key = None
        self.current_extra_centers = set()

    def _sweep(self, start, finish):
        start, finish = np.asarray(start), np.asarray(finish)
        low = np.floor((np.minimum(start, finish) - self.brush_radius) / self.resolution).astype(
            int
        )
        high = np.ceil((np.maximum(start, finish) + self.brush_radius) / self.resolution).astype(
            int
        )
        yy, xx = np.mgrid[low[1] : high[1] + 1, low[0] : high[0] + 1]
        points = (np.column_stack([xx.ravel(), yy.ravel()]) + 0.5) * self.resolution
        delta = finish - start
        length2 = delta @ delta
        fraction = (
            np.clip((points - start) @ delta / length2, 0, 1)
            if length2 > 0
            else np.zeros(len(points))
        )
        eligible = (
            np.sum((points - (start + fraction[:, None] * delta)) ** 2, axis=1)
            <= self.brush_radius**2 + 1e-14
        )
        return set(map(tuple, np.column_stack([yy.ravel(), xx.ravel()])[eligible]))

    def _integrate(self, samples):
        added = set()
        for i in range(self.sample_count, len(samples)):
            sample = samples[i]
            if sample[2]:
                added.update(self._sweep(sample[:2], sample[:2]))
            if (
                i
                and sample[2]
                and samples[i - 1][2]
                and np.linalg.norm(np.asarray(sample[:2]) - samples[i - 1][:2]) <= 0.20
            ):
                added.update(self._sweep(samples[i - 1][:2], sample[:2]))
        self.sample_count = len(samples)
        self.cleaned.update(added)
        return added

    def _classification(self, grid, keys):
        keys = np.asarray(list(keys), dtype=int).reshape(-1, 2)
        points = (keys[:, ::-1] + 0.5) * self.resolution
        cells = np.floor((points - grid.origin) / grid.resolution).astype(int)
        inside = (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < grid.data.shape[1])
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < grid.data.shape[0])
        )
        values = np.full(len(keys), -1, dtype=int)
        values[inside] = grid.data[cells[inside, 1], cells[inside, 0]]
        return keys, values

    def observe(self, grid, samples, extra_paths=(), service=None):
        added = self._integrate(samples)
        # Keep discovered obligations, but do not discover NEW floor using a
        # boundary center whose supporting public geometry has since changed.
        public_key = (
            grid.data.shape,
            tuple(grid.origin),
            grid.resolution,
            hash(grid.data.tobytes()),
            hash(grid.reachable.tobytes()),
        )
        native = []
        for space in [grid.navigation_space, grid.navigation_centers]:
            if space is not None:
                native.append(
                    (
                        space.free.shape,
                        tuple(space.origin),
                        space.resolution,
                        hash(space.free.tobytes()),
                    )
                )
        public_key += (tuple(native),)
        if public_key != self.extra_geometry_key:
            self.current_extra_centers.clear()
            self.extra_geometry_key = public_key
        for path in extra_paths:
            points = {tuple(np.round(point, 12)) for point in path}
            self.extra_centers.update(points)
            self.current_extra_centers.update(points)
        key = (public_key, len(self.current_extra_centers))
        if key != self.geometry_key:
            low = np.floor(grid.origin / self.resolution).astype(int)
            high = np.ceil(
                (grid.origin + np.array(grid.data.shape[::-1]) * grid.resolution) / self.resolution
            ).astype(int)
            yy, xx = np.mgrid[low[1] : high[1], low[0] : high[0]]
            keys = np.column_stack([yy.ravel(), xx.ravel()])
            keys, values = self._classification(grid, keys)
            keys = keys[(values >= 0) & (values <= 25)]
            centers = grid.origin + (grid.centers[:, ::-1] + 0.5) * grid.resolution
            if self.current_extra_centers:
                centers = np.vstack([centers, list(self.current_extra_centers)])
            tree = cKDTree(centers)
            for begin in range(0, len(keys), 65536):
                if service is not None:
                    service()
                part = keys[begin : begin + 65536]
                distances = tree.query((part[:, ::-1] + 0.5) * self.resolution)[0]
                self.opportunities.update(map(tuple, part[distances <= self.brush_radius + 1e-12]))
            old, values = self._classification(grid, self.opportunities)
            self.opportunities.difference_update(map(tuple, old[values >= 65]))
            self.geometry_key = key
        if added:
            keys, values = self._classification(grid, added)
            self.opportunities.update(map(tuple, keys[(values >= 0) & (values <= 25)]))

    def mask(self, grid, cleaned=False):
        yy, xx = np.mgrid[: grid.data.shape[0], : grid.data.shape[1]]
        points = grid.origin + (np.column_stack([xx.ravel(), yy.ravel()]) + 0.5) * grid.resolution
        keys = np.floor(points / self.resolution).astype(int)[:, ::-1]
        selected = self.cleaned if cleaned else self.opportunities
        return (
            np.array([tuple(key) in selected for key in keys], dtype=bool).reshape(grid.data.shape)
            & grid.free
        )

    def summary(self):
        count = len(self.opportunities)
        cleaned = len(self.opportunities & self.cleaned)
        return {
            "estimated_observed_cleanable_cells": count,
            "estimated_cleaned_cells": cleaned,
            "estimated_coverage": cleaned / max(1, count),
            "estimated_coverage_resolution_m": self.resolution,
            "estimated_coverage_frame": "map",
            "estimated_coverage_denominator": "persistent_observed_opportunities",
        }

    def path_gain(self, path):
        """Unexecuted known brush cells on a proposed polyline; no plan credit."""
        path = np.asarray(path, float)
        if path.ndim != 2 or path.shape[1] != 2 or not len(path) or not np.isfinite(path).all():
            raise ValueError("Invalid public work path")
        swept = self._sweep(path[0], path[0])
        for start, finish in zip(path, path[1:], strict=False):
            swept.update(self._sweep(start, finish))
        return (swept & self.opportunities) - self.cleaned

    def point_gains(self, points):
        """Exact remaining lattice-cell counts within each proposed brush disk.

        Public map origins need not coincide with the persistent lattice.
        Re-rasterizing missing cells before convolution can invent marginal gain.
        This query never changes obligations or credits proposed motion.
        """
        points = np.asarray(points, float).reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError("Finite brush centers required")
        missing = np.asarray(list(self.opportunities - self.cleaned), int).reshape(-1, 2)
        if not len(missing):
            return np.zeros(len(points), dtype=int)
        return cKDTree((missing[:, ::-1] + 0.5) * self.resolution).query_ball_point(
            points, float(np.sqrt(self.brush_radius**2 + 1e-14)), return_length=True
        )
