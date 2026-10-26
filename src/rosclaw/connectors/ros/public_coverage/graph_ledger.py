"""Local coverage patches registered to public SLAM graph node pairs.

map->odom drift is NOT a global map gauge change. Samples are associated with
nearby stable SLAM node IDs; later graph corrections update each patch locally.
Two node positions determine a local rigid frame, not a perfect deformation
field. These are public coverage estimates; independent physics remains Judge.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .coverage_ledger import CoverageLedger
from .online import AnchoredCoverage, Transform2D

IDENTITY = Transform2D(0, 0, 0)


class GraphAnchoredCoverage(AnchoredCoverage):
    def __init__(self, *, graph_epoch: str, **kwargs):
        super().__init__(**kwargs)
        if not graph_epoch:
            raise ValueError("Explicit public SLAM graph epoch required")
        self.graph_epoch = graph_epoch
        self.nodes = {}
        self.patches = {}
        self.orphaned = []

    def update_graph(self, nodes: dict[int, tuple[float, float]], *, graph_epoch: str):
        if graph_epoch != self.graph_epoch:
            raise ValueError("SLAM graph epoch changed; explicit registration required")
        checked = {}
        for key, point in nodes.items():
            p = np.asarray(point, float)
            if type(key) is not int or key < 0 or p.shape != (2,) or not np.isfinite(p).all():
                raise ValueError(
                    "Finite public SLAM node positions and stable integer IDs required"
                )
            checked[key] = tuple(map(float, p))
        self.nodes = checked

    def frame(self, pair):
        if any(key not in self.nodes for key in pair):
            return None
        a, b = [np.asarray(self.nodes[key]) for key in pair]
        delta = b - a
        if np.linalg.norm(delta) < 0.05:
            return None
        return Transform2D(float(a[0]), float(a[1]), math.atan2(delta[1], delta[0]))

    def pair_for(self, point):
        if len(self.nodes) < 2:
            raise ValueError("Two distinct public SLAM graph nodes required")
        ordered = sorted(self.nodes, key=lambda k: (math.dist(self.nodes[k], point), k))
        first = ordered[0]
        second = next(
            (key for key in ordered[1:] if math.dist(self.nodes[key], self.nodes[first]) >= 0.05),
            None,
        )
        if second is None:
            raise ValueError("Public SLAM graph has no distinct registration baseline")
        return tuple(sorted([first, second]))

    def patch(self, pair):
        if pair not in self.patches:
            self.patches[pair] = AnchoredCoverage(
                odom_epoch=self.epoch,
                brush_radius=self.anchor.brush_radius,
                resolution=self.anchor.resolution,
            )
        return self.patches[pair]

    def sample_map(self, *, stamp, map_xy, odom_xy, cleaner, odom_epoch, odom_fresh, cleaner_fresh):
        odom_point = np.asarray(odom_xy, float)
        map_point = np.asarray(map_xy, float)
        if (
            odom_point.shape != (2,)
            or map_point.shape != (2,)
            or not np.isfinite(odom_point).all()
            or not np.isfinite(map_point).all()
        ):
            raise ValueError("Finite public odom/map samples required")
        if odom_epoch != self.epoch:
            raise ValueError("Odom epoch changed; explicit registration required")
        if self.evidence_samples and stamp <= self.evidence_samples[-1]["stamp"]:
            raise ValueError("Out-of-order odometry cannot earn cleaning credit")
        pair = self.pair_for(map_xy)
        frame = self.frame(pair)
        local = frame.inverse([map_xy])[0]
        self.patch(pair).sample(
            stamp=stamp,
            odom_xy=local,
            cleaner=cleaner,
            odom_epoch=odom_epoch,
            odom_fresh=odom_fresh,
            cleaner_fresh=cleaner_fresh,
        )
        self.evidence_samples.append(
            {
                "stamp": stamp,
                "pair": pair,
                "local_xy": local.tolist(),
                "public_map_xy": list(map_xy),
                "public_odom_xy": list(odom_xy),
                "cleaner": cleaner,
                "odom_fresh": bool(odom_fresh),
                "cleaner_fresh": bool(cleaner_fresh),
            }
        )
        self.sample_count += 1

    def sync(self):
        self.anchor.opportunities = set()
        self.anchor.cleaned = set()
        self.retry.records = []
        self.orphaned = []
        for pair, patch in self.patches.items():
            frame = self.frame(pair)
            if frame is None:
                self.orphaned.append(pair)
                continue  # Retain local debt, award no credit without registration.
            self.anchor.opportunities.update(
                self._keys(frame.apply(self._points(patch.anchor.opportunities)))
            )
            self.anchor.cleaned.update(self._keys(frame.apply(self._points(patch.anchor.cleaned))))
            for record in patch.retry.records:
                self.retry.records.append(
                    {
                        **record,
                        "target": frame.apply([record["target"]])[0],
                        "robot": frame.apply([record["robot"]])[0],
                    }
                )
        return self

    def observe(self, grid, transform=IDENTITY, *, extra_paths=()):
        if transform != IDENTITY:
            raise ValueError(
                "Graph patches use public map poses; map->odom is not a map gauge change"
            )
        self.sync()
        view = CoverageLedger(
            resolution=self.anchor.resolution, brush_radius=self.anchor.brush_radius
        )
        view.observe(grid, [], extra_paths=extra_paths)
        new = view.opportunities - self.anchor.opportunities
        if new:
            ids = sorted(self.nodes)
            if len(ids) < 2:
                raise ValueError("Public SLAM registration unavailable")
            points = self._points(new)
            nearest = cKDTree([self.nodes[key] for key in ids]).query(points)[1]
            for index in set(map(int, nearest)):
                node = ids[index]
                pair = self.pair_for(self.nodes[node])
                frame = self.frame(pair)
                self.patch(pair).anchor.opportunities.update(
                    self._keys(frame.inverse(points[nearest == index]))
                )
        self.discovered = True
        self.sync()
        return self.view(IDENTITY)

    def failed(self, path, robot_xy, transform=IDENTITY, now=0.0):
        pair = self.pair_for(path[0])
        frame = self.frame(pair)
        self.patch(pair).retry.record(frame.inverse(path), frame.inverse([robot_xy])[0], now)

    def merge_discovered(self, planned):
        # Planner snapshots may add obligations, never synthetic executed credit.
        for pair, patch in planned.patches.items():
            self.patch(pair).anchor.opportunities.update(patch.anchor.opportunities)
        self.discovered = self.discovered or planned.discovered

    def snapshot(self):
        result = copy.copy(self)
        result.nodes = dict(self.nodes)
        result.patches = {}
        result.evidence_samples = list(self.evidence_samples)
        result.anchor = copy.copy(self.anchor)
        result.anchor.opportunities = set(self.anchor.opportunities)
        result.anchor.cleaned = set(self.anchor.cleaned)
        result.retry = copy.copy(self.retry)
        result.retry.records = copy.deepcopy(self.retry.records)
        result.orphaned = list(self.orphaned)
        for pair, patch in self.patches.items():
            item = copy.copy(patch)
            item.anchor = copy.copy(patch.anchor)
            item.anchor.opportunities = set(patch.anchor.opportunities)
            item.anchor.cleaned = set(patch.anchor.cleaned)
            item.evidence_samples = list(patch.evidence_samples)
            item.retry = copy.copy(patch.retry)
            item.retry.records = copy.deepcopy(patch.retry.records)
            result.patches[pair] = item
        return result

    def status(self, grid, transform=IDENTITY):
        self.sync()
        result = super().status(grid, IDENTITY)
        result.update(
            coordinate_anchor="public_slam_node_pairs",
            graph_epoch=self.graph_epoch,
            registration_limit="local rigid patches from node positions; deformation inside a patch remains uncertain",
            unlocalized_obligations=sum(
                len(self.patches[p].anchor.opportunities) for p in self.orphaned
            ),
            orphaned_anchor_pairs=self.orphaned,
        )
        return result

    def save(self, path: Path):
        data = {
            "schema_version": "rosclaw.graph_anchored_coverage.v1",
            "epoch": self.epoch,
            "graph_epoch": self.graph_epoch,
            "resolution": self.anchor.resolution,
            "brush_radius": self.anchor.brush_radius,
            "nodes": self.nodes,
            "samples": self.evidence_samples,
            "sample_count": self.sample_count,
            "discovered": self.discovered,
            "patches": [],
        }
        for pair, patch in self.patches.items():
            data["patches"].append(
                {
                    "pair": pair,
                    "opportunities": np.asarray(sorted(patch.anchor.opportunities), int).tolist(),
                    "cleaned": np.asarray(sorted(patch.anchor.cleaned), int).tolist(),
                    "retry": [
                        {**r, "target": r["target"].tolist(), "robot": r["robot"].tolist()}
                        for r in patch.retry.records
                    ],
                }
            )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, sort_keys=True) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, *, odom_epoch: str, graph_epoch: str):
        data = json.loads(path.read_text())
        if data["schema_version"] != "rosclaw.graph_anchored_coverage.v1" or (
            data["epoch"],
            data["graph_epoch"],
        ) != (odom_epoch, graph_epoch):
            raise ValueError("Coverage registration epoch mismatch")
        result = cls(
            odom_epoch=odom_epoch,
            graph_epoch=graph_epoch,
            brush_radius=data["brush_radius"],
            resolution=data["resolution"],
        )
        result.update_graph(
            {int(k): tuple(v) for k, v in data["nodes"].items()}, graph_epoch=graph_epoch
        )
        for raw in data["patches"]:
            patch = result.patch(tuple(raw["pair"]))
            patch.anchor.opportunities = set(map(tuple, raw["opportunities"]))
            patch.anchor.cleaned = set(map(tuple, raw["cleaned"]))
            patch.retry.records = [
                {**r, "target": np.asarray(r["target"]), "robot": np.asarray(r["robot"])}
                for r in raw["retry"]
            ]
        result.evidence_samples = data["samples"]
        result.sample_count = data["sample_count"]
        result.discovered = data["discovered"]
        return result.sync()  # Fresh live graph is still required by the observer.
