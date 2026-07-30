"""Marker-free visual hand state from depth (Physical Evolution Lab §7.2, PR-PE-3).

Phase-1 honesty about what this IS and IS NOT:

* it finds the NEAREST depth cluster inside a declared ROI and reports
  its 3D centroid + a forward-most fingertip CANDIDATE — an external
  proprioception signal good enough for onset/settle/contact-distance
  evidence;
* it is NOT a skeletal hand pose estimator.  When the cluster quality
  gate fails (too few pixels, too dispersed, validity too low, ROI
  empty) the result is an honest UNKNOWN — never a guessed pose
  (v3 §7.5: Unknown/occluded must be rejected, not labeled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .calibration import CameraIntrinsics, CameraPoseContract

STATE_VERSION = "rosclaw.handcam_state.v1"


def _nearest_depth_mode(
    valid_mm: np.ndarray, *, min_pixels: int, window_mm: float, bin_mm: int = 25
) -> tuple[float | None, float | None]:
    """Nearest depth MODE via histogram — robust to hand size.

    A global quantile silently degrades when the hand occupies less than
    ``quantile`` of the ROI (test-driven fix: a 1.2%-of-frame blob fell
    below a 2% quantile, so the 'near cluster' became the whole scene).
    The nearest bin with ≥ min_pixels is the hand candidate regardless
    of its frame share; none → (None, None) = honest no-hand.
    """
    if len(valid_mm) == 0:
        return None, None
    max_depth = int(valid_mm.max()) + bin_mm
    counts, edges = np.histogram(valid_mm, bins=np.arange(0, max_depth + bin_mm, bin_mm))
    for index, count in enumerate(counts):
        if count >= min_pixels:
            return float(edges[index]), float(edges[index + 1] + window_mm)
    return None, None


@dataclass(frozen=True)
class ClusterQuality:
    pixels: int
    valid_ratio: float
    spread_m: float
    ok: bool
    reason: str


@dataclass(frozen=True)
class VisualHandState:
    state: str  # "ok" | "unknown"
    reason: str
    centroid_3d: tuple[float, float, float] | None
    fingertip_3d: tuple[float, float, float] | None
    cluster: ClusterQuality | None
    depth_valid_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": STATE_VERSION,
            "state": self.state,
            "reason": self.reason,
            "centroid_3d": self.centroid_3d,
            "fingertip_3d": self.fingertip_3d,
            "cluster": None
            if self.cluster is None
            else {
                "pixels": self.cluster.pixels,
                "valid_ratio": round(self.cluster.valid_ratio, 4),
                "spread_m": round(self.cluster.spread_m, 4),
            },
            "depth_valid_ratio": round(self.depth_valid_ratio, 4),
        }


def _quality_gate(
    points_m: np.ndarray,
    *,
    min_pixels: int,
    max_spread_m: float,
    max_extent_m: float,
    valid_ratio: float,
    min_valid_ratio: float,
) -> ClusterQuality:
    pixels = int(points_m.shape[0])
    if valid_ratio < min_valid_ratio:
        return ClusterQuality(
            pixels, valid_ratio, float("nan"), False, "depth validity too low in ROI"
        )
    if pixels < min_pixels:
        return ClusterQuality(
            pixels, valid_ratio, float("nan"), False, "cluster too small (no hand?)"
        )
    # Physical extent: a hand is ~0.1–0.2 m across; a full-ROI scene at
    # any depth is not a hand even when its std is modest (test-driven
    # fix: whole-foreground-at-300 mm passed a pure spread gate).
    extent = float(np.linalg.norm(np.ptp(points_m, axis=0)))
    if math.isfinite(extent) and extent > max_extent_m:
        return ClusterQuality(
            pixels, valid_ratio, extent, False, "cluster extent larger than a hand (scene)"
        )
    spread = float(np.linalg.norm(points_m.std(axis=0)))
    if not math.isfinite(spread) or spread > max_spread_m:
        return ClusterQuality(
            pixels, valid_ratio, spread, False, "cluster too dispersed (scene, not hand)"
        )
    return ClusterQuality(pixels, valid_ratio, spread, True, "ok")


def estimate_hand_state(
    depth_mm: np.ndarray,
    contract: CameraPoseContract,
    *,
    near_window_mm: float = 80.0,
    min_pixels: int = 300,
    max_spread_m: float = 0.25,
    max_extent_m: float = 0.30,
    min_valid_ratio: float = 0.5,
    fingertip_quantile: float = 1.0,
) -> VisualHandState:
    """Estimate one hand's visual state from a depth frame (numpy uint16 mm)."""
    intr: CameraIntrinsics = contract.intrinsics
    if depth_mm is None or depth_mm.size == 0:
        return VisualHandState("unknown", "no depth frame", None, None, None, 0.0)

    roi = contract.roi
    if roi:
        x0, y0 = roi.get("x0", 0), roi.get("y0", 0)
        x1 = roi.get("x1", depth_mm.shape[1])
        y1 = roi.get("y1", depth_mm.shape[0])
        region = depth_mm[y0:y1, x0:x1]
        offset = (x0, y0)
    else:
        region = depth_mm
        offset = (0, 0)

    valid = region[region > 0]
    valid_ratio = float(len(valid)) / float(region.size) if region.size else 0.0
    if len(valid) < min_pixels:
        return VisualHandState(
            "unknown",
            f"insufficient valid depth in ROI ({len(valid)} px)",
            None,
            None,
            ClusterQuality(len(valid), valid_ratio, float("nan"), False, "insufficient depth"),
            valid_ratio,
        )

    thresh_low, thresh_high = _nearest_depth_mode(
        valid, min_pixels=min_pixels, window_mm=near_window_mm
    )
    if thresh_low is None:
        return VisualHandState("unknown", "no near cluster", None, None, None, valid_ratio)
    mask = (region > 0) & (region >= thresh_low) & (region <= thresh_high)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return VisualHandState("unknown", "no near cluster", None, None, None, valid_ratio)

    depths_m = region[ys, xs].astype(np.float64) / 1000.0
    us = xs.astype(np.float64) + offset[0]
    vs = ys.astype(np.float64) + offset[1]
    points = np.stack(
        [
            (us - intr.ppx) * depths_m / intr.fx,
            (vs - intr.ppy) * depths_m / intr.fy,
            depths_m,
        ],
        axis=1,
    )
    quality = _quality_gate(
        points,
        min_pixels=min_pixels,
        max_spread_m=max_spread_m,
        max_extent_m=max_extent_m,
        valid_ratio=valid_ratio,
        min_valid_ratio=min_valid_ratio,
    )
    if not quality.ok:
        return VisualHandState("unknown", quality.reason, None, None, quality, valid_ratio)

    centroid = tuple(float(v) for v in points.mean(axis=0))
    # Fingertip candidate: mean of the forward-most (nearest-to-camera) points.
    tip_thresh = float(np.percentile(points[:, 2], fingertip_quantile))
    tips = points[points[:, 2] <= tip_thresh + 0.01]
    fingertip = tuple(float(v) for v in tips.mean(axis=0)) if len(tips) else centroid
    return VisualHandState("ok", "ok", centroid, fingertip, quality, valid_ratio)


@dataclass(frozen=True)
class StabilityResult:
    settled: bool
    settle_time_s: float | None
    onset_time_s: float | None
    max_post_settle_displacement_m: float
    frames: int
    unknown_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "settled": self.settled,
            "settle_time_s": self.settle_time_s,
            "onset_time_s": self.onset_time_s,
            "max_post_settle_displacement_m": round(self.max_post_settle_displacement_m, 4),
            "frames": self.frames,
            "unknown_frames": self.unknown_frames,
        }


def measure_visual_stability(
    states: list[VisualHandState],
    timestamps_s: list[float],
    *,
    settle_window_s: float = 0.5,
    settle_threshold_m: float = 0.01,
) -> StabilityResult:
    """Onset→settle over a state sequence: the first motion onset and the
    earliest time after which the centroid stays within
    ``settle_threshold_m`` for ``settle_window_s``.  Unknown frames are
    counted and break a settle window (honest UNKNOWN ≠ stable)."""
    if len(states) != len(timestamps_s):
        raise ValueError("states and timestamps must align")
    centroids = [s.centroid_3d if s.state == "ok" else None for s in states]
    known = [c for c in centroids if c is not None]
    unknown_frames = len(centroids) - len(known)
    if len(known) < 2:
        return StabilityResult(False, None, None, float("nan"), len(states), unknown_frames)

    # Onset: first frame whose centroid moved > 3x settle threshold from
    # the initial pose.
    onset_index = None
    base = known[0]
    for index, centroid in enumerate(centroids):
        if centroid is None:
            continue
        if math.dist(base, centroid) > 3 * settle_threshold_m:
            onset_index = index
            break

    # Settle: earliest index after which consecutive ok frames stay
    # within threshold for the window.
    settle_time = None
    run_start: int | None = None
    anchor: tuple[float, float, float] | None = None
    for index, centroid in enumerate(centroids):
        if centroid is None:
            run_start = None
            anchor = None
            continue
        if run_start is None:
            run_start = index
            anchor = centroid
            continue
        assert anchor is not None
        if math.dist(anchor, centroid) > settle_threshold_m:
            run_start = index
            anchor = centroid
            continue
        if timestamps_s[index] - timestamps_s[run_start] >= settle_window_s:
            settle_time = timestamps_s[run_start]
            break

    post = 0.0
    if settle_time is not None:
        settled_points = [
            c for c, t in zip(centroids, timestamps_s, strict=True) if c is not None and t >= settle_time
        ]
        if settled_points:
            anchor2 = settled_points[0]
            post = max(math.dist(anchor2, c) for c in settled_points)

    return StabilityResult(
        settled=settle_time is not None,
        settle_time_s=settle_time,
        onset_time_s=timestamps_s[onset_index] if onset_index is not None else None,
        max_post_settle_displacement_m=post,
        frames=len(states),
        unknown_frames=unknown_frames,
    )


def contact_distance_m(a: VisualHandState, b: VisualHandState) -> float | None:
    """Nearest-cluster distance between two hands' fingertip candidates;
    None (unknown) unless BOTH hands are ok — a missing hand is not zero
    distance."""
    if a.state != "ok" or b.state != "ok" or not a.fingertip_3d or not b.fingertip_3d:
        return None
    return math.dist(a.fingertip_3d, b.fingertip_3d)
