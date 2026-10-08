"""Time-paired SIM occupancy accounting; no sensor or actuator access.

The observer must pair an independent occupied-cell snapshot with each brush
pose at the same simulator timestamp. No final mask is applied to old poses.
Only sampled footprints are credited, never interpolation across unknown
obstacle movement. Runtime integration and physical acceptance are separate.
"""

import math
from dataclasses import dataclass

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.verification.coverage import CleaningPose, CoverageVerifier


def coverage_grid_hash(verifier):
    """Bind cell coordinates, frame and brush as well as the cell denominator."""
    return digest(
        {
            "width": verifier.width,
            "height": verifier.height,
            "resolution": verifier.resolution,
            "origin": verifier.origin,
            "frame_id": verifier.frame_id,
            "brush": verifier.polygon,
            "accessible": sorted(verifier.accessible),
        }
    )


@dataclass(frozen=True)
class OccupancySnapshot:
    run_id: str
    mission_id: str
    frame_id: str
    sim_time_sec: float
    sequence: int
    occupied_cells: tuple[int, ...]
    geometry_hash: str
    source: str
    complete: bool
    ground_truth_age_sec: float
    evidence_domain: str = "GAZEBO_PHYSICS"

    def artifact_hash(self):
        return digest(self.__dict__)


class OccupancyAccounting:
    """Fail-closed versioned adapter around the unchanged fixed denominator."""

    def __init__(self, verifier, *, run_id, mission_id, geometry_hash, max_age_sec=0.3):
        if not isinstance(verifier, CoverageVerifier):
            raise TypeError("coverage verifier required")
        if not all(type(s) is str and s for s in [run_id, mission_id, geometry_hash]):
            raise ValueError("occupancy requires frozen run, mission and geometry identities")
        if type(max_age_sec) not in (int, float) or not 0 < max_age_sec <= 0.3:
            raise ValueError("ground truth freshness must be finite and bounded")
        self.verifier = verifier
        self.run_id, self.mission_id, self.geometry_hash = run_id, mission_id, geometry_hash
        self.max_age_sec = max_age_sec
        self.previous_time = None
        self.previous_sequence = None
        self.snapshot_hashes = []
        self.fault = None
        self.denominator = frozenset(verifier.accessible)
        self.grid_hash = coverage_grid_hash(verifier)

    def observe(self, pose, snapshot, *, artifact_hash):
        if self.fault:
            raise ValueError(f"occupancy fault is latched: {self.fault}")
        try:
            if not isinstance(pose, CleaningPose) or not isinstance(snapshot, OccupancySnapshot):
                raise ValueError("paired CleaningPose and OccupancySnapshot required")
            if snapshot.artifact_hash() != artifact_hash:
                raise ValueError("occupancy artifact integrity mismatch")
            if (snapshot.run_id, snapshot.mission_id, snapshot.geometry_hash) != (
                self.run_id,
                self.mission_id,
                self.geometry_hash,
            ):
                raise ValueError("occupancy source binding mismatch")
            if snapshot.frame_id != self.verifier.frame_id:
                raise ValueError("occupancy frame mismatch")
            if (
                snapshot.evidence_domain != "GAZEBO_PHYSICS"
                or snapshot.source != "independent_gazebo_model_geometry"
            ):
                raise ValueError("occupancy must come from the independent SIM geometry observer")
            if snapshot.complete is not True:
                raise ValueError("occupancy observation incomplete")
            if type(snapshot.sequence) is not int or snapshot.sequence < 0:
                raise ValueError("occupancy sequence invalid")
            if (
                self.previous_sequence is not None
                and snapshot.sequence != self.previous_sequence + 1
            ):
                raise ValueError("occupancy sequence gap or reorder")
            if any(
                type(t) not in (int, float) or not math.isfinite(t)
                for t in [snapshot.sim_time_sec, pose.time_sec, snapshot.ground_truth_age_sec]
            ):
                raise ValueError("occupancy time must be finite")
            if snapshot.sim_time_sec != pose.time_sec:
                raise ValueError("occupancy and cleaning pose must have identical SIM timestamps")
            if snapshot.sim_time_sec < 0:
                raise ValueError("occupancy SIM time must be nonnegative")
            if self.previous_time is not None and snapshot.sim_time_sec <= self.previous_time:
                raise ValueError("occupancy SIM time must strictly increase")
            if not 0 <= snapshot.ground_truth_age_sec < self.max_age_sec:
                raise ValueError("occupancy ground truth stale")
            cells = snapshot.occupied_cells
            if (
                type(cells) is not tuple
                or any(type(i) is not int for i in cells)
                or len(set(cells)) != len(cells)
                or not set(cells) <= self.denominator
            ):
                raise ValueError("occupancy cells must be unique accessible integers")
            if frozenset(self.verifier.accessible) != self.denominator:
                raise ValueError("coverage denominator changed")
            if coverage_grid_hash(self.verifier) != self.grid_hash:
                raise ValueError("coverage grid or brush geometry changed")
            self.verifier.set_temporary_blocked(list(cells))
            self.verifier.observe(pose, frame_id=snapshot.frame_id, interpolate=False)
            self.previous_time, self.previous_sequence = snapshot.sim_time_sec, snapshot.sequence
            self.snapshot_hashes.append(artifact_hash)
        except (ValueError, TypeError) as exc:
            self.fault = str(exc)
            raise

    def result(self):
        return {
            "schema_version": "rosclaw.time_paired_coverage.v1",
            "evidence_role": "computed_from_time_paired_independent_observations",
            "coverage": self.verifier.result(),
            "fixed_denominator_cells": len(self.denominator),
            "complete": self.fault is None and bool(self.snapshot_hashes),
            "fault": self.fault,
            "occupancy_sample_count": len(self.snapshot_hashes),
            "occupancy_chain_hash": digest(self.snapshot_hashes),
            "interpolation": "disabled; sampled brush footprints only",
        }

    def observe_sample(self, sample):
        """Consume one trusted observer transport sample before granting credit.

        This checks bytes and frozen source identities, not transport authority.
        The daemon must configure this receiver; an action cannot enable it.
        """
        if self.fault:
            raise ValueError("occupancy fault is latched: " + self.fault)
        try:
            payload = sample["occupancy"]
            if not isinstance(payload, dict) or type(payload.get("occupied_cells")) is not list:
                raise ValueError("versioned occupancy transport payload required")
            snapshot = OccupancySnapshot(
                **{**payload, "occupied_cells": tuple(payload["occupied_cells"])}
            )
            pose = CleaningPose(
                **{k: sample[k] for k in ["x", "y", "yaw", "time_sec", "cleaning_enabled"]}
            )
            if (
                sample.get("observation_complete") is not True
                or type(sample.get("collision_count")) is not int
                or sample["collision_count"] != 0
            ):
                raise ValueError("complete collision-free observer sample required")
            self.observe(pose, snapshot, artifact_hash=sample["occupancy_hash"])
        except (KeyError, TypeError, ValueError) as exc:
            self.fault = str(exc)
            raise ValueError("invalid time-paired observer sample: " + self.fault) from exc
