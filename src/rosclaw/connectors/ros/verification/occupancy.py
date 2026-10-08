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
