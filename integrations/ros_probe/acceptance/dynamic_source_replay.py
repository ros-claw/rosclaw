"""Recompute historical masks from retained actual component packets, read only.

This requires a closed observer audit. Matching source replay does not replace
canonical receipts, Native task state, brush proof, contacts or physical stop.
"""

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from dynamic_scenario import retained_packet

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
from rosclaw.connectors.ros.verification.mission import replay_coverage
from rosclaw.connectors.ros.verification.occupancy import coverage_grid_hash
from rosclaw.connectors.ros.verification.occupancy_geometry import OccupancyProjector


def replay_component_occupancy(path, evidence, binding):
    path = Path(path)
    if (
        binding.get("map_world_identity_approved") is not True
        or binding.get("mission_id") != evidence["mission_id"]
        or binding.get("run_id") != evidence["occupancy_binding"]["run_id"]
        or binding.get("body_snapshot_hash") != evidence.get("body_snapshot_hash")
    ):
        raise ValueError("exact approved known-fixture source and mission binding required")
    _, temporal = replay_coverage(evidence)
    if temporal is None or not temporal["complete"]:
        raise ValueError("complete temporal mission artifact required")
    summary_path = Path(str(path) + ".summary.json")
    if path.stat().st_size > 1_000_000_000 or summary_path.stat().st_size > 65536:
        raise ValueError("bounded closed observer audit required")
    summary = json.loads(summary_path.read_bytes())
    if (
        type(summary) is not dict
        or type(summary.get("events_written")) is not int
        or type(summary.get("dropped_events")) is not int
        or summary.get("complete") is not True
        or summary.get("writer_stopped") is not True
        or summary.get("dropped_events") != 0
        or summary.get("writer_error") is not None
    ):
        raise ValueError("complete closed observer audit with no dropped events required")
    requested = {}
    for pose, sample in zip(evidence["trajectory"], evidence["occupancy_samples"], strict=True):
        sequence = sample["occupancy"]["sequence"]
        if sequence in requested:
            raise ValueError("duplicate canonical occupancy sequence")
        requested[sequence] = (pose, sample)
    if not requested:
        raise ValueError("nonempty canonical temporal trajectory required")
    matched = set()
    sequence = 0
    previous = None
    packet_sequence = packet_time = None
    grid = CoverageVerifier(**evidence["grid"])
    if coverage_grid_hash(grid) != coverage_grid_hash(CoverageVerifier(**binding["grid"])):
        raise ValueError("canonical fixed grid/brush differs from admitted source binding")
    file_hash = hashlib.sha256()
    packets = []
    projector = None
    with path.open("rb") as stream:
        while True:
            line = stream.readline(2_000_001)
            if not line:
                break
            if len(line) > 2_000_000 or not line.endswith(b"\n"):
                raise ValueError("bounded complete observer audit row required")
            file_hash.update(line)
            row = json.loads(line)
            if type(row) is not dict:
                raise ValueError("closed observer audit row must be an object")
            saved = row.pop("artifact_sha256", None)
            if (
                row.get("schema_version") != "rosclaw.coverage_audit_event.v1"
                or type(row.get("sequence")) is not int
                or row["sequence"] != sequence + 1
                or row.get("previous_hash") != previous
                or digest(row) != saved
                or row.get("run_id") != binding["run_id"]
            ):
                raise ValueError("closed observer audit chain/identity mismatch")
            sequence, previous = row["sequence"], saved
            if row.get("kind") != "physics_snapshot_received":
                continue
            decoded = retained_packet(row, binding)
            packet = decoded["packet"]
            seq, stamp = packet["sequence"], packet["sim_time_sec"]
            if packet_sequence is not None and (seq != packet_sequence + 1 or stamp <= packet_time):
                raise ValueError("retained actual component source gap/time regression")
            packet_sequence, packet_time = seq, stamp
            if (
                decoded["geometry"].artifact_hash()
                != evidence["occupancy_binding"]["geometry_hash"]
            ):
                raise ValueError(
                    "actual collision geometry differs from canonical temporal binding"
                )
            if projector is None:
                projector = OccupancyProjector(grid, decoded["geometry"])
            if seq not in requested:
                continue
            pose, sample = requested[seq]
            body = decoded["body_world_pose"]
            yaw = math.atan2(
                2 * (body[6] * body[5] + body[3] * body[4]), 1 - 2 * (body[4] ** 2 + body[5] ** 2)
            )
            if (
                packet["paused"]
                or pose["time_sec"] != stamp
                or pose["x"] != body[0]
                or pose["y"] != body[1]
                or abs((pose["yaw"] - yaw + math.pi) % (2 * math.pi) - math.pi) > 1e-12
            ):
                raise ValueError("canonical pose differs from the same actual component packet")
            projected = projector.project(
                decoded["model_poses"],
                run_id=binding["run_id"],
                mission_id=binding["mission_id"],
                sequence=seq,
                frame_id=evidence["frame_id"],
                sim_time_sec=stamp,
                ground_truth_age_sec=sample["occupancy"]["ground_truth_age_sec"],
                complete=True,
            )
            payload = asdict(projected)
            payload["occupied_cells"] = list(payload["occupied_cells"])
            if (
                payload != sample["occupancy"]
                or projected.artifact_hash() != sample["occupancy_hash"]
            ):
                raise ValueError(
                    "canonical historical occupied mask differs from actual components"
                )
            matched.add(seq)
            packets.append(
                {"sequence": seq, "sim_time_sec": stamp, "packet_sha256": decoded["packet_sha256"]}
            )
    if (
        summary.get("events_written") != sequence
        or summary.get("last_event_hash") != previous
        or matched != set(requested)
    ):
        raise ValueError("final source closure or complete canonical packet correspondence missing")
    return {
        "schema_version": "rosclaw.dynamic_component_replay.v1",
        "evidence_role": "DERIVED_EXACT_SOURCE_REPLAY_NOT_NEW_PHYSICS_OR_AUTHENTICATION",
        "source_replay_match": True,
        "matched_canonical_samples": len(matched),
        "fixed_denominator_cells": temporal["fixed_denominator_cells"],
        "observer_audit_sha256": file_hash.hexdigest(),
        "observer_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "original_packets": packets,
        "ground_truth_age_note": "uses the canonical recorded observer processing age; does not manufacture a new live freshness observation",
        "physical_acceptance": "NOT_VERIFIED",
    }
