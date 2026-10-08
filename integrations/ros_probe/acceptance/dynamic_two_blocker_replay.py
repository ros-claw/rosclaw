"""Closed original D3 component/temporal correspondence, not new physics."""

import hashlib
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from dynamic_scenario import retained_packet, scenario_policy
from dynamic_source_replay import replay_component_occupancy

from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
from rosclaw.connectors.ros.verification.occupancy_geometry import OccupancyProjector


def verify_confirmations(events, packets, fixture, spec):
    binding = fixture["binding"]
    scenario_policy(spec, binding)
    if spec["case"] != "D3":
        raise ValueError("two-blocker replay requires D3")
    expected = [
        (0, "CONFIRMING_INTRODUCTION", spec["obstacle_name"]),
        (0, "CONFIRMING_WITHDRAWAL", spec["obstacle_name"]),
        (1, "CONFIRMING_INTRODUCTION", spec["second_obstacle_name"]),
        (1, "CONFIRMING_WITHDRAWAL", spec["second_obstacle_name"]),
    ]
    if len(events) != 4:
        raise ValueError("all four independent blocker confirmations required")
    grid = CoverageVerifier(**binding["grid"])
    times = []
    results = []
    geometry = None
    for event, (stage, state, name) in zip(events, expected, strict=True):
        if type(event.get("blocking_stage")) is not int or (
            event["blocking_stage"],
            event.get("state"),
            event.get("obstacle_name"),
        ) != (stage, state, name):
            raise ValueError("original nonconcurrent D3 confirmation order differs")
        sha = event.get("packet_sha256")
        decoded = packets.get(sha)
        if decoded is None:
            raise ValueError("confirmation lacks original closed component packet")
        if geometry is None:
            geometry = decoded["geometry"].artifact_hash()
        if geometry != decoded["geometry"].artifact_hash():
            raise ValueError("D3 collision geometry changed")
        packet = decoded["packet"]
        stamp = packet["sim_time_sec"]
        if event.get("sim_time_sec") != stamp or packet["paused"]:
            raise ValueError("same advancing actual component time required")
        receipt = datetime.fromisoformat(event["captured_at"])
        if receipt.tzinfo is None:
            raise ValueError("original aware confirmation receipt required")
        age = (int(receipt.timestamp() * 1e9) - packet["captured_at_unix_ns"]) / 1e9
        poses = {pose.model_name: pose for pose in decoded["model_poses"]}
        pose = poses[name]
        desired = (
            (spec["target_xy"] if stage == 0 else spec["second_target_xy"])
            if state == "CONFIRMING_INTRODUCTION"
            else next(row["pose"][:2] for row in fixture["obstacles"] if row["name"] == name)
        )
        if (
            event.get("actual_xy") != [pose.x, pose.y]
            or math.hypot(pose.x - desired[0], pose.y - desired[1]) >= 0.001
        ):
            raise ValueError("confirmation differs from original actual blocker position")
        masks = {}
        for source_name in (spec["obstacle_name"], spec["second_obstacle_name"]):
            one = replace(
                decoded["geometry"],
                model_radii=tuple(
                    (n, r) for n, r in decoded["geometry"].model_radii if n == source_name
                ),
            )
            projector = OccupancyProjector(grid, one)
            masks[source_name] = projector.project(
                (poses[source_name],),
                run_id=binding["run_id"],
                mission_id=binding["mission_id"],
                sequence=packet["sequence"],
                frame_id=grid.frame_id,
                sim_time_sec=stamp,
                ground_truth_age_sec=age,
                complete=True,
            ).occupied_cells
        other = spec["second_obstacle_name"] if stage == 0 else spec["obstacle_name"]
        if masks[other]:
            raise ValueError("D3 blockers are concurrently occupying the permitted grid")
        if bool(masks[name]) != (state == "CONFIRMING_INTRODUCTION"):
            raise ValueError(
                "D3 introduction/withdrawal lacks actual occupied/cleared grid evidence"
            )
        times.append(stamp)
        results.append(
            {
                "blocking_stage": stage,
                "state": state,
                "model_name": name,
                "sim_time_sec": stamp,
                "packet_sha256": sha,
                "occupied_cells": list(masks[name]),
                "other_model_occupied_cells": list(masks[other]),
            }
        )
    if (
        times[1] - times[0] < spec["dwell_sim_sec"]
        or times[3] - times[2] < spec["second_dwell_sim_sec"]
        or times[2] - times[1] < spec["gap_sim_sec"]
    ):
        raise ValueError("observed D3 dwell/gap is shorter than frozen original timing")
    return results


def replay_two_blocker_scenario(observer_path, events_path, scenario_path, evidence, fixture):
    common = replay_component_occupancy(observer_path, evidence, fixture["binding"])
    scenario_raw = Path(scenario_path).read_bytes()
    if not 0 < len(scenario_raw) <= 65536:
        raise ValueError("bounded frozen D3 scenario required")
    spec = json.loads(scenario_raw)
    scenario_policy(spec, fixture["binding"])
    event_hash = hashlib.sha256()
    events = []
    count = 0
    last = None
    with Path(events_path).open("rb") as source:
        for line in source:
            if len(line) > 2_000_000 or not line.endswith(b"\n"):
                raise ValueError("bounded complete original D3 event required")
            event_hash.update(line)
            count += 1
            if count > 100000:
                raise ValueError("bounded D3 event file required")
            row = json.loads(line)
            if (
                row.get("run_id") != spec["run_id"]
                or row.get("mission_id") != spec["mission_id"]
                or row.get("scenario_sha256") != hashlib.sha256(scenario_raw).hexdigest()
                or row.get("physical_acceptance") != "NOT_VERIFIED"
            ):
                raise ValueError("D3 event source identity differs")
            if row.get("kind") == "SCENARIO_FAILED":
                raise ValueError("D3 scenario retained a failure")
            if row.get("kind") == "ACTUAL_POSTUPDATE_POSITION_CONFIRMED":
                events.append(row)
            last = row
    if (
        last is None
        or last.get("kind") != "PERTURBATION_COMPLETE_REQUIRES_NATIVE_AND_CREDIT_VALIDATION"
    ):
        raise ValueError("D3 source writer did not complete both perturbations")
    wanted = {event["packet_sha256"] for event in events}
    packets = {}
    observer_hash = hashlib.sha256()
    with Path(observer_path).open("rb") as source:
        for line in source:
            if len(line) > 2_000_000 or not line.endswith(b"\n"):
                raise ValueError("bounded complete original observer row required")
            observer_hash.update(line)
            row = json.loads(line)
            if (
                row.get("kind") == "physics_snapshot_received"
                and row["payload"]["packet_sha256"] in wanted
            ):
                decoded = retained_packet(row, fixture["binding"])
                sha = decoded["packet_sha256"]
                if sha in packets:
                    raise ValueError("original confirmation component packet repeated")
                packets[sha] = decoded
    if observer_hash.hexdigest() != common["observer_audit_sha256"]:
        raise ValueError("closed original observer source changed during D3 replay")
    windows = verify_confirmations(events, packets, fixture, spec)
    return {
        "schema_version": "rosclaw.dynamic_two_blocker_source_replay.v1",
        "evidence_role": "DERIVED_ORIGINAL_SOURCE_CORRESPONDENCE_NOT_NEW_PHYSICS_OR_AUTHENTICATION",
        "exact_canonical_component_replay": common,
        "observed_nonconcurrent_windows": windows,
        "first_mask_empty_during_second_blocker": True,
        "scenario_events_sha256": event_hash.hexdigest(),
        "scenario_source_sha256": hashlib.sha256(scenario_raw).hexdigest(),
        "physical_acceptance": "NOT_VERIFIED",
    }
