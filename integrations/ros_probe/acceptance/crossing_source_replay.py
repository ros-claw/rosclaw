"""Closed D1 original-source correspondence, never new physics or authorization."""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from crossing_fixture import crossing_intersects_swaths, crossing_waypoints
from dynamic_scenario import retained_packet, scenario_policy
from dynamic_source_replay import replay_component_occupancy

from rosclaw.connectors.ros.diagnosis.coverage_audit import read_audit


def verify_crossing_confirmations(events, packets, swaths, action_rows, spec, fixture, evidence):
    """Match every bounded step to original component bytes and a unique main goal."""
    binding = fixture["binding"]
    scenario_policy(spec, binding)
    if spec["case"] != "D1":
        raise ValueError("explicit D1 crossing required")
    waypoints = crossing_waypoints(spec["target_xy"], spec["crossing_source"])
    if len(events) != len(waypoints) + 1:
        raise ValueError("every crossing step and final withdrawal need original confirmation")
    starts = [
        row
        for row in action_rows
        if row["kind"] == "goal_started" and row["payload"].get("stage") == "MAIN_COVERAGE"
    ]
    if len(starts) != 1:
        raise ValueError("one canonical-action-bound actual main coverage goal required")
    start = starts[0]
    action, goal = start.get("action_id"), start["payload"].get("nav_goal_id")
    if not any(
        row["kind"] == "action_admitted"
        and row["payload"].get("capability_id") == "coverage.execute"
        and row.get("action_id") == action
        for row in action_rows
    ):
        raise ValueError("actual coverage action admission required")
    if action not in evidence.get("action_ids", []) or any(
        row.get("run_id") != binding["run_id"]
        or row.get("body_snapshot_hash") != binding["body_snapshot_hash"]
        or row.get("action_id") != action
        for row in action_rows
    ):
        raise ValueError("closed main audit differs from canonical action/Body/source")
    ends = [
        row
        for row in action_rows
        if row["kind"] == "goal_ended" and row["payload"].get("nav_goal_id") == goal
    ]
    if len(ends) != 1:
        raise ValueError("closed unique main coverage interval required")
    start_wall, end_wall = [
        datetime.fromisoformat(row["captured_at"]).timestamp() for row in (start, ends[0])
    ]
    parked = next(
        row["pose"][:2] for row in fixture["obstacles"] if row["name"] == spec["obstacle_name"]
    )
    stamps, hashes, geometry = [], [], None
    for index, (event, desired) in enumerate(zip(events, [*waypoints, parked], strict=True)):
        withdrawal = index == len(waypoints)
        expected_state = (
            "CONFIRMING_WITHDRAWAL"
            if withdrawal
            else ("CONFIRMING_INTRODUCTION" if index == 0 else "CONFIRMING_CROSSING")
        )
        if (
            event.get("state") != expected_state
            or type(event.get("crossing_index")) is not int
            or event["crossing_index"] != min(index, len(waypoints) - 1)
        ):
            raise ValueError("ordered actual D1 step confirmations required")
        decoded = packets.get(event.get("packet_sha256"))
        if decoded is None:
            raise ValueError("crossing confirmation lacks original component packet")
        packet = decoded["packet"]
        actual_geometry = decoded["geometry"].artifact_hash()
        if geometry is not None and actual_geometry != geometry:
            raise ValueError("actual crossing collision geometry changed")
        geometry = actual_geometry
        stamp = packet["sim_time_sec"]
        pose = next(p for p in decoded["model_poses"] if p.model_name == spec["obstacle_name"])
        receipt_wall = datetime.fromisoformat(event["captured_at"]).timestamp()
        age = receipt_wall - packet["captured_at_unix_ns"] / 1e9
        if (
            packet["paused"]
            or not 0 <= age < 0.3
            or event.get("sim_time_sec") != stamp
            or event.get("actual_xy") != [pose.x, pose.y]
            or math.dist([pose.x, pose.y], desired) >= 0.001
        ):
            raise ValueError("fresh original crossing confirmation time/position differs")
        if not withdrawal:
            main = event.get("main_coverage", {})
            if (
                main
                != {
                    "action_id": action,
                    "nav_goal_id": goal,
                    "audit_event_sha256": start["artifact_sha256"],
                }
                or not start_wall <= packet["captured_at_unix_ns"] / 1e9 <= receipt_wall < end_wall
            ):
                raise ValueError(
                    "actual crossing must remain inside the same main coverage interval"
                )
            swath = swaths.get(event.get("swath_event_sha256"))
            if (
                swath is None
                or swath["sim_time_sec"] > stamp
                or not start_wall
                <= datetime.fromisoformat(swath["captured_at"]).timestamp()
                <= packet["captured_at_unix_ns"] / 1e9
                or not crossing_intersects_swaths(
                    waypoints[0], waypoints[-1], swath, frame_id=binding["grid"]["frame_id"]
                )
            ):
                raise ValueError("crossing lacks original transverse main swath intersection")
            if stamps and stamp - stamps[-1] < spec["crossing_source"]["interval_sim_sec"] - 1e-9:
                raise ValueError("actual crossing moved faster than frozen step interval")
            stamps.append(stamp)
        elif stamps and stamp <= stamps[-1]:
            raise ValueError("actual withdrawal must follow the completed crossing")
        hashes.append(event["packet_sha256"])
    if stamps[-1] - stamps[0] > spec["crossing_source"]["maximum_crossing_sim_sec"]:
        raise ValueError("actual crossing exceeded frozen SIM deadline")
    during = [p for p in evidence["trajectory"] if stamps[0] <= p["time_sec"] <= stamps[-1]]
    if (
        len(during) < 2
        or during[0]["time_sec"] - stamps[0] > 0.300000001
        or stamps[-1] - during[-1]["time_sec"] > 0.300000001
        or any(
            b["time_sec"] - a["time_sec"] > 0.300000001
            for a, b in zip(during, during[1:], strict=False)
        )
        or any(p["cleaning_enabled"] is not True for p in during)
    ):
        raise ValueError("actual canonical enabled cleaning samples required during crossing")
    return {
        "confirmed_waypoints": len(waypoints),
        "crossing_start_sim_sec": stamps[0],
        "crossing_end_sim_sec": stamps[-1],
        "canonical_action_id": action,
        "main_goal_id": goal,
        "original_confirmation_packet_hashes": hashes,
        "physical_acceptance": "NOT_VERIFIED",
    }


def replay_crossing_scenario(directory, evidence):
    root = Path(directory)
    fixture = json.loads((root / "physics.json").read_bytes())
    scenario_raw = (root / "scenario.json").read_bytes()
    if not 0 < len(scenario_raw) <= 65536:
        raise ValueError("bounded D1 source policy required")
    spec = json.loads(scenario_raw)
    audits = list(root.glob("plan-events-*.jsonl"))
    if len(audits) != 1:
        raise ValueError("exclusive closed original observer required")
    common = replay_component_occupancy(audits[0], evidence, fixture["binding"])
    events_raw = (root / "dynamic-scenario-events.jsonl").read_bytes()
    if len(events_raw) > 20_000_000 or not events_raw.endswith(b"\n"):
        raise ValueError("bounded complete original crossing event file required")
    rows = [json.loads(line) for line in events_raw.splitlines()]
    if (
        not rows
        or rows[-1].get("kind") != "PERTURBATION_COMPLETE_REQUIRES_NATIVE_AND_CREDIT_VALIDATION"
    ):
        raise ValueError("crossing source did not finish traversal and withdrawal")
    if any(
        row.get("run_id") != fixture["binding"]["run_id"]
        or row.get("mission_id") != spec["mission_id"]
        or row.get("scenario_sha256") != hashlib.sha256(scenario_raw).hexdigest()
        or row.get("physical_acceptance") != "NOT_VERIFIED"
        or row.get("kind") in {"SCENARIO_FAILED", "TASK_RUNNER_STOP_REQUESTED"}
        for row in rows
    ):
        raise ValueError("crossing event identity or completion differs")
    events = [row for row in rows if row.get("kind") == "ACTUAL_POSTUPDATE_POSITION_CONFIRMED"]
    wanted = {row["packet_sha256"] for row in events}
    packets, swaths, source_hash = {}, {}, hashlib.sha256()
    with audits[0].open("rb") as stream:
        for line in stream:
            source_hash.update(line)
            row = json.loads(line)
            if row["kind"] == "swaths":
                swaths[row["artifact_sha256"]] = row
            if (
                row["kind"] == "physics_snapshot_received"
                and row["payload"]["packet_sha256"] in wanted
            ):
                packet = retained_packet(row, fixture["binding"])
                packets[packet["packet_sha256"]] = packet
    if source_hash.hexdigest() != common["observer_audit_sha256"]:
        raise ValueError("original crossing source changed during closed replay")
    action_rows = []
    for path in (root / "actions").glob("coverage-audit-*.jsonl"):
        if path.stat().st_size > 32_000_000:
            raise ValueError("bounded main coverage audit required")
        summary = json.loads(Path(str(path) + ".summary.json").read_bytes())
        audit = read_audit(path)
        if (
            summary.get("complete") is not True
            or summary.get("dropped_events") != 0
            or summary.get("writer_error") is not None
            or summary.get("writer_stopped") is not True
            or summary.get("events_written") != len(audit)
            or not audit
            or summary.get("last_event_hash") != audit[-1]["artifact_sha256"]
        ):
            raise ValueError("complete closed daemon coverage audit required")
        if any(
            row["kind"] == "goal_started" and row["payload"].get("stage") == "MAIN_COVERAGE"
            for row in audit
        ):
            action_rows.extend(audit)
    result = verify_crossing_confirmations(
        events, packets, swaths, action_rows, spec, fixture, evidence
    )
    return {
        "schema_version": "rosclaw.dynamic_crossing_source_replay.v1",
        "evidence_role": "DERIVED_ORIGINAL_SOURCE_CORRESPONDENCE_NOT_NEW_PHYSICS_OR_AUTHENTICATION",
        "exact_canonical_component_replay": common,
        "crossing": result,
        "scenario_source_sha256": hashlib.sha256(scenario_raw).hexdigest(),
        "scenario_events_sha256": hashlib.sha256(events_raw).hexdigest(),
        "physical_acceptance": "NOT_VERIFIED",
    }
