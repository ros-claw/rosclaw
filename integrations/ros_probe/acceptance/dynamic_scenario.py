"""Owned fixture obstacle perturbation; no robot command or acceptance credit.

Start before the live observer audit to consume its genesis. The Native task
runs separately through Agentd/rosclawd. All placement decisions use matched
fresh independent component packets; world service ACK is never pose proof.
"""

import argparse
import hashlib
import json
import math
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

from audit_cursor import AuditCursor
from faults import observed, service
from prepared_obstacle import PlacementClearanceUnavailableError, pose_request

from rosclaw.connectors.ros.verification.occupancy_geometry import parse_physics_packet


def retained_packet(row, binding):
    """Validate original packet bytes at original receipt; live freshness is separate."""
    payload = row["payload"]
    raw = payload["raw_packet_utf8"].encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != payload["packet_sha256"]:
        raise ValueError("retained actual packet bytes changed")
    if json.loads(raw) != payload["packet"]:
        raise ValueError("retained decoded packet differs from original bytes")
    captured = datetime.fromisoformat(row["captured_at"])
    if captured.tzinfo is None:
        raise ValueError("timezone-aware actual packet receipt required")
    decoded = parse_physics_packet(
        raw,
        **{
            k: binding[k]
            for k in (
                "run_id",
                "body_snapshot_hash",
                "attachment_hash",
                "world_name",
                "body_model_name",
            )
        },
        obstacle_names=tuple(binding["obstacle_names"]),
        scene_model_names=frozenset(binding["scene_model_names"]),
        received_at_unix_ns=int(captured.timestamp() * 1e9),
        maximum_body_planar_radius_m=binding.get("maximum_body_planar_radius_m"),
    )
    if (
        row["run_id"] != binding["run_id"]
        or row["sim_time_sec"] != decoded["packet"]["sim_time_sec"]
    ):
        raise ValueError("retained audit and original packet identity/time mismatch")
    return decoded


def scenario_policy(spec, binding):
    if (
        spec.get("schema_version") != "rosclaw.dynamic_fixture_scenario.v1"
        or spec.get("case") not in {"D1", "D2", "D3", "D4", "D6"}
        or spec.get("run_id") != binding["run_id"]
        or spec.get("mission_id") != binding["mission_id"]
        or spec.get("obstacle_name") not in binding["obstacle_names"]
    ):
        raise ValueError("preregistered scenario and exact source identity required")
    for key, low, high in (
        ("introduce_after_cleaning_sim_sec", 1, 120),
        ("dwell_sim_sec", 10, 30),
        ("wall_timeout_sec", 60, 1920),
    ):
        value = spec.get(key)
        if type(value) not in (int, float) or not low <= value <= high:
            raise ValueError("bounded frozen scenario times required")
    target = spec.get("target_xy")
    if (
        type(target) is not list
        or len(target) != 2
        or any(type(v) not in (int, float) or not -1.2 <= v <= 1.2 for v in target)
    ):
        raise ValueError("bounded fixture target pair required")
    if spec["case"] == "D3":
        if (
            spec.get("second_obstacle_name") not in binding["obstacle_names"]
            or spec["second_obstacle_name"] == spec["obstacle_name"]
        ):
            raise ValueError("two disjoint preloaded blocker identities required")
        second = spec.get("second_target_xy")
        if (
            type(second) is not list
            or len(second) != 2
            or second == target
            or any(type(v) not in (int, float) or not -1.2 <= v <= 1.2 for v in second)
        ):
            raise ValueError("bounded distinct second target required")
        for key, low, high in (("second_dwell_sim_sec", 10, 30), ("gap_sim_sec", 2, 30)):
            if type(spec.get(key)) is not int or not low <= spec[key] <= high:
                raise ValueError("frozen D3 dwell and nonconcurrent gap required")
    if spec["case"] == "D1":
        from crossing_fixture import crossing_waypoints

        points = crossing_waypoints(target, spec.get("crossing_source"))
        if spec["dwell_sim_sec"] != math.ceil(
            (len(points) - 1) * spec["crossing_source"]["interval_sim_sec"]
        ):
            raise ValueError("D1 nominal traversal duration differs from common dwell field")
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("/evidence"))
    parser.add_argument("--scenario", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    raw = args.scenario.read_bytes()
    if len(raw) > 65536:
        raise ValueError("bounded scenario file required")
    fixture = json.loads((root / "physics.json").read_text())
    binding = fixture["binding"]
    spec = scenario_policy(json.loads(raw), binding)
    name = spec["obstacle_name"]
    parked = next(o["pose"][:2] for o in fixture["obstacles"] if o["name"] == name)
    start = time.monotonic()
    until = start + spec["wall_timeout_sec"]
    cursor = None
    packets = OrderedDict()
    sequence = stamp = None
    geometry_hash = None
    first_on = occupied_at = None
    state = "WAITING_FOR_CLEANING"
    target = spec["target_xy"]
    mutation_after_ns = None
    confirmation_deadline = None
    clearance_wait_started = None
    blocking_stage, withdrawn_at = 0, None
    crossing, crossing_index, crossing_started, swath_row, main_window = None, 0, None, None, None
    if spec["case"] == "D1":
        from crossing_fixture import MainCoverageWindow, crossing_waypoints

        crossing = crossing_waypoints(spec["target_xy"], spec["crossing_source"])
        main_window = MainCoverageWindow(root, binding)
    with (root / "dynamic-scenario-events.jsonl").open("x", buffering=1) as log:

        def emit(kind, **values):
            log.write(
                json.dumps(
                    {
                        "kind": kind,
                        "captured_at": datetime.now(UTC).isoformat(),
                        "scenario_sha256": hashlib.sha256(raw).hexdigest(),
                        "run_id": binding["run_id"],
                        "mission_id": binding["mission_id"],
                        "physical_acceptance": "NOT_VERIFIED",
                        "obstacle_name": name,
                        "blocking_stage": blocking_stage,
                        **values,
                    }
                )
                + "\n"
            )

        emit("SCENARIO_STARTED", policy=spec)
        try:
            while time.monotonic() < until:
                if args.scenario.read_bytes() != raw:
                    raise ValueError("frozen scenario changed")
                if (root / "stop-dynamic-scenario.json").exists():
                    emit("TASK_RUNNER_STOP_REQUESTED", final_state=state)
                    return
                paths = list(root.glob("plan-events-*.jsonl"))
                if len(paths) > 1:
                    raise ValueError("exclusive observer audit required")
                if paths and cursor is None:
                    cursor = AuditCursor(paths[0], run_id=binding["run_id"])
                for row in cursor.poll() if cursor is not None else ():
                    if row["kind"] == "swaths":
                        swath_row = row
                    if row["kind"] != "physics_snapshot_received":
                        continue
                    decoded = retained_packet(row, binding)
                    actual_geometry = decoded["geometry"].artifact_hash()
                    if geometry_hash is None:
                        geometry_hash = actual_geometry
                    if actual_geometry != geometry_hash:
                        raise ValueError("actual closed scene collision geometry changed")
                    packet = decoded["packet"]
                    if sequence is not None and (
                        packet["sequence"] != sequence + 1 or packet["sim_time_sec"] <= stamp
                    ):
                        raise ValueError("actual component source gap/time regression")
                    sequence, stamp = packet["sequence"], packet["sim_time_sec"]
                    packets[decoded["packet_sha256"]] = decoded
                    while len(packets) > 12:
                        packets.popitem(last=False)
                if not (root / "body.json").exists() or not packets:
                    time.sleep(0.02)
                    continue
                sample = observed()
                if (
                    not 0
                    <= (
                        datetime.now(UTC) - datetime.fromisoformat(sample["captured_at"])
                    ).total_seconds()
                    < 0.3
                ):
                    raise ValueError("fresh actual paired observation required")
                decoded = packets.get(sample.get("physics_packet_sha256"))
                if decoded is None:
                    # Observer publication may precede audit writer flush.
                    time.sleep(0.02)
                    continue
                packet = decoded["packet"]
                if [sample["x"], sample["y"]] != decoded["body_world_pose"][:2]:
                    raise ValueError("paired body observation differs from actual component packet")
                if (
                    sample["time_sec"] != packet["sim_time_sec"]
                    or packet["paused"]
                    or not 0 <= time.time_ns() - packet["captured_at_unix_ns"] < 300_000_000
                ):
                    raise ValueError("fresh advancing same-time actual component packet required")
                pose = next(p for p in decoded["model_poses"] if p.model_name == name)
                if first_on is None and sample["cleaning_enabled"]:
                    first_on = sample["time_sec"]
                    emit("FIRST_ENABLED_CLEANING", sim_time_sec=first_on)
                body = json.loads((root / "body.json").read_text())
                main_context = main_window.poll() if main_window is not None else None
                if (
                    crossing is not None
                    and crossing_started is None
                    and getattr(main_window, "finished", False)
                ):
                    raise ValueError("D1 main coverage ended before crossing introduction")
                if (
                    crossing_started is not None
                    and state != "CONFIRMING_WITHDRAWAL"
                    and not (state == "OCCUPIED" and crossing_index == len(crossing) - 1)
                ):
                    if main_context is None or not sample["cleaning_enabled"]:
                        raise ValueError(
                            "D1 crossing left the actual enabled main coverage interval"
                        )
                    if (
                        sample["time_sec"] - crossing_started
                        > spec["crossing_source"]["maximum_crossing_sim_sec"]
                    ):
                        raise TimeoutError("D1 immutable crossing SIM deadline")
                if (
                    state == "WAITING_FOR_CLEANING"
                    and first_on is not None
                    and sample["time_sec"] >= first_on + spec["introduce_after_cleaning_sim_sec"]
                ) or (
                    state == "WAITING_BETWEEN_BLOCKERS"
                    and sample["time_sec"] >= withdrawn_at + spec["gap_sim_sec"]
                ):
                    if crossing is not None:
                        from crossing_fixture import crossing_intersects_swaths

                        if main_context is None or swath_row is None:
                            time.sleep(0.02)
                            continue
                        if not crossing_intersects_swaths(
                            crossing[0],
                            crossing[-1],
                            swath_row,
                            frame_id=binding["grid"]["frame_id"],
                        ):
                            raise ValueError(
                                "D1 declared traversal does not cross an actual main swath"
                            )
                    try:
                        request = pose_request(
                            fixture, body, sample, name=name, x=target[0], y=target[1]
                        )
                    except PlacementClearanceUnavailableError:
                        if clearance_wait_started is None:
                            clearance_wait_started = sample["time_sec"]
                            wait_path = root / "dynamic-placement-waits.jsonl"
                            with wait_path.open("a" if wait_path.exists() else "x") as waits:
                                waits.write(
                                    json.dumps(
                                        {
                                            "kind": "WAITING_FOR_ACTUAL_BODY_CLEARANCE",
                                            "run_id": binding["run_id"],
                                            "mission_id": binding["mission_id"],
                                            "scenario_sha256": hashlib.sha256(raw).hexdigest(),
                                            "sim_time_sec": sample["time_sec"],
                                            "original_wall_timeout_sec": spec["wall_timeout_sec"],
                                            "physical_acceptance": "NOT_VERIFIED",
                                            "blocking_stage": blocking_stage,
                                            "obstacle_name": name,
                                            "before": sample,
                                        }
                                    )
                                    + "\n"
                                )
                        time.sleep(0.02)
                        continue  # Original wall/SIM deadlines and first-ON time stay intact.
                    if clearance_wait_started is not None:
                        with (root / "dynamic-placement-waits.jsonl").open("a") as waits:
                            waits.write(
                                json.dumps(
                                    {
                                        "kind": "ACTUAL_BODY_CLEARANCE_AVAILABLE",
                                        "run_id": binding["run_id"],
                                        "mission_id": binding["mission_id"],
                                        "scenario_sha256": hashlib.sha256(raw).hexdigest(),
                                        "sim_time_sec": sample["time_sec"],
                                        "physical_acceptance": "NOT_VERIFIED",
                                        "before": sample,
                                    }
                                )
                                + "\n"
                            )
                        clearance_wait_started = None
                    mutation_after_ns = time.time_ns()
                    emit("INTRODUCTION_REQUESTED", request=request, before=sample)
                    response = service("set_pose", "gz.msgs.Pose", request)
                    emit("INTRODUCTION_ACK_REQUIRES_ACTUAL_PACKET", response=response)
                    state = "CONFIRMING_INTRODUCTION"
                    confirmation_deadline = time.monotonic() + 5
                elif state in {
                    "CONFIRMING_INTRODUCTION",
                    "CONFIRMING_WITHDRAWAL",
                    "CONFIRMING_CROSSING",
                }:
                    if (
                        packet["captured_at_unix_ns"] > mutation_after_ns
                        and math.hypot(pose.x - target[0], pose.y - target[1]) < 0.001
                    ):
                        if crossing is not None and state != "CONFIRMING_WITHDRAWAL":
                            if main_context is None or not sample["cleaning_enabled"]:
                                raise ValueError(
                                    "D1 actual crossing confirmation lacks enabled main goal"
                                )
                            if crossing_started is None:
                                crossing_started = pose.sim_time_sec
                        emit(
                            "ACTUAL_POSTUPDATE_POSITION_CONFIRMED",
                            state=state,
                            packet_sha256=decoded["packet_sha256"],
                            actual_xy=[pose.x, pose.y],
                            sim_time_sec=pose.sim_time_sec,
                            **(
                                {
                                    "crossing_index": crossing_index,
                                    "main_coverage": main_context,
                                    "swath_event_sha256": swath_row["artifact_sha256"],
                                }
                                if crossing is not None
                                else {}
                            ),
                        )
                        if state == "CONFIRMING_WITHDRAWAL":
                            if spec["case"] == "D3" and blocking_stage == 0:
                                emit(
                                    "FIRST_BLOCKER_WITHDRAWN_CONFIRMED",
                                    packet_sha256=decoded["packet_sha256"],
                                    sim_time_sec=pose.sim_time_sec,
                                )
                                withdrawn_at = pose.sim_time_sec
                                blocking_stage = 1
                                name = spec["second_obstacle_name"]
                                parked = next(
                                    o["pose"][:2] for o in fixture["obstacles"] if o["name"] == name
                                )
                                target = spec["second_target_xy"]
                                state = "WAITING_BETWEEN_BLOCKERS"
                                continue
                            emit("PERTURBATION_COMPLETE_REQUIRES_NATIVE_AND_CREDIT_VALIDATION")
                            return
                        occupied_at = pose.sim_time_sec
                        state = "OCCUPIED"
                    elif time.monotonic() > confirmation_deadline:
                        raise ValueError("service ACK lacks independent actual pose confirmation")
                elif state == "OCCUPIED" and crossing is not None:
                    from crossing_fixture import crossing_pose_request

                    if crossing_index == len(crossing) - 1:
                        target = parked
                        request = pose_request(
                            fixture, body, sample, name=name, x=target[0], y=target[1]
                        )
                        state = "CONFIRMING_WITHDRAWAL"
                        emit(
                            "CROSSING_TRAVERSAL_COMPLETE_REQUIRES_CLOSED_SOURCE",
                            sim_time_sec=sample["time_sec"],
                        )
                    elif (
                        sample["time_sec"]
                        >= occupied_at + spec["crossing_source"]["interval_sim_sec"]
                    ):
                        target = crossing[crossing_index + 1]
                        try:
                            request = crossing_pose_request(
                                fixture,
                                body,
                                sample,
                                name=name,
                                previous_xy=crossing[crossing_index],
                                target_xy=target,
                            )
                        except PlacementClearanceUnavailableError:
                            time.sleep(0.02)
                            continue
                        crossing_index += 1
                        state = "CONFIRMING_CROSSING"
                    else:
                        time.sleep(0.02)
                        continue
                    mutation_after_ns = time.time_ns()
                    emit(
                        "CROSSING_MOVE_REQUESTED",
                        request=request,
                        before=sample,
                        crossing_index=crossing_index,
                    )
                    response = service("set_pose", "gz.msgs.Pose", request)
                    emit("CROSSING_MOVE_ACK_REQUIRES_ACTUAL_PACKET", response=response)
                    confirmation_deadline = time.monotonic() + 5
                elif (
                    state == "OCCUPIED"
                    and spec["case"] in {"D2", "D3", "D6"}
                    and sample["time_sec"]
                    >= occupied_at
                    + (
                        spec["dwell_sim_sec"]
                        if blocking_stage == 0
                        else spec["second_dwell_sim_sec"]
                    )
                ):
                    target = parked
                    request = pose_request(
                        fixture, body, sample, name=name, x=target[0], y=target[1]
                    )
                    mutation_after_ns = time.time_ns()
                    emit("WITHDRAWAL_REQUESTED", request=request, before=sample)
                    response = service("set_pose", "gz.msgs.Pose", request)
                    emit("WITHDRAWAL_ACK_REQUIRES_ACTUAL_PACKET", response=response)
                    state = "CONFIRMING_WITHDRAWAL"
                    confirmation_deadline = time.monotonic() + 5
                time.sleep(0.02)
            raise TimeoutError("original bounded scenario wall deadline")
        except Exception as exc:
            emit("SCENARIO_FAILED", error=type(exc).__name__ + ": " + str(exc), state=state)
            raise


if __name__ == "__main__":
    main()
