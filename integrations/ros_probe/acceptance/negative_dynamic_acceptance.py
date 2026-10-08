"""D4 closed-source SIM negative acceptance; no task promotion or ROS access."""

import hashlib
import json
import math
from pathlib import Path

from dynamic_scenario import retained_packet, scenario_policy
from dynamic_source_replay import replay_component_occupancy
from native import negative_native_progress
from negative_stop_evidence import closed_brush_stop

from rosclaw.connectors.ros.verification.mission import replay_coverage


def bounded_json(path, limit=2_000_000):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise ValueError("bounded retained negative evidence required")
    return json.loads(raw), raw


def validate_d4_negative(root, stop):
    """Combine genuine failed Task/receipt with obstruction, contacts and stop."""
    root = Path(root).resolve()
    terminal, _ = bounded_json(root / "native-negative-terminal.json")
    if (
        type(terminal) is not dict
        or terminal.get("case") != "D4"
        or terminal.get("status") != "CANONICAL_NEGATIVE_TERMINAL_OBSERVED_NOT_PHYSICS_VERIFIED"
        or terminal.get("task_state_modified") is not False
        or type(terminal.get("canonical_receipts")) is not list
        or len(terminal["canonical_receipts"]) != 1
    ):
        raise ValueError("one retained actual D4 Native terminal receipt required")
    bundle = terminal["canonical_receipts"][0]

    class RetainedReceipt:
        def get_execution_receipt(self, action_id):
            if bundle.get("action_id") != action_id:
                raise ValueError("retained receipt does not match actual Native transaction")
            return bundle

    # Re-read the original SQLite in read-only mode. Retained state is not a
    # substitute for TaskKernel or daemon outcome, and no row is ever updated.
    checked = negative_native_progress(root, "D4", RetainedReceipt())
    if checked != terminal:
        raise ValueError("actual Native task/transaction no longer matches retained terminal")
    verification = bundle["receipt"].get("verification_result", {})
    if bundle["receipt"].get("final_state") != "BLOCKED":
        raise ValueError("D4 source-complete partial coverage requires canonical BLOCKED")
    artifact = verification.get("evidence_artifact", {})
    evidence, _ = bounded_json(artifact["path"], limit=1_000_000_000)
    coverage, temporal = replay_coverage(evidence)
    if (
        temporal is None
        or not temporal["complete"]
        or coverage["coverage_ratio"] >= 0.98
        or coverage["trace_gaps"] != 0
    ):
        raise ValueError("D4 requires genuine incomplete coverage with complete observations")
    if list((root / "actions").glob("*.verification.json")):
        raise ValueError("negative episode cannot contain successful mission verification")
    binding, _ = bounded_json(root / "physics_binding.json")
    spec, spec_raw = bounded_json(root / "scenario.json", limit=65536)
    scenario_policy(spec, binding)
    if spec["case"] != "D4":
        raise ValueError("permanent obstruction protocol required")
    audits = list(root.glob("plan-events-*.jsonl"))
    actors = list(root.glob("brush-events-*.jsonl"))
    if len(audits) != 1 or len(actors) != 1:
        raise ValueError("exclusive closed observer and independent actor histories required")
    source = replay_component_occupancy(audits[0], evidence, binding)
    brush_binding, _ = bounded_json(root / "brush_binding.json")
    if any(
        brush_binding.get(k) != binding.get(k)
        for k in ("run_id", "body_snapshot_hash", "attachment_hash")
    ):
        raise ValueError("independent brush and physics bindings differ")
    actuator = closed_brush_stop(actors[0], brush_binding, stop)
    confirm = None
    first_on = None
    stage = 0
    scenario_sha = hashlib.sha256(spec_raw).hexdigest()
    with (root / "dynamic-scenario-events.jsonl").open("rb") as stream:
        for line in iter(lambda: stream.readline(2_000_001), b""):
            if len(line) > 2_000_000 or not line.endswith(b"\n"):
                raise ValueError("bounded complete scene event required")
            row = json.loads(line)
            if (
                type(row) is not dict
                or row.get("scenario_sha256") != scenario_sha
                or row.get("run_id") != binding["run_id"]
                or row.get("mission_id") != binding["mission_id"]
                or row.get("physical_acceptance") != "NOT_VERIFIED"
            ):
                raise ValueError("frozen scene event identity mismatch")
            kinds = [
                "SCENARIO_STARTED",
                "FIRST_ENABLED_CLEANING",
                "INTRODUCTION_REQUESTED",
                "INTRODUCTION_ACK_REQUIRES_ACTUAL_PACKET",
                "ACTUAL_POSTUPDATE_POSITION_CONFIRMED",
                "TASK_RUNNER_STOP_REQUESTED",
            ]
            if stage >= len(kinds) or row.get("kind") != kinds[stage]:
                raise ValueError("permanent obstruction lacks exclusive ordered scene history")
            if stage == 4:
                xy = row.get("actual_xy")
                if (
                    row.get("state") != "CONFIRMING_INTRODUCTION"
                    or type(xy) is not list
                    or len(xy) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)
                    or math.hypot(xy[0] - spec["target_xy"][0], xy[1] - spec["target_xy"][1])
                    >= 0.001
                ):
                    raise ValueError("actual introduced position differs from frozen target")
                confirm = row
            if stage == 1:
                first_on = row.get("sim_time_sec")
                if (
                    type(first_on) not in (int, float)
                    or not math.isfinite(first_on)
                    or first_on < 0
                ):
                    raise ValueError("actual initial cleaning SIM time required")
            if stage == 5 and row.get("final_state") != "OCCUPIED":
                raise ValueError("permanent obstruction not retained until runner stop")
            stage += 1
    if stage != 6:
        raise ValueError("scene history did not finish with permanent obstruction")
    if confirm["sim_time_sec"] < first_on + spec["introduce_after_cleaning_sim_sec"] or not any(
        p["cleaning_enabled"] is True and p["time_sec"] <= confirm["sim_time_sec"]
        for p in evidence["trajectory"]
    ):
        raise ValueError("obstruction must follow actual enabled cleaning in this task")
    occupied_unclean = set(evidence["occupancy_samples"][-1]["occupancy"]["occupied_cells"]) & {
        cell for cell in evidence["grid"]["accessible_cells"] if coverage["mask"][cell] != 1
    }
    if (
        evidence["trajectory"][-1]["time_sec"] < confirm["sim_time_sec"] + spec["dwell_sim_sec"]
        or not occupied_unclean
    ):
        raise ValueError("canonical partial coverage lacks retained occupied unclean cells")
    stop_end = stop["samples"][-1]["time_sec"]
    confirmed_hash = False
    final_packet_time = None
    with audits[0].open("rb") as stream:
        for line in iter(lambda: stream.readline(2_000_001), b""):
            row = json.loads(line)
            if row["kind"] != "physics_snapshot_received":
                continue
            decoded = retained_packet(row, binding)
            stamp = decoded["packet"]["sim_time_sec"]
            if decoded["packet_sha256"] == confirm["packet_sha256"]:
                if stamp != confirm["sim_time_sec"]:
                    raise ValueError("scene confirmation differs from original actual packet")
                confirmed_hash = True
            if stamp >= confirm["sim_time_sec"]:
                obstacle = next(
                    p for p in decoded["model_poses"] if p.model_name == spec["obstacle_name"]
                )
                if (
                    math.hypot(obstacle.x - spec["target_xy"][0], obstacle.y - spec["target_xy"][1])
                    >= 0.001
                ):
                    raise ValueError("permanent obstruction was moved before source closure")
                final_packet_time = stamp
    if not confirmed_hash or final_packet_time is None or final_packet_time < stop_end:
        raise ValueError("actual obstacle source does not reach independent stop window")
    first = evidence["trajectory"][0]["time_sec"]
    observed = 0
    stop_observed = []
    previous_time = None
    first_on_observed = False
    with (root / "witness.jsonl").open("rb") as stream:
        for line in iter(lambda: stream.readline(2_000_001), b""):
            if len(line) > 2_000_000 or not line.endswith(b"\n"):
                raise ValueError("complete bounded contact observer record required")
            row = json.loads(line)
            stamp = row.get("time_sec")
            if type(stamp) not in (int, float) or not math.isfinite(stamp):
                raise ValueError("finite original witness SIM time required")
            if first <= stamp <= stop_end:
                age = row.get("ground_truth_age_ms")
                if (
                    row.get("observation_complete") is not True
                    or type(row.get("collision_count")) is not int
                    or row["collision_count"] != 0
                    or row.get("brush_source_binding") != brush_binding
                    or row.get("physics_source_fault") is not None
                    or row.get("brush_source_fault") is not None
                    or row.get("evidence_domain") != "GAZEBO_PHYSICS"
                    or type(age) not in (int, float)
                    or not math.isfinite(age)
                    or not 0 <= age < 300
                    or (previous_time is not None and not 0 <= stamp - previous_time <= 0.3)
                ):
                    raise ValueError("complete zero-contact observer source required throughout D4")
                observed += 1
                if stamp == first_on and row.get("cleaning_enabled") is True:
                    first_on_observed = True
                previous_time = stamp
                if stop["samples"][0]["time_sec"] <= stamp:
                    stop_observed.append(stamp)
    if (
        not first_on_observed
        or observed < 20
        or len(stop_observed) < 20
        or stop_observed[-1] < stop_end - 0.3
        or stop_observed[0] > stop["samples"][0]["time_sec"] + 0.3
    ):
        raise ValueError("contact observation does not cover independent stop window")
    return {
        "status": "PASS_EXPECTED_SAFE_FAILURE",
        "case": "D4",
        "physical_acceptance": "SIMULATION",
        "task_kernel_succeeded": False,
        "task_state": terminal["task"]["state"],
        "task_state_modified": False,
        "coverage_ratio": coverage["coverage_ratio"],
        "closed_source_replay": source,
        "independent_actuator_stop": actuator,
        "permanent_obstacle_source_confirmed": True,
        "permanently_occupied_unclean_cells": sorted(occupied_unclean),
        "zero_contact_observer_samples": observed,
    }
