"""SDK capture of actual engineering failures, without admitting navigation.

This role saves non-ready observations for diagnosis. It never labels them L1
ready or compiles a navigation workspace from an unready candidate.
"""

import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from simulation_owned_process_fault import pause_owned_process, resume_owned_process
from simulation_unadmitted_sink import require_unadmitted_sink

COUNTERFACTUAL_CASES = {
    "rsk_tf_future_stamp",
    "rsk_global_sensor_frame_unresolvable",
    "rsk_node_clock_mismatch",
    "rsk_controller_plugin_unavailable",
    "rsk_sensor_frame_unresolvable",
}

CASES = COUNTERFACTUAL_CASES | {
    "j100_navigation_onboarding",
    "map_to_odom_missing",
    "scan_stale",
    "nav2_lifecycle_inactive",
    "footprint_undersized",
    "duplicate_velocity_route",
}


def review_engineering_failure(native, candidate_review, requirements):
    """Separate observed engineering failure from discovery's unbound proposal.

    Only diagnose a circle with an explicit empty polygon, matching base frame,
    unique original costmap publisher and fresh parameter readback. Other
    geometry remains for the full immutable-Body L1 reviewer.
    """
    issues = []
    candidate = candidate_review["candidate"]
    geometry = candidate.get("geometry", {})
    radius = geometry.get("physical_radius_m")
    now = datetime.fromisoformat(candidate_review["reviewed_at"])
    if geometry.get("complete") is True and type(radius) in (int, float) and math.isfinite(radius):
        observations = native.get("observations", {})
        for topic in requirements["costmap_topics"]:
            rows = [t for t in native["graph"]["topics"] if t["name"] == topic]
            if len(rows) != 1 or len(rows[0].get("publishers", [])) != 1:
                continue
            node = rows[0]["publishers"][0]
            params = observations.get("node_parameters", {}).get(node, {})
            stamp = observations.get("parameter_captured_at", {}).get(node)
            try:
                age = (now - datetime.fromisoformat(stamp)).total_seconds()
                polygon = json.loads(params.get("footprint", "null"))
            except (ValueError, TypeError):
                continue
            nav_radius = params.get("robot_radius")
            if (
                0 <= age <= 0.5
                and polygon == []
                and params.get("robot_base_frame") == candidate["frames"]["base"]
                and type(nav_radius) in (int, float)
                and math.isfinite(nav_radius)
                and 0 < nav_radius < radius
            ):
                issues.append("COSTMAP_FOOTPRINT_UNDERSIZES_URDF_ENVELOPE:" + topic)
    return {
        "schema_version": "rosclaw.engineering_observation_review.v1",
        "status": "BLOCKED" if issues else candidate_review["status"],
        "original_candidate_status": candidate_review["status"],
        "issue_codes": sorted(issues),
        "source_snapshot_hash": candidate_review["source_snapshot_hash"],
        "reviewed_at": candidate_review["reviewed_at"],
        "authorization": False,
        "binding_verified": False,
    }


def current_capture(state):
    now = datetime.now(UTC)
    if not state.get("native") or state["native"].get("completeness", {}).get("graph") is not True:
        return False
    return all(
        state.get(name) is not None
        and 0 <= (now - datetime.fromisoformat(state[name]["captured_at"])).total_seconds() <= age
        for name, age in [("native", 0.5), ("physics", 0.3)]
    )


def validate_engineering_capture(settings):
    policy = settings.get("engineering_capture")
    if policy is None:
        return None
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
        or settings.get("generic_source_observer") is not True
        or settings.get("fault", "none") != "none"
        or type(policy) is not dict
        or set(policy) != {"schema_version", "case", "phase", "run_id"}
        or policy["schema_version"] != "rosclaw.engineering_failure_capture.v1"
        or policy["case"] not in CASES
        or policy["phase"] not in {"before_repair", "rollback"}
        or (policy["case"] == "j100_navigation_onboarding" and policy["phase"] != "before_repair")
        or policy["run_id"] != settings.get("run_id")
        or "governed_sink" not in settings
    ):
        raise ValueError("closed isolated SDK engineering failure capture required")
    return policy


def counterfactual_readback_available(native, case):
    """Only gate raw capture on actual parameter reads, never task readiness."""
    if case not in COUNTERFACTUAL_CASES:
        return True
    observations = native.get("observations", {})
    now = datetime.fromisoformat(native["captured_at"])
    for node, params in observations.get("node_parameters", {}).items():
        stamp = observations.get("parameter_captured_at", {}).get(node)
        try:
            at = datetime.fromisoformat(stamp)
            if at.tzinfo is None or not 0 <= (now - at).total_seconds() <= 0.5:
                continue
        except (TypeError, ValueError):
            continue
        if case == "rsk_node_clock_mismatch":
            if node.endswith("/controller_server") and params.get("use_sim_time") is False:
                return True
        elif case == "rsk_controller_plugin_unavailable":
            if params.get("FollowPath.plugin") == "rsk_fixture::UnavailableController":
                return True
        elif case == "rsk_tf_future_stamp":
            if node.endswith("/amcl") and params.get("transform_tolerance") == 30.0:
                return True
        elif case in {"rsk_sensor_frame_unresolvable", "rsk_global_sensor_frame_unresolvable"} and (
            params.get("voxel_layer.scan.sensor_frame") == "rsk_missing_sensor_frame"
            or params.get("obstacle_layer.scan.sensor_frame") == "rsk_missing_sensor_frame"
        ):
            return True
    return False


def capture_failure(root, policy, state, spin_until, save):
    """Require original non-ready ROS facts and an untouched zero-command sink."""
    config_path = root / "execution_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    sources = json.loads((root / "initial-source-witness-observer-sources.json").read_text())
    if sources["run_id"] != policy["run_id"]:
        raise ValueError("original source-only observer run differs")
    body_hash = config.get("body_snapshot_hash")
    requirements = json.loads((root / "engineering-original-requirements.json").read_text())
    sink_before = require_unadmitted_sink(
        root, run_id=policy["run_id"], body_snapshot_hash=body_hash
    )
    process_fault = None
    onboarding = policy["case"] == "j100_navigation_onboarding"
    counterfactual = policy["case"] in COUNTERFACTUAL_CASES
    expected_statuses = (
        {"PROPOSED", "UNKNOWN", "BLOCKED"}
        if counterfactual
        else {"PROPOSED"}
        if onboarding
        else {"UNKNOWN", "BLOCKED"}
    )
    try:
        if policy["case"] == "scan_stale":
            # This staged bringup policy represents a bridge left paused after
            # initialization. The repair changes that policy in NEW staging.
            descriptor = json.loads((root / "bringup.json").read_text())
            if descriptor.get("scan_bridge_state") != "paused_after_initialization":
                raise ValueError("declared scan bridge pause policy required")
            process_fault = pause_owned_process(root, "scan_bridge")
            save("engineering-owned-scan-bridge-pause-original.json", process_fault)
        earliest = time.monotonic() + (8 if counterfactual else 2)
        spin_until(
            lambda: (
                time.monotonic() >= earliest
                and current_capture(state)
                and state["native"] is not None
                and counterfactual_readback_available(state["native"], policy["case"])
                and state["review"] is not None
                and review_engineering_failure(state["native"], state["review"], requirements)[
                    "status"
                ]
                in expected_statuses
                and state["physics"] is not None
                and state["physics"].get("observation_complete") is True
            ),
            15,
        )
        sink_after = require_unadmitted_sink(
            root, run_id=policy["run_id"], body_snapshot_hash=body_hash
        )
        save("engineering-original-native.json", state["native"])
        save("engineering-original-system.json", state["model"].to_dict())
        save("engineering-original-candidate-review.json", state["review"])
        save(
            "engineering-original-failure-review.json",
            review_engineering_failure(state["native"], state["review"], requirements),
        )
        save("engineering-original-physics.json", state["physics"])
        save(
            "engineering-capture-ready.json",
            {
                "policy": policy,
                "status": (
                    "ACTUAL_COUNTERFACTUAL_RAW_CAPTURE_EXTERNAL_REVIEW_REQUIRED"
                    if counterfactual
                    else "ACTUAL_UNBOUND_ONBOARDING_CAPTURE_EXTERNAL_REVIEW_REQUIRED"
                    if onboarding
                    else "ACTUAL_NONREADY_ENGINEERING_CAPTURE_EXTERNAL_REVIEW_REQUIRED"
                ),
                "sink_before": sink_before,
                "sink_after": sink_after,
                "navigation_workspace_compiled": False,
                "body_snapshot_hash": body_hash,
                "physical_binding_scope": "SOURCE_ONLY_UNENROLLED; no Body-bound or lease-enabled physical observation claimed",
                "navigation_goals_sent": 0,
                "L1": "NOT_EVALUATED" if onboarding or counterfactual else "NOT_READY",
                "L2": "NOT_RUN",
                "claim_scope": "SIMULATION_ONLY",
            },
        )
        spin_until(lambda: (root / "host_capture_ack.json").exists(), 120)
    finally:
        if process_fault is not None:
            resume_owned_process(process_fault)
            save("engineering-owned-scan-resume-after-capture-original.json", process_fault)
