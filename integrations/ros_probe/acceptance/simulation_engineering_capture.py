"""SDK capture of actual engineering failures, without admitting navigation.

This role saves non-ready observations for diagnosis. It never labels them L1
ready or compiles a navigation workspace from an unready candidate.
"""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from simulation_owned_process_fault import pause_owned_process, resume_owned_process
from simulation_unadmitted_sink import require_unadmitted_sink

CASES = {
    "map_to_odom_missing",
    "scan_stale",
    "nav2_lifecycle_inactive",
    "footprint_undersized",
    "duplicate_velocity_route",
}


def current_capture(state):
    now = datetime.now(UTC)
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
        or policy["run_id"] != settings.get("run_id")
        or "governed_sink" not in settings
    ):
        raise ValueError("closed isolated SDK engineering failure capture required")
    return policy


def capture_failure(root, policy, state, spin_until, save):
    """Require original non-ready ROS facts and an untouched zero-command sink."""
    config_path = root / "execution_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    sources = json.loads((root / "initial-source-witness-observer-sources.json").read_text())
    if sources["run_id"] != policy["run_id"]:
        raise ValueError("original source-only observer run differs")
    body_hash = config.get("body_snapshot_hash")
    sink_before = require_unadmitted_sink(
        root, run_id=policy["run_id"], body_snapshot_hash=body_hash
    )
    process_fault = None
    try:
        if policy["case"] == "scan_stale":
            # This staged bringup policy represents a bridge left paused after
            # initialization. The repair changes that policy in NEW staging.
            descriptor = json.loads((root / "bringup.json").read_text())
            if descriptor.get("scan_bridge_state") != "paused_after_initialization":
                raise ValueError("declared scan bridge pause policy required")
            process_fault = pause_owned_process(root, "scan_bridge")
            save("engineering-owned-scan-bridge-pause-original.json", process_fault)
        earliest = time.monotonic() + 2
        spin_until(
            lambda: (
                time.monotonic() >= earliest
                and current_capture(state)
                and state["native"] is not None
                and state["review"] is not None
                and state["review"]["status"] in {"UNKNOWN", "BLOCKED"}
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
        save("engineering-original-physics.json", state["physics"])
        save(
            "engineering-capture-ready.json",
            {
                "policy": policy,
                "status": "ACTUAL_NONREADY_ENGINEERING_CAPTURE_EXTERNAL_REVIEW_REQUIRED",
                "sink_before": sink_before,
                "sink_after": sink_after,
                "navigation_workspace_compiled": False,
                "body_snapshot_hash": body_hash,
                "physical_binding_scope": "SOURCE_ONLY_UNENROLLED; no Body-bound or lease-enabled physical observation claimed",
                "navigation_goals_sent": 0,
                "L1": "NOT_READY",
                "L2": "NOT_RUN",
                "claim_scope": "SIMULATION_ONLY",
            },
        )
        spin_until(lambda: (root / "host_capture_ack.json").exists(), 120)
    finally:
        if process_fault is not None:
            resume_owned_process(process_fault)
            save("engineering-owned-scan-resume-after-capture-original.json", process_fault)
