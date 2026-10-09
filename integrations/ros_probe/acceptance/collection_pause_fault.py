"""Retain collector-fault correspondence; no acceptance, stop or authority."""

import base64
import hashlib
import math
import time
from pathlib import Path

from backend_probe_world import bounded_source
from probe_scene_geometry import decode_scene_json


def retain_source_loss(directory, error):
    root = Path(directory)
    paths = {
        "policy": root / "backend-collection-pause-policy.json",
        "request": root / "backend-collection-pause-request.json",
        "requested": root / "backend-collection-pause-record.json",
        "applied": root / "backend-collection-pause-applied.json",
        "before": root / "backend-collection-pause-before.json",
        "projection": root / "backend-observer/backend-observation-latest.json",
    }
    originals = {key: bounded_source(path, 2_000_000) for key, path in paths.items()}
    decoded = {key: decode_scene_json(raw) for key, raw in originals.items()}
    policy_hash = hashlib.sha256(originals["policy"]).hexdigest()
    if (
        decoded["request"] != {"fixture_policy_sha256": policy_hash}
        or decoded["requested"].get("fixture_policy_sha256") != policy_hash
        or decoded["applied"].get("fixture_policy_sha256") != policy_hash
        or decoded["before"].get("policy_sha256") != policy_hash
        or decoded["applied"].get("actual_linux_process_state") != "T"
        or decoded["requested"].get("observer_pid") != decoded["applied"].get("observer_pid")
        or decoded["requested"].get("World_or_robot_signaled") is not False
    ):
        raise ValueError("original registered request and actual owned collector pause required")
    now = time.monotonic()
    stamp = decoded["projection"]["snapshot"]["sampled_monotonic_sec"]
    pause_time = decoded["applied"]["confirmed_monotonic_sec"]
    if (
        type(stamp) not in (int, float)
        or type(pause_time) not in (int, float)
        or not math.isfinite(stamp)
        or not math.isfinite(pause_time)
        or stamp > now
        or pause_time > now
    ):
        raise ValueError("typed original collector source timing required")
    return {
        "schema_version": "rosclaw.collection_pause_source_loss.v1",
        "observed_monotonic_sec": now,
        "original_error": str(error),
        "collector_stale_at_first_fault": now - stamp >= 0.15,
        "fault_observed_after_actual_pause": now >= pause_time,
        "post_fault_contact_state": "UNKNOWN_REQUIRES_INDEPENDENT_SOURCE",
        "physical_acceptance": "NOT_VERIFIED",
        "physical_stop_proof": "NOT_MEASURED",
        "authorization": False,
        "originals": {
            key: {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "original_base64": base64.b64encode(raw).decode("ascii"),
            }
            for key, raw in originals.items()
        },
    }
