"""Closed D5 collection-loss safe-failure correspondence; no robot access."""

import base64
import hashlib
import math
from pathlib import Path

from native import negative_native_progress
from negative_dynamic_acceptance import bounded_json
from negative_stop_evidence import closed_brush_stop
from owned_collection_pause import OwnedCollectionPause
from probe_scene_geometry import decode_scene_json


def validate_collection_negative(directory, stop):
    root = Path(directory).resolve()
    terminal, _ = bounded_json(root / "native-negative-terminal.json")
    if (
        type(terminal) is not dict
        or terminal.get("case") != "D5"
        or terminal.get("status") != "CANONICAL_NEGATIVE_TERMINAL_OBSERVED_NOT_PHYSICS_VERIFIED"
        or terminal.get("task_state_modified") is not False
        or type(terminal.get("canonical_receipts")) is not list
        or len(terminal["canonical_receipts"]) != 1
    ):
        raise ValueError("one genuine D5 negative Native terminal required")
    bundle = terminal["canonical_receipts"][0]

    class RetainedReceipt:
        def get_execution_receipt(self, action_id):
            if bundle.get("action_id") != action_id:
                raise ValueError("canonical negative receipt does not match actual transaction")
            return bundle

    checked = negative_native_progress(root, "D5", RetainedReceipt())
    if checked != terminal or list((root / "actions").glob("*.verification.json")):
        raise ValueError("actual negative root/transaction differs or a success exists")
    loss, _ = bounded_json(root / "backend-collection-source-loss.json", limit=20_000_000)
    if (
        loss.get("schema_version") != "rosclaw.collection_pause_source_loss.v1"
        or loss.get("collector_stale_at_first_fault") is not True
        or loss.get("fault_observed_after_actual_pause") is not True
        or loss.get("authorization") is not False
        or loss.get("physical_stop_proof") != "NOT_MEASURED"
        or loss.get("post_fault_contact_state") != "UNKNOWN_REQUIRES_INDEPENDENT_SOURCE"
        or type(loss.get("originals")) is not dict
        or set(loss["originals"])
        != {"policy", "request", "requested", "applied", "before", "projection"}
    ):
        raise ValueError("original actual collector staleness and applied pause required")
    decoded = {}
    names = {
        "policy": "policy",
        "request": "request",
        "requested": "record",
        "applied": "applied",
        "before": "before",
    }
    for key, original in loss["originals"].items():
        raw = base64.b64decode(original["original_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != original.get("sha256"):
            raise ValueError("retained original collection-loss bytes changed")
        decoded[key] = decode_scene_json(raw)
        if (
            key in names
            and (root / f"backend-collection-pause-{names[key]}.json").read_bytes() != raw
        ):
            raise ValueError("closed registered collector source changed after first failure")
    fault_time = loss.get("observed_monotonic_sec")
    source_time = decoded["projection"]["snapshot"]["sampled_monotonic_sec"]
    pause_time = decoded["applied"]["confirmed_monotonic_sec"]
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in (fault_time, source_time, pause_time)
        )
        or fault_time - source_time < 0.15
        or fault_time < pause_time
    ):
        raise ValueError("actual original first-fault stale interval required")
    policy_sha = loss["originals"]["policy"]["sha256"]
    if (
        decoded["request"] != {"fixture_policy_sha256": policy_sha}
        or decoded["applied"].get("actual_linux_process_state") != "T"
        or decoded["applied"].get("fixture_policy_sha256") != policy_sha
        or decoded["requested"].get("fixture_policy_sha256") != policy_sha
        or decoded["requested"].get("observer_pid") != decoded["applied"].get("observer_pid")
        or decoded["requested"].get("World_or_robot_signaled") is not False
        or decoded["before"].get("policy_sha256") != policy_sha
    ):
        raise ValueError("actual owned collector pause correspondence differs")
    prefix, _ = bounded_json(root / "closed-collection-prefix-source.json")
    plan, _ = bounded_json(root / "backend-stack-source-plan.json")
    policy = decoded["policy"]

    class NoChildren:
        children = ()

    OwnedCollectionPause(root, NoChildren(), plan, root / "backend-collection-pause-policy.json")
    if decoded["requested"].get("fixture_policy") != policy:
        raise ValueError("original requested pause policy differs from registered source")
    if (
        any(
            policy.get(k) != plan.get(k)
            for k in ("run_id", "body_snapshot_hash", "constraint_policy_hash")
        )
        or prefix.get("constraint_policy_hash") != plan.get("constraint_policy_hash")
        or prefix.get("authorization") is not False
        or prefix.get("physical_acceptance") != "NOT_VERIFIED"
        or prefix.get("source_window") != "PREFIX_BEFORE_REGISTERED_SOURCE_LOSS"
        or prefix.get("post_prefix_source_health") != "UNKNOWN"
        or prefix.get("original_service_wire_required") is not True
        or prefix.get("original_service_SDK_wire_replays", 0) < 1
        or prefix.get("spatial_source_join_required") is not True
        or prefix.get("completed_exact_scene_joins", 0) < 1
        or prefix.get("robot_collision_count") != 0
        or prefix.get("probe_completed_cache_cycles", 0) < 1
        or prefix.get("healthy_prefix_monotonic_sec")
        != decoded["before"]["backend_projection"]["snapshot"]["sampled_monotonic_sec"]
    ):
        raise ValueError("original all-step/SDK healthy prefix correspondence required")
    for path, key in (
        (root / "backend-observer/backend-observation-events.jsonl", "original_source_sha256"),
        (root / "backend-observer/backend-observation-events.jsonl.summary.json", "summary_sha256"),
    ):
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 1_000_000_000:
            raise ValueError("bounded original closed backend source required")
        with path.open("rb") as source:
            actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if actual_hash != prefix.get(key):
            raise ValueError("actual SDK-prefix original audit/summary changed")
    actors = list(root.glob("brush-events-*.jsonl"))
    if len(actors) != 1:
        raise ValueError("one independent closed actuator history required")
    brush, _ = bounded_json(root / "brush_binding.json")
    if any(brush.get(k) != plan.get(k) for k in ("run_id", "body_snapshot_hash")):
        raise ValueError("independent actuator and original backend Body differ")
    actuator = closed_brush_stop(actors[0], brush, stop)
    return {
        "case": "D5",
        "subcase": "OWNED_COLLECTION_PAUSE",
        "task_state": checked["task"]["state"],
        "task_kernel_succeeded": False,
        "independent_brush_off_and_lease_release": actuator,
        "original_source_before_loss": prefix,
        "post_loss_robot_contacts": "UNKNOWN",
        "physical_acceptance": "SIMULATION_SAFE_FAILURE",
        "other_D5_subcases": "NOT_RUN",
        "v1_done": False,
    }
