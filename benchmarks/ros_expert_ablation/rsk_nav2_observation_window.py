"""Copy a bounded read-only window from an already owned SDK's original files.

This module starts no process and imports no ROS, Runtime, driver or executor.
The source-only SDK owner must keep its existing capture loop alive. A window
does not enroll a Body, renew the original case, or authorize an action.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from integrations.ros_probe.acceptance.simulation_atomic_read import read_bounded_regular

OPERATIONS = frozenset({"ENDPOINT_QOS", "CLOCK_TF_WINDOW", "NATIVE_DIAGNOSTICS"})
WINDOW_SEC = 5.6
MAX_NATIVE_BYTES = 8_000_000


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _put(directory, name, raw):
    path = directory / name
    with path.open("xb") as stream:
        stream.write(raw)
    return {"file": name, "sha256": _sha(raw), "bytes": len(raw)}


def _unadmitted_sink(execution, run_id, body_snapshot_hash):
    raw = read_bounded_regular(
        execution / "governed-sink-status.json", 16384, allow_atomic_unlink=True
    )
    sink = json.loads(raw)
    required = {
        "run_id",
        "body_snapshot_hash",
        "terminal",
        "action_id",
        "daemon_instance_id",
        "lease_expires_ns",
        "lease_updates",
        "received_commands",
        "published_velocity_messages",
    }
    if (
        not required <= set(sink)
        or sink["run_id"] != run_id
        or sink["body_snapshot_hash"] != body_snapshot_hash
        or sink["terminal"] is not False
        or sink["action_id"] is not None
        or sink["daemon_instance_id"] is not None
        or any(
            type(sink[key]) is not int or sink[key] != 0
            for key in [
                "lease_expires_ns",
                "lease_updates",
                "received_commands",
                "published_velocity_messages",
            ]
        )
    ):
        raise ValueError("original never-admitted zero-command SDK sink required")
    for name in ["admission.json", "proof.json"]:
        try:
            os.lstat(execution / "execution-control" / name)
        except FileNotFoundError:
            continue
        raise ValueError("read-only SDK diagnostic window cannot use an admitted sink")
    return raw


def _native(execution):
    pointer_raw = read_bounded_regular(
        execution / "live-native-pointer.json", 4096, allow_atomic_unlink=True
    )
    pointer = json.loads(pointer_raw)
    observed_ns = time.monotonic_ns()
    if (
        type(pointer) is not dict
        or set(pointer) != {"file", "sha256", "captured_at", "received_monotonic_ns"}
        or type(pointer["file"]) is not str
        or re.fullmatch(r"original-native-[0-9]{3,6}\.json", pointer["file"]) is None
        or type(pointer["received_monotonic_ns"]) is not int
        or not 0 <= observed_ns - pointer["received_monotonic_ns"] <= 500_000_000
    ):
        raise ValueError("current original SDK Native pointer within500ms required")
    raw = read_bounded_regular(execution / pointer["file"], MAX_NATIVE_BYTES)
    native = json.loads(raw)
    stamp = datetime.fromisoformat(native["captured_at"])
    if (
        _sha(raw) != pointer["sha256"]
        or pointer["captured_at"] != native["captured_at"]
        or stamp.tzinfo is None
        or native.get("environment", {}).get("rmw") != "rmw_cyclonedds_cpp"
        or native.get("environment", {}).get("ros_generation") != "ros2"
        or native.get("completeness", {}).get("graph") is not True
    ):
        raise ValueError("complete original fixed-RMW Native hash/time binding required")
    return pointer_raw, raw, stamp, observed_ns, pointer


def capture_owned_unadmitted_window(
    execution,
    output,
    *,
    run_id,
    body_snapshot_hash,
    source_revision,
    operation,
):
    """Observe one already owned source-only SDK, without any state mutation.

    SDK/container/source/config and original target-scope verification remain
    obligations of the closed operator/reviewer. This file collector alone is
    not a model provider, a pilot gate, or evidence of navigation/Body admission.
    """
    if (
        type(run_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
        or type(source_revision) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
        or operation not in OPERATIONS
    ):
        raise ValueError("frozen source/run and closed diagnostic operation required")
    execution = Path(execution).resolve(strict=True)
    output = Path(output)
    output.mkdir(exist_ok=False)
    started_ns = time.monotonic_ns()
    before_sink = _unadmitted_sink(execution, run_id, body_snapshot_hash)
    before_read_started_ns = time.monotonic_ns()
    before_pointer, before_raw, before_stamp, before_ns, before = _native(execution)
    before_read_duration_sec = (time.monotonic_ns() - before_read_started_ns) / 1_000_000_000
    references = {
        "native_before": _put(output, "native-before-original.json", before_raw),
        "pointer_before": _put(output, "pointer-before-original.json", before_pointer),
        "sink_before": _put(output, "sink-before-original.json", before_sink),
    }
    time.sleep(WINDOW_SEC)
    after_read_started_ns = time.monotonic_ns()
    after_pointer, after_raw, after_stamp, after_ns, after = _native(execution)
    after_read_duration_sec = (time.monotonic_ns() - after_read_started_ns) / 1_000_000_000
    after_sink = _unadmitted_sink(execution, run_id, body_snapshot_hash)
    elapsed = (time.monotonic_ns() - started_ns) / 1_000_000_000
    span = (after_stamp - before_stamp).total_seconds()
    if (
        not WINDOW_SEC <= elapsed <= 15
        or not 5 <= span <= 10
        or after_ns <= before_ns
        or before["file"] == after["file"]
        or before["sha256"] == after["sha256"]
        or int(after["file"][16:-5]) <= int(before["file"][16:-5])
    ):
        raise ValueError("bounded distinct increasing original SDK observation window required")
    references.update(
        {
            "native_after": _put(output, "native-after-original.json", after_raw),
            "pointer_after": _put(output, "pointer-after-original.json", after_pointer),
            "sink_after": _put(output, "sink-after-original.json", after_sink),
        }
    )
    result = {
        "schema_version": "rosclaw.rsk_owned_nav2_observation_window.v2",
        "source_revision": source_revision,
        "run_id": run_id,
        "requested_operation": operation,
        "native_capture_span_sec": span,
        "host_window_elapsed_sec": elapsed,
        "host_observed_monotonic_ns_before": before_ns,
        "host_observed_monotonic_ns_after": after_ns,
        "native_original_file_read_durations_sec": [
            before_read_duration_sec,
            after_read_duration_sec,
        ],
        "native_original_file_before": before["file"],
        "native_original_file_after": after["file"],
        "source_body_snapshot_hash": body_snapshot_hash,
        **references,
        "source_only_unenrolled": True,
        "canonical_navigation_success": False,
        "Body_or_L1_admission": "NOT_GRANTED",
        "case_rebound": False,
        "control_commands_sent": 0,
        "read_only_collector_starts_Worlds": 0,
        "execution_authority": "NONE",
    }
    _put(
        output,
        "window-original-manifest.json",
        (json.dumps(result, indent=2) + "\n").encode(),
    )
    return result
