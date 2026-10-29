"""Durable, scoped evidence for a completed public-observation coverage slice.

This is not a simulator-truth verifier or a whole-mission score. Callers must
already validate original Body/source identities, timestamps and native results.
The archive preserves those facts so completion is inspectable after execution.
"""

import gzip
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path


def snapshot_record(sample, roles):
    return {
        "body_id": sample.body_id,
        "body_snapshot_hash": sample.body_snapshot_hash,
        "driver_sha256": sample.driver_sha256,
        "odom_epoch": sample.odom_epoch,
        "graph_epoch": sample.graph_epoch,
        "native_s": sample.native_s,
        "captured_monotonic_s": sample.captured_monotonic_s,
        "odom_xy": list(sample.odom_xy),
        "map_xy": None if sample.map_xy is None else list(sample.map_xy),
        "cleaner": sample.cleaner,
        "linear_speed": sample.linear_speed,
        "yaw_rate": sample.yaw_rate,
        "native_motion_settled": sample.native_motion_settled,
        "soc": sample.soc if "battery" in roles else None,
        "docked": sample.docked if "dock" in roles else None,
        "required_roles": sorted(roles),
        "sources": [asdict(fact) for fact in sample.sources if fact.role in roles],
    }


def validate_archive(payload):
    """Validate persisted finite evidence without treating old facts as live."""
    if (
        payload.get("schema") != "rosclaw.finite_coverage_evidence.v1"
        or payload.get("scope") != "finite_observed_coverage_slice_only"
        or payload.get("whole_house_success") is not None
        or payload.get("execution_mode") not in {"SIMULATION", "FIXTURE"}
        or payload.get("judge_read") is not False
    ):
        raise ValueError("Finite non-hardware evidence scope required")
    identity, binding = payload["action"], payload["binding"]
    if (
        identity["capability_id"] != "coverage.execute_slice"
        or identity["body_id"] != binding["body_id"]
        or identity["body_snapshot_hash"] != binding["body_snapshot_hash"]
        or binding["execution_mode"] != payload["execution_mode"]
        or binding.get("durable_task_evidence") is not True
        or not identity["session_id"]
        or not identity["action_id"]
    ):
        raise ValueError("Exact owning action and compiled Body required")
    sources = {source["role"]: source for source in binding["sources"]}
    if set(sources) != {
        "odom",
        "cleaner",
        "occupancy",
        "global_costmap",
        "local_costmap",
        "graph",
        "map_tf",
        "base_tf",
        "scan",
        "depth",
        "battery",
        "dock",
    }:
        raise ValueError("Complete original public source binding required")
    goals = payload["goals"]
    if not (
        type(goals["minimum_completed_work"]) is int
        and 1 <= goals["minimum_completed_work"] <= binding["maximum_work"]
        and type(goals["minimum_completed_strokes"]) is int
        and 0 <= goals["minimum_completed_strokes"] <= binding["maximum_work"]
        and type(payload["completed_work"]) is int
        and type(payload["completed_strokes"]) is int
        and 0
        <= payload["completed_strokes"]
        <= payload["completed_work"]
        <= binding["maximum_work"]
    ):
        raise ValueError("Original finite work goal bounds required")

    def check(sample):
        for key in ["body_id", "body_snapshot_hash", "driver_sha256", "odom_epoch"]:
            if sample[key] != binding[key]:
                raise ValueError("Archive observation identity differs")
        if not {"odom", "cleaner"}.issubset(sample["required_roles"]):
            raise ValueError("Original odometry and cleaner evidence required")
        facts = {fact["role"]: fact for fact in sample["sources"]}
        if set(facts) != set(sample["required_roles"]) or len(facts) != len(sample["sources"]):
            raise ValueError("Missing or ambiguous archive source")
        native, capture = sample["native_s"], sample["captured_monotonic_s"]
        if not all(
            type(v) in (int, float) and math.isfinite(v)
            for v in [
                native,
                capture,
                *sample["odom_xy"],
                sample["linear_speed"],
                sample["yaw_rate"],
            ]
        ):
            raise ValueError("Finite original observation numbers required")
        if type(sample["cleaner"]) is not bool:
            raise ValueError("Actual brush observation required")
        if "map_tf" in sample["required_roles"] and sample["graph_epoch"] != binding["graph_epoch"]:
            raise ValueError("Original public graph epoch changed")
        if "battery" in sample["required_roles"] and not (
            type(sample["soc"]) in (int, float)
            and math.isfinite(sample["soc"])
            and binding["minimum_work_soc"] <= sample["soc"] <= 1
        ):
            raise ValueError("Original observed work energy reserve required")
        if "dock" in sample["required_roles"] and sample["docked"] is not False:
            raise ValueError("Original work requires observed undocked state")
        for role, fact in facts.items():
            source = sources[role]
            if any(
                fact[key] != source[key]
                for key in ["topic", "message_type", "frame_id", "child_frame_id", "publisher_gid"]
            ):
                raise ValueError("Original source identity changed")
            age = capture - fact["received_monotonic_s"]
            if not math.isfinite(age) or not 0 <= age < source["max_receive_age_s"]:
                raise ValueError("Original capture-relative receive freshness failed")
            header = fact["header_native_s"]
            if source["max_header_age_s"] is None:
                if header is not None:
                    raise ValueError("Unexpected source header")
            elif (
                header is None
                or not math.isfinite(header)
                or not -source["max_header_future_s"]
                <= native - header
                <= source["max_header_age_s"]
            ):
                raise ValueError("Original native header freshness failed")

    start, final = payload["initial"], payload["final"]
    check(start)
    check(final)
    if set(start["required_roles"]) != set(sources):
        raise ValueError("Initial full bound-source observation required")
    if (
        final["native_s"] < start["native_s"]
        or final["captured_monotonic_s"] < start["captured_monotonic_s"]
    ):
        raise ValueError("Reversed action evidence interval")
    if not (
        final["native_motion_settled"] is True
        and final["cleaner"] is False
        and abs(final["linear_speed"]) < 0.02
        and abs(final["yaw_rate"]) < 0.03
    ):
        raise ValueError("Fresh final stopped/off original-terminal evidence required")
    primitives = payload["primitives"]
    ids = set()
    for index, step in enumerate(primitives, 1):
        if step["parent_action_id"] != identity["action_id"] or step["primitive_id"] in ids:
            raise ValueError("Replayed or unrelated primitive evidence")
        ids.add(step["primitive_id"])
        expected_id = (
            "pcov_" + hashlib.sha256(f"{identity['action_id']}:{index}".encode()).hexdigest()
        )
        if step["primitive_id"] != expected_id:
            raise ValueError("Original ordered primitive identity required")
        if step["evidence_domain"] != payload["execution_mode"]:
            raise ValueError("Primitive evidence domain differs")
        check(step["before"])
        check(step["after"])
        if (
            not start["captured_monotonic_s"]
            <= step["before"]["captured_monotonic_s"]
            <= step["after"]["captured_monotonic_s"]
            <= final["captured_monotonic_s"]
        ):
            raise ValueError("Primitive falls outside original action interval")
        if step["capability_id"] != "cleaning.disable" and set(
            step["before"]["required_roles"]
        ) != set(sources):
            raise ValueError("Full source guards required before active work")
        if step["after"]["native_motion_settled"] is not True:
            raise ValueError("Unknown original native primitive result")
        dispatch = step["dispatch"]
        if (
            dispatch.get("parent_action_id") != identity["action_id"]
            or dispatch.get("owned_primitive_id") != step["primitive_id"]
        ):
            raise ValueError("Original primitive receipt identity differs")
        if step["capability_id"] == "navigation.follow_path":
            status = dispatch.get("status")
            if type(status) is not int or status not in {4, 5, 6}:
                raise ValueError("Original navigation terminal required")
            path = step["arguments"].get("path_xy")
            if (
                not isinstance(path, list)
                or not 2 <= len(path) <= 1024
                or any(
                    not isinstance(point, list)
                    or len(point) != 2
                    or any(
                        type(value) not in (int, float) or not math.isfinite(value)
                        for value in point
                    )
                    for point in path
                )
            ):
                raise ValueError("Original finite public path required")
            if (
                sum(math.dist(a, b) for a, b in zip(path, path[1:], strict=False)) > 3.0 + 1e-9
                or step["before"]["map_xy"] is None
                or math.dist(path[0], step["before"]["map_xy"]) > 0.10
            ):
                raise ValueError("Original path length or entry limit exceeded")
            if step["final_state"] == "COMPLETED" and (
                status != 4
                or step["after"]["map_xy"] is None
                or math.dist(path[-1], step["after"]["map_xy"]) > 0.05
            ):
                raise ValueError("Actual original navigation endpoint unverified")
        elif step["capability_id"] not in {"cleaning.enable", "cleaning.disable"}:
            raise ValueError("Unexpected nested primitive")
        elif step["arguments"] != {} or step["after"]["cleaner"] is not (
            step["capability_id"] == "cleaning.enable"
        ):
            raise ValueError("Original observed brush phase differs")
        if step["final_state"] == "COMPLETED":
            if step["verification"].get("passed") is not True:
                raise ValueError("Completed primitive lacks original verification")
        elif not (
            step["final_state"] == "FAILED"
            and step["capability_id"] == "navigation.follow_path"
            and dispatch.get("status") in {5, 6}
            and dispatch.get("cancel_result_received") is True
            and dispatch.get("live_path_invalidation")
        ):
            raise ValueError("Unexplained failed primitive in completion evidence")
    trace = payload["samples"]
    previous = None
    brush_distance = 0.0
    brush_samples = 0
    for sample in trace:
        check(sample)
        if (
            not start["captured_monotonic_s"]
            <= sample["captured_monotonic_s"]
            <= final["captured_monotonic_s"]
        ):
            raise ValueError("Sample belongs outside original action interval")
        if previous is not None:
            stamp = next(f["header_native_s"] for f in sample["sources"] if f["role"] == "odom")
            prior = next(f["header_native_s"] for f in previous["sources"] if f["role"] == "odom")
            if stamp <= prior:
                raise ValueError("Replayed or reversed native sample")
            if previous["cleaner"] and sample["cleaner"]:
                brush_distance += math.dist(previous["odom_xy"], sample["odom_xy"])
        brush_samples += int(sample["cleaner"])
        previous = sample
    goals = payload["goals"]
    if (
        payload["completed_work"] < goals["minimum_completed_work"]
        or payload["completed_strokes"] < goals["minimum_completed_strokes"]
        or not primitives
        or not brush_samples
        or (goals["minimum_completed_strokes"] > 0 and brush_distance <= 0.05)
    ):
        raise ValueError("Finite goal or actual brush evidence unmet")
    latest = start
    for step in primitives:
        for phase in ["before", "after"]:
            if "battery" in step[phase]["required_roles"]:
                latest = step[phase]
    return {
        "actual_brush_motion_m": brush_distance,
        "actual_brush_samples": brush_samples,
        "primitive_count": len(primitives),
        "original_source_roles": sorted(sources),
        "last_guarded_work_state": {
            "native_s": latest["native_s"],
            "captured_monotonic_s": latest["captured_monotonic_s"],
            "soc": latest["soc"],
            "docked": latest["docked"],
            "map_xy": latest["map_xy"],
            "scope": "historical public observation at stated capture; no new dispatch permission",
        },
    }


def persist_archive(directory, payload):
    """Write once, fsync and independently read back before any evidence upgrade."""
    summary = validate_archive(payload)
    data = gzip.compress(json.dumps(payload, sort_keys=True, allow_nan=False).encode(), mtime=0)
    digest = hashlib.sha256(data).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    path = directory / (
        hashlib.sha256(payload["action"]["action_id"].encode()).hexdigest() + ".json.gz"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    reread = path.read_bytes()
    if (
        hashlib.sha256(reread).hexdigest() != digest
        or validate_archive(json.loads(gzip.decompress(reread))) != summary
    ):
        raise ValueError("Finite evidence readback integrity failed")
    parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return dict(
        uri=f"rosclaw-evidence://finite-coverage/{path.stem}/{digest}",
        sha256=digest,
        filename=path.name,
        compressed_bytes=len(data),
        **summary,
    )
