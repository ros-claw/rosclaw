"""Checkpoint read-only discovery while an owned inactive bootstrap runs.

This SDK child never activates controllers, admits a Body or dispatches tasks.
The outer bootstrap deadline remains authoritative even during discovery.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path


def validate_observation_seconds(seconds):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 1 <= seconds <= 60:
        raise ValueError("finite read-only observation duration within 1..60 seconds required")
    return seconds


def observation_output(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve() != path
        or path.is_relative_to(Path("/evidence"))
    ):
        raise ValueError(
            "absolute non-symlink observation output outside immutable source required"
        )
    return path


def atomic_json(path, raw):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def observe_window(*, seconds, output, discover, spin_once, snapshot, monotonic=time.monotonic):
    """Keep original snapshots plus a hash-bound observation-window sidecar.

    A checkpoint is not a completed window or sensor acceptance. If the outer
    supervisor kills this process, the last complete checkpoint remains usable.
    Sidecar/hash mismatch is incomplete evidence, never evidence of success.
    """
    seconds = validate_observation_seconds(seconds)
    output = observation_output(output)
    sidecar = output.with_suffix(".observation.json")
    history = output.with_suffix(".checkpoints")
    if (
        output.exists()
        or sidecar.exists()
        or sidecar.is_symlink()
        or history.exists()
        or history.is_symlink()
    ):
        raise ValueError("fresh observation outputs required")
    history.mkdir(mode=0o700)
    started = monotonic()
    deadline = started + seconds
    next_checkpoint = started + 1
    checkpoints = 0

    def checkpoint(complete):
        nonlocal checkpoints
        raw = (json.dumps(snapshot(), indent=2, allow_nan=False) + "\n").encode()
        recorded = monotonic()
        metadata = {
            "schema_version": "rosclaw.generic_bootstrap_observation.v1",
            "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
            "started_monotonic_sec": started,
            "deadline_monotonic_sec": deadline,
            "recorded_monotonic_sec": recorded,
            "requested_duration_sec": seconds,
            "observed_duration_sec": recorded - started,
            "checkpoint_count": checkpoints + 1,
            "window_status": "COMPLETE" if complete else "IN_PROGRESS",
            "controller_activation": False,
            "Body_admitted": False,
            "physical_acceptance": "NOT_EVALUATED",
        }
        meta_raw = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode()
        # Preserve each original snapshot independently before updating latest.
        for name, content in (
            (f"{checkpoints + 1:03d}.json", raw),
            (f"{checkpoints + 1:03d}.observation.json", meta_raw),
        ):
            with (history / name).open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        atomic_json(output, raw)
        atomic_json(sidecar, meta_raw)
        checkpoints += 1

    while monotonic() < deadline:
        discover()
        remaining = deadline - monotonic()
        if remaining > 0:
            spin_once(min(0.1, remaining))
        now = monotonic()
        if now >= next_checkpoint and now < deadline:
            checkpoint(False)
            next_checkpoint = now + 1
    checkpoint(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_observation_seconds(args.duration)
    observation_output(args.output)
    # ROS imports occur only in this owned SDK child, never during plan checks.
    sys.path.insert(0, str(Path(__file__).parents[1] / "ros2"))
    import rclpy
    from probe import ReadOnlyProbe

    rclpy.init()
    node = ReadOnlyProbe()
    try:
        observe_window(
            seconds=args.duration,
            output=args.output,
            discover=node.discover_reads,
            spin_once=lambda seconds: rclpy.spin_once(node, timeout_sec=seconds),
            snapshot=node.snapshot,
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
