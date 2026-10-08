"""Bounded read-only live stop measurement for successful and failed fixtures.

Pose calculation never grants task acceptance. Cleaner/lease/contact/source and
canonical task gates remain separate, especially during observer-loss faults.
"""

import math
import time
from datetime import UTC, datetime

from observations import latest_completed_observation

SOURCE = "independent_gazebo_ground_truth_subscription"


def stop_geometry(samples):
    if not isinstance(samples, list) or not 20 <= len(samples) <= 128:
        raise ValueError("bounded complete stop window required")
    previous_wall = previous_sim = None
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("source") != SOURCE:
            raise ValueError("independent Gazebo pose source required")
        if any(
            type(sample.get(k)) not in (int, float) or not math.isfinite(sample[k])
            for k in ("x", "y", "yaw", "time_sec")
        ):
            raise ValueError("finite independent pose and SIM time required")
        stamp = datetime.fromisoformat(sample["captured_at"])
        if stamp.tzinfo is None:
            raise ValueError("timezone-aware independent pose receipt required")
        if previous_wall is not None and (
            stamp <= previous_wall or sample["time_sec"] <= previous_sim
        ):
            raise ValueError("stop requires distinct advancing actual physics samples")
        previous_wall, previous_sim = stamp, sample["time_sec"]
    wall_span = (previous_wall - datetime.fromisoformat(samples[0]["captured_at"])).total_seconds()
    sim_span = previous_sim - samples[0]["time_sec"]
    if wall_span < 2.5 or sim_span < 2.5:
        raise ValueError("three-second collection lacks sufficient actual physics span")
    displacement = max(
        math.hypot(s["x"] - samples[0]["x"], s["y"] - samples[0]["y"]) for s in samples
    )
    rotation = max(
        abs(
            math.atan2(
                math.sin(s["yaw"] - samples[0]["yaw"]), math.cos(s["yaw"] - samples[0]["yaw"])
            )
        )
        for s in samples
    )
    if displacement > 0.01 or rotation > 0.03:
        raise ValueError("independent actual standstill threshold exceeded")
    return {
        "evidence_role": "INDEPENDENT_STOP_GEOMETRY_NOT_TASK_ACCEPTANCE",
        "physical_acceptance": "NOT_VERIFIED",
        "sample_count": len(samples),
        "wall_span_sec": wall_span,
        "sim_span_sec": sim_span,
        "displacement_m": displacement,
        "yaw_change_rad": rotation,
    }


def collect_stop_geometry(path):
    samples = []
    until = time.monotonic() + 3
    while time.monotonic() < until:
        sample = latest_completed_observation(path)
        age = (datetime.now(UTC) - datetime.fromisoformat(sample["captured_at"])).total_seconds()
        if not 0 <= age < 0.3:
            raise ValueError("fresh independent stop pose required")
        if not samples or sample != samples[-1]:
            samples.append(sample)
        time.sleep(0.1)
    return {**stop_geometry(samples), "samples": samples}
