"""Passive Gazebo fixture perturbations; never publishes robot motor commands.

Run inside the disposable fixture container. Contact injection is destructive
only to this simulated episode and requires a reset before a normal journey.
"""

import argparse
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/evidence")


def observed():
    with (ROOT / "witness.jsonl").open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 32768))
        line = stream.read().splitlines()[-1]
    sample = json.loads(line)
    captured = datetime.fromisoformat(sample["captured_at"])
    if not 0 <= (datetime.now(UTC) - captured).total_seconds() < 1:
        raise RuntimeError("fresh independent fixture observation is required")
    if not sample["observation_complete"]:
        raise RuntimeError("independent observation is incomplete")
    return sample


def service(name, request_type, request):
    result = subprocess.run(
        [
            "gz",
            "service",
            "-s",
            "/world/ros_expert/" + name,
            "--reqtype",
            request_type,
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            request,
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    if "data: true" not in result.stdout:
        raise RuntimeError("fixture mutation was not acknowledged: " + result.stdout)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["contact", "dynamic"])
    parser.add_argument("--x", type=float, default=0)
    parser.add_argument("--y", type=float, default=0.5)
    parser.add_argument("--dwell", type=float, default=20)
    args = parser.parse_args()
    if not math.isfinite(args.dwell) or not 1 <= args.dwell <= 60:
        parser.error("dwell must be between 1 and 60 seconds")
    before = observed()
    if args.case == "contact":
        if before["cleaning_enabled"] or before["lease_remaining_sec"] > 0:
            raise RuntimeError("contact self-test requires an inactive daemon lease")
        x, y, size = before["x"], before["y"], "0.2 0.2 0.3"
    else:
        x, y, size = args.x, args.y, "0.25 0.25 0.6"
        if (
            not all(math.isfinite(v) for v in (x, y))
            or math.hypot(x - before["x"], y - before["y"]) < 0.65
        ):
            raise RuntimeError("dynamic obstacle spawn must be clear of the actual robot")
    name = "ros_expert_test_obstacle"
    z = 0.1 if args.case == "contact" else 0.3
    sdf = f'<sdf version="1.9"><model name="{name}"><static>true</static><pose>{x} {y} {z} 0 0 0</pose><link name="body"><collision name="body"><geometry><box><size>{size}</size></box></geometry></collision><visual name="body"><geometry><box><size>{size}</size></box></geometry></visual></link></model></sdf>'
    record = {
        "case": args.case,
        "captured_at": datetime.now(UTC).isoformat(),
        "before": before,
        "position": [x, y],
        "source": "gazebo_fixture_world_control",
    }
    record["create_response"] = service(
        "create", "gz.msgs.EntityFactory", "sdf: " + json.dumps(sdf)
    )
    try:
        samples = []
        dwell = 5 if args.case == "contact" else args.dwell
        until = time.monotonic() + dwell
        while time.monotonic() < until:
            samples.append(observed())
            time.sleep(0.1)
        record["observations"] = samples
        record["dwell_sec"] = dwell
        if args.case == "contact":
            record["passed"] = (
                max(s["physics_collision_count"] for s in samples)
                > before["physics_collision_count"]
            )
        else:
            record["passed"] = all(s["collision_count"] == 0 for s in samples)
    finally:
        record["remove_response"] = service(
            "remove", "gz.msgs.Entity", f'name: "{name}" type: MODEL'
        )
        record["removed_at"] = datetime.now(UTC).isoformat()
        (ROOT / (args.case + "_fault.json")).write_text(json.dumps(record, indent=2) + "\n")
    if not record["passed"]:
        raise RuntimeError("fixture perturbation acceptance failed")
    print(json.dumps({"case": args.case, "passed": True}))


if __name__ == "__main__":
    main()
