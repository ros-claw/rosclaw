"""Read-only probe worker boundary for Agent context.

The Agent exchanges bounded JSON with this connector worker; it never opens a
ROS connection, imports native ROS bindings, registers an executor or drives a
robot. The worker exposes only the existing read-only snapshot operation.
"""

import json
import subprocess
import sys

from rosclaw.connectors.ros.intelligence import RosSystemModel

MAX_SNAPSHOT_BYTES = 5_000_000


def read_snapshot(*, endpoint, robot_id, deep=True, body=None):
    request = {"endpoint": endpoint, "robot_id": robot_id, "deep": deep, "body": body}
    result = subprocess.run(
        [sys.executable, "-m", "rosclaw.connectors.ros.context.probe_client"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=4,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("read-only ROS probe worker failed: " + result.stderr[-300:])
    if len(result.stdout.encode()) > MAX_SNAPSHOT_BYTES:
        raise ValueError("ROS snapshot exceeds the context observation limit")
    return RosSystemModel.from_dict(json.loads(result.stdout))


def main():
    from rosclaw.connectors.ros.expert import inspect_system

    request = json.loads(sys.stdin.read(MAX_SNAPSHOT_BYTES + 1))
    if not isinstance(request, dict) or set(request) != {"endpoint", "robot_id", "deep", "body"}:
        raise ValueError("probe worker accepts only the read-only snapshot request")
    model = inspect_system(**request)
    payload = json.dumps(model.to_dict())
    if len(payload.encode()) > MAX_SNAPSHOT_BYTES:
        raise ValueError("ROS snapshot exceeds the context observation limit")
    print(payload)


if __name__ == "__main__":
    main()
