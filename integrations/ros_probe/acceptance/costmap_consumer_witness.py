"""SDK-only read-only occupancy-grid originals. No Runtime, action or publisher."""

import argparse
import hashlib
import json
import math
import os
import signal
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path


def inspect_grid(grid, points):
    """Query actual map-frame cells; never guess a missing frame transform."""
    info = grid["info"]
    width, height = info["width"], info["height"]
    resolution = info["resolution"]
    data = grid["data"]
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or not 0 < width * height <= 65536
        or not all(
            type(x) in (int, float) and math.isfinite(x) for x in [resolution, *info["origin"]]
        )
        or resolution <= 0
        or len(info["origin"]) != 3
        or type(data) is not list
        or len(data) != width * height
        or any(type(x) is not int or not -1 <= x <= 100 for x in data)
        or type(points) is not list
        or not 1 <= len(points) <= 16
    ):
        raise ValueError("bounded finite original occupancy grid and explicit points required")
    ox, oy, yaw = info["origin"]
    c, s = math.cos(yaw), math.sin(yaw)
    result = []
    for point in points:
        if (
            type(point) is not dict
            or set(point) != {"frame", "x", "y"}
            or type(point["frame"]) is not str
            or not 0 < len(point["frame"]) <= 512
            or any(
                type(point[k]) not in (int, float) or not math.isfinite(point[k])
                for k in ["x", "y"]
            )
        ):
            raise ValueError("explicit finite point in a declared frame required")
        if point["frame"] != grid["frame_id"]:
            result.append({"point": point, "status": "UNKNOWN_FRAME_TRANSFORM_NOT_PROVIDED"})
            continue
        dx, dy = point["x"] - ox, point["y"] - oy
        x, y = (
            math.floor((c * dx + s * dy) / resolution),
            math.floor((-s * dx + c * dy) / resolution),
        )
        inside = 0 <= x < width and 0 <= y < height
        result.append(
            {
                "point": point,
                "status": "OBSERVED_CELL" if inside else "OUTSIDE_GRID",
                "cell": [x, y],
                "value": data[y * width + x] if inside else None,
            }
        )
    return result


def main():
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.signals import SignalHandlerOptions

    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("isolated owned SDK occupancy-grid capture required")
    settings = json.loads(Path("/settings.json").read_text())
    expected = settings.get("costmap_consumer_capture")
    if (
        expected
        != {
            "schema_version": "rosclaw.costmap_consumer_capture.v1",
            "configuration_sha256": args.configuration_sha256,
        }
        or args.run_id != settings["run_id"]
    ):
        raise ValueError("explicit pinned read-only SDK capture role required")
    raw = Path("/assets/costmap-witness.json").read_bytes()
    if not 0 < len(raw) <= 100000 or hashlib.sha256(raw).hexdigest() != args.configuration_sha256:
        raise ValueError("original witness inputs differ from operator pin")
    config = json.loads(raw)
    if (
        set(config) != {"schema_version", "topics", "points"}
        or config["schema_version"] != "rosclaw.costmap_witness_inputs.v1"
        or type(config["topics"]) is not list
        or not 1 <= len(config["topics"]) <= 2
        or len(set(config["topics"])) != len(config["topics"])
        or any(
            type(t) is not str or not t.startswith("/") or len(t) > 512 for t in config["topics"]
        )
        or type(config["points"]) is not list
        or not 1 <= len(config["points"]) <= 16
    ):
        raise ValueError("bounded explicit read-only grid topics and query points required")
    requirements = json.loads(Path("/assets/requirements.json").read_text())
    if not set(config["topics"]) <= set(requirements["costmap_topics"]):
        raise ValueError("witness topics must come from independently pinned requirements")
    root = Path("/evidence")
    state = {
        "samples": 0,
        "bytes": 0,
        "run_id": args.run_id,
        "configuration_sha256": args.configuration_sha256,
        "source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "motion_commands": 0,
        "action_goals": 0,
    }
    index = (root / "costmap-consumer-index.jsonl").open("x")
    # The parent owns this process. Default rclpy SIGTERM teardown can invalidate
    # the context while the executor is building its next wait set. Defer SDK
    # shutdown until the bounded spin returns instead of masking RCLError.
    requested_shutdown = []

    def request_shutdown(number, _frame):
        requested_shutdown.append(number)

    rclpy.init(
        args=["--ros-args", "-p", "use_sim_time:=true"],
        signal_handler_options=SignalHandlerOptions.NO,
    )
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    node = Node("rosclaw_costmap_consumer_witness")

    def received(topic, message):
        receive_ns = time.monotonic_ns()
        origin = message.info.origin
        q = origin.orientation
        if (
            not all(math.isfinite(v) for v in [q.x, q.y, q.z, q.w])
            or abs(q.x) > 1e-8
            or abs(q.y) > 1e-8
            or not math.isclose(q.z * q.z + q.w * q.w, 1.0, abs_tol=1e-6)
        ):
            raise ValueError("planar normalized grid origin required")
        grid = {
            "topic": topic,
            "frame_id": message.header.frame_id,
            "info": {
                "width": message.info.width,
                "height": message.info.height,
                "resolution": message.info.resolution,
                "origin": [origin.position.x, origin.position.y, 2 * math.atan2(q.z, q.w)],
            },
            "data": list(message.data),
            "stamp_sec": message.header.stamp.sec,
            "stamp_nanosec": message.header.stamp.nanosec,
            "received_at": datetime.now(UTC).isoformat(),
            "received_monotonic_ns": receive_ns,
            "run_id": args.run_id,
        }
        grid["queries"] = inspect_grid(grid, config["points"])
        payload = (json.dumps(grid, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if state["samples"] >= 1000 or state["bytes"] + len(payload) > 20000000:
            raise RuntimeError("bounded original costmap witness budget exhausted")
        filename = f"costmap-consumer-original-{state['samples']:06}.json"
        with (root / filename).open("xb") as stream:
            stream.write(payload)
        row = {
            "file": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "topic": topic,
            "received_at": grid["received_at"],
            "received_monotonic_ns": receive_ns,
            "queries": grid["queries"],
        }
        index.write(json.dumps(row) + "\n")
        index.flush()
        with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(row, stream)
        os.replace(temporary, root / "costmap-consumer-latest.json")
        state["samples"] += 1
        state["bytes"] += len(payload)

    subscriptions = [
        node.create_subscription(
            OccupancyGrid,
            topic,
            lambda message, topic=topic: received(topic, message),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        for topic in config["topics"]
    ]
    try:
        while not requested_shutdown:
            rclpy.spin_once(node, timeout_sec=0.05)
        state.update(
            status="FINISHED_READ_ONLY_CAPTURE_NOT_ACCEPTANCE",
            operator_shutdown_signal=requested_shutdown[0],
        )
    except BaseException as error:
        state.update(status="FAILED_ORIGINAL_READ_ONLY_CAPTURE", error=str(error))
        raise
    finally:
        state.setdefault("status", "FINISHED_READ_ONLY_CAPTURE_NOT_ACCEPTANCE")
        state.update(subscriptions=len(subscriptions))
        (root / "costmap-consumer-outcome.json").write_text(json.dumps(state, indent=2) + "\n")
        index.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
