"""Pinned SDK operator localization refresh for stopped disposable SIM bodies.

Calls Nav2's documented request_nomotion_update service. AMCL still consumes
actual laser/odometry and publishes its own estimated covariance. This process
publishes no pose, transform, command or action and never changes a Body limit.
"""

import json
import os
import time
from contextlib import suppress
from pathlib import Path

import rclpy
from gazebo_source_witness.prepare import check_prepared_assets
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from simulation_source_capture import BoundedSourceJsonlCapture
from std_srvs.srv import Empty


def main():
    if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("isolated operator-owned SIM fixture required")
    settings = json.loads(Path("/settings.json").read_text())
    if "governed_sink" not in settings:
        raise ValueError("explicit governed SIM fixture role required")
    check_prepared_assets(
        "/assets", settings["inventory_sha256"], "/tmp/localization-refresh-assets"
    )
    sources = json.loads(Path("/assets/observer-sources.json").read_text())
    config = json.loads(Path("/assets/localization-refresh.json").read_text())
    namespace = sources["observer_node"].rsplit("/", 1)[0]
    if (
        set(config)
        != {
            "schema_version",
            "run_id",
            "service",
            "period_sec",
            "operator_owned_disposable_sim_only",
        }
        or config["schema_version"] != "rosclaw.sim_localization_refresh.v1"
        or config["run_id"] != settings["run_id"]
        or config["run_id"] != sources["run_id"]
        or config["operator_owned_disposable_sim_only"] is not True
        or config["service"] != namespace + "/request_nomotion_update"
        or type(config["period_sec"]) not in (int, float)
        or not 0.1 <= config["period_sec"] <= 0.2
    ):
        raise ValueError("pinned closed operator localization refresh context required")
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    node = Node("rosclaw_fixture_localization_refresh", namespace=namespace)
    audit = BoundedSourceJsonlCapture(
        Path("/evidence/localization-refresh-original.jsonl"),
        max_segments=1,
        max_bytes=2_000_000,
        max_rows=5000,
        max_line_bytes=4096,
    )
    client = node.create_client(Empty, config["service"])
    pending = None
    sent_ns = None

    def record(event):
        audit.write(
            json.dumps(
                {
                    "event": event,
                    "monotonic_ns": time.monotonic_ns(),
                    "run_id": config["run_id"],
                    "service": config["service"],
                }
            )
            + "\n"
        )

    def tick():
        nonlocal pending, sent_ns
        if pending is not None:
            if pending.done():
                pending.result()
                record("NOMOTION_UPDATE_ACKNOWLEDGED")
                pending = None
            elif time.monotonic_ns() - sent_ns > 300_000_000:
                raise ValueError("operator localization refresh service response expired")
            else:
                return
        if client.service_is_ready():
            sent_ns = time.monotonic_ns()
            pending = client.call_async(Empty.Request())
            record("NOMOTION_UPDATE_REQUESTED")

    node.create_timer(config["period_sec"], tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
    try:
        with suppress(ExternalShutdownException, KeyboardInterrupt):
            rclpy.spin(node)
    finally:
        audit.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
