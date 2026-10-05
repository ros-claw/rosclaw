"""Measured CPU-fallback / CUDA-IPC comparison using official backend test nodes.

Run in an owned Isaac ROS 5/Lyrical GPU container with the official
rosidl_buffer_backends cuda_buffer_backend test components built and sourced.
This is a transport benchmark, not a cleaning benchmark or a full copy trace.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def collect(output, backend):
    import rclpy
    from std_msgs.msg import Bool, Float64, UInt32

    rclpy.init()
    node = rclpy.create_node("transport_benchmark_observer")
    latencies = []
    validation = []
    counts = []
    node.create_subscription(Float64, "/latency_ms", lambda m: latencies.append(m.data), 1000)
    node.create_subscription(Bool, "/validation_result", lambda m: validation.append(m.data), 1000)
    node.create_subscription(Bool, "/backend_validation", lambda m: validation.append(m.data), 1000)
    node.create_subscription(UInt32, "/subscriber_count", lambda m: counts.append(m.data), 1000)
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline and (not counts or max(counts) < 300):
            rclpy.spin_once(node, timeout_sec=0.1)
        ordered = sorted(latencies[20:])
        passed = bool(ordered) and counts and max(counts) >= 290 and validation and all(validation)
        result = {
            "status": "PASS" if passed else "FAIL",
            "captured_at": datetime.now(UTC).isoformat(),
            "buffer_backend": backend,
            "image_width": 1920,
            "image_height": 1080,
            "payload_bytes": 1920 * 1080 * 3,
            "publish_period_ms": 50,
            "published_target": 300,
            "received": max(counts, default=0),
            "warmup_samples_discarded": 20,
            "latency_samples_ms": latencies,
            "latency_median_ms": statistics.median(ordered) if ordered else None,
            "latency_p95_ms": ordered[int(0.95 * (len(ordered) - 1))] if ordered else None,
            "content_and_backend_validated": bool(validation) and all(validation),
            "copy_trace_complete": False,
            "zero_copy_verified": False,
            "source": "official_cuda_buffer_backend_components_and_observed_ROS_metrics",
            "notes": "Subscriber validates pixels by copying CUDA output to CPU. Timing measures arrival before validation; this is not a zero-copy endpoint or pure GPU compute benchmark.",
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        if not passed:
            raise RuntimeError("transport benchmark did not receive valid frames")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run(output, backend):
    from launch import LaunchDescription, LaunchService
    from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler, TimerAction
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch_ros.actions import ComposableNodeContainer
    from launch_ros.descriptions import ComposableNode

    subscriber_env = os.environ.copy()
    if backend == "CPU":
        subscriber_env["CUDA_BUFFER_UID_OVERRIDE"] = "99999"
    observer = ExecuteProcess(
        cmd=[
            sys.executable,
            str(Path(__file__).resolve()),
            "--collect",
            "--backend",
            backend,
            "--output",
            str(output),
        ],
        output="screen",
    )
    subscriber = ComposableNodeContainer(
        name="benchmark_subscriber",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        output="screen",
        env=subscriber_env,
        composable_node_descriptions=[
            ComposableNode(
                package="cuda_buffer_backend",
                plugin="CudaImageSubscriber",
                name="benchmark_sub",
                parameters=[{"expected_backend": "cpu" if backend == "CPU" else "cuda"}],
            )
        ],
    )
    publisher = ComposableNodeContainer(
        name="benchmark_publisher",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                package="cuda_buffer_backend",
                plugin="CudaImagePublisher",
                name="benchmark_pub",
                parameters=[
                    {
                        "max_publish_count": 300,
                        "publish_rate_ms": 50,
                        "image_width": 1920,
                        "image_height": 1080,
                    }
                ],
            )
        ],
    )
    service = LaunchService()
    service.include_launch_description(
        LaunchDescription(
            [
                observer,
                subscriber,
                TimerAction(period=3.0, actions=[publisher]),
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=observer,
                        on_exit=[EmitEvent(event=Shutdown(reason="measurement settled"))],
                    )
                ),
            ]
        )
    )
    service.run()
    if not output.exists() or json.loads(output.read_text())["status"] != "PASS":
        raise RuntimeError("measured transport acceptance failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["CPU", "CUDA"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "201":
        raise ValueError("the owned non-robot acceptance domain ROS_DOMAIN_ID=201 is required")
    if args.collect:
        collect(args.output, args.backend)
    else:
        subprocess.run(["nvidia-smi", "-L"], check=True)
        run(args.output, args.backend)
