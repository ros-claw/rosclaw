"""Actual Isaac ROS Resize and multiprocess CUDA IPC in an owned GPU fixture.

Uses upstream ResizeNode and upstream CUDA backend test components. A CPU image
subscriber supplies exact-time CameraInfo and deliberately forces input CPU
fallback. Native output validation also copies pixels to CPU. This is not a
zero-copy graph or an Isaac Sim cleaning acceptance.
"""

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def run(output: Path, shared_cuda_library: Path):
    import rclpy
    from composition_interfaces.srv import LoadNode
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Bool, Float64, UInt32

    if not shared_cuda_library.is_file():
        raise ValueError("the installed shared CUDA allocation library is required")
    rclpy.init()
    node = rclpy.create_node("isaac_resize_acceptance_observer")
    children, logs = [], []
    counts, validity, backends, latency, shapes = [], [], [], [], []

    def load(container, package, plugin, name, params, remaps=()):
        client = node.create_client(LoadNode, f"/{container}/_container/load_node")
        try:
            if not client.wait_for_service(timeout_sec=15):
                raise RuntimeError("component service unavailable")
            request = LoadNode.Request()
            request.package_name, request.plugin_name, request.node_name = package, plugin, name
            request.parameters = [
                Parameter(key, value=value).to_parameter_msg() for key, value in params.items()
            ]
            request.remap_rules = list(remaps)
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=20)
            if not future.done() or future.result() is None or not future.result().success:
                raise RuntimeError("official component failed to load")
        finally:
            node.destroy_client(client)

    def wait_for_frames(target, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(child.poll() is not None for child in children):
                raise RuntimeError("native component process exited")
            if (
                counts
                and max(counts) >= target
                and len(validity) >= target
                and len(backends) >= target
                and len(latency) >= target
                and len(shapes) >= target
            ):
                return
            rclpy.spin_once(node, timeout_sec=0.05)
        raise TimeoutError("complete native validation metrics were not observed")

    result = {"status": "FAIL", "copy_trace_complete": False, "zero_copy_verified": False}
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        # The binary Resize component embeds allocation symbols. Preloading the
        # existing shared implementation binds component and transport plugin
        # to one actual process-wide pool rather than unrelated DSO pools.
        child_env["LD_PRELOAD"] = str(shared_cuda_library)
        for name in ["resize_component", "resize_subscriber", "resize_publisher"]:
            log = output.with_name(output.stem + "-" + name + ".log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [
                        "ros2",
                        "run",
                        "rclcpp_components",
                        "component_container",
                        "--ros-args",
                        "-r",
                        "__node:=" + name,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                )
            )
        load(
            "resize_component",
            "isaac_ros_image_proc",
            "nvidia::isaac_ros::image_proc::ResizeNode",
            "actual_isaac_resize",
            {"output_width": 480, "output_height": 270, "encoding_desired": "rgb8"},
            [
                "image:=/test_cuda_image",
                "camera_info:=/resize_input_camera_info",
                "resize/image:=/test_resized_image",
            ],
        )
        load(
            "resize_subscriber",
            "cuda_buffer_backend",
            "CudaImageSubscriber",
            "actual_resize_cuda_validator",
            {"expected_backend": "cuda"},
            ["test_cuda_image:=/test_resized_image"],
        )
        node.create_subscription(UInt32, "/subscriber_count", lambda m: counts.append(m.data), 1000)
        node.create_subscription(
            Bool, "/validation_result", lambda m: validity.append(m.data), 1000
        )
        node.create_subscription(
            Bool, "/backend_validation", lambda m: backends.append(m.data), 1000
        )
        node.create_subscription(Float64, "/latency_ms", lambda m: latency.append(m.data), 1000)
        node.create_subscription(
            Image,
            "/test_cuda_image_cpu",
            lambda m: shapes.append([m.width, m.height, m.step, m.encoding]),
            1000,
        )
        camera = node.create_publisher(CameraInfo, "/resize_input_camera_info", 10)

        def publish_camera(image):
            info = CameraInfo()
            info.header = image.header
            info.width, info.height = 1920, 1080
            info.distortion_model, info.d = "plumb_bob", [0.0] * 5
            info.k = [1000.0, 0.0, 960.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [1000.0, 0.0, 960.0, 0.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            camera.publish(info)

        node.create_subscription(Image, "/test_cuda_image", publish_camera, 10)
        publisher_params = {"image_width": 1920, "image_height": 1080}
        # Validate one cold-start frame before the 20Hz episode. Cold GPU setup
        # can exceed the upstream IPC pool's recycling interval; no lifetime
        # threshold is increased and no missing/corrupt frame is accepted.
        load(
            "resize_publisher",
            "cuda_buffer_backend",
            "CudaImagePublisher",
            "warmup_publisher",
            {**publisher_params, "max_publish_count": 1, "publish_rate_ms": 1000},
        )
        wait_for_frames(1, 15)
        if not all(validity + backends) or shapes != [[480, 270, 1440, "rgb8"]]:
            raise RuntimeError("cold-start graph validation failed")
        load(
            "resize_publisher",
            "cuda_buffer_backend",
            "CudaImagePublisher",
            "actual_gpu_publisher",
            {**publisher_params, "max_publish_count": 300, "publish_rate_ms": 50},
        )
        wait_for_frames(301, 35)
        if not all(validity + backends) or any(s != [480, 270, 1440, "rgb8"] for s in shapes):
            raise RuntimeError("native output content/backend/geometry validation failed")
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        # Cleanup precedes evidence serialization, including failed starts.
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()
        node.destroy_node()
        rclpy.shutdown()
        result.update(
            {
                "captured_at": datetime.now(UTC).isoformat(),
                "evidence_domain": "GPU_FIXTURE",
                "input_dimensions": [1920, 1080],
                "observed_output_shapes": shapes,
                "publish_period_ms": 50,
                "published_target": 300,
                "warmup_validated_frames": 1,
                "received_validated_frames": max(0, max(counts, default=0) - 1),
                "content_and_backend_validated": bool(validity and backends)
                and all(validity + backends),
                "output_buffer_backend": "CUDA" if backends and all(backends) else "UNKNOWN",
                "component": "nvidia::isaac_ros::image_proc::ResizeNode",
                "publisher_processor_subscriber_separate_processes": True,
                "shared_cuda_pool_preload": str(shared_cuda_library),
                "latency_samples_ms": latency,
                "hardware_motion": False,
                "input_metadata_observer_forces_cpu_fallback": True,
                "notes": "Exact-time metadata uses the real input header. Input CPU fallback and validator DtoH copies are explicit; no complete copy profiler trace exists.",
            }
        )
        output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shared-cuda-library", type=Path, default=Path("/opt/ros/lyrical/lib/libcuda_buffer.so")
    )
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "201":
        raise ValueError("the owned non-robot domain ROS_DOMAIN_ID=201 is required")
    run(args.output, args.shared_cuda_library)
