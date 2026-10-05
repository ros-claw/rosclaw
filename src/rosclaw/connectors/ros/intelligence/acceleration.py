"""Source-backed Isaac ROS 5 options, separate from execution authority.

Hardware presence does not justify replacing a working CPU navigation stack.
Compatibility checks describe candidates; live backend/copy measurements still
come from the performance observer and a measured before/after benchmark.
"""

from rosclaw.connectors.ros.intelligence.system_model import RosSystemModel

ISAAC_ROS_5_SOURCES = {
    "platform": "https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/index.html",
    "buffer": "https://nvidia-isaac-ros.github.io/concepts/rosidl_buffer/cuda_buffer_backend.html",
    "migration": "https://nvidia-isaac-ros.github.io/concepts/rosidl_buffer/nitros_migration.html",
}


def acceleration_options(model: RosSystemModel) -> list[dict]:
    checks = {
        "ros2": model.environment.get("ros_generation") == "ros2",
        "ros2_lyrical": model.environment.get("distro") == "lyrical",
        # Graph names, host architecture and package installation are not
        # sufficient evidence of a usable accelerator in this ROS environment.
        "gpu_runtime": None,
        "supported_platform": None,
        "sensor_calibration": None,
        "measured_workload_bottleneck": None,
    }
    status = "BLOCKED" if False in checks.values() else "UNKNOWN"
    return [
        {
            "id": name,
            "release": "5.0",
            "status": status,
            "requirements": checks,
            "workload": workload,
            "automatic_install": False,
            "optimization_verified": False,
            "execution_entry": "request_action",
            "official_sources": ISAAC_ROS_5_SOURCES,
            "validation_probes": [
                "backend_type",
                "separate_process_transport",
                "cpu_fallback",
                "buffer_lifetime",
                "memory_copy",
                "before_after_benchmark",
            ],
        }
        for name, workload in [
            ("isaac_ros_cuvslam", "calibrated visual odometry and localization"),
            ("isaac_ros_nvblox", "depth-based 3D reconstruction and navigation costmaps"),
            ("isaac_ros_tensor_rt", "model-specific neural inference"),
            ("rosidl_buffer_cuda", "same-host compatible CUDA buffer transport"),
        ]
    ]
