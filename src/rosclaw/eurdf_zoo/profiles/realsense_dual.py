"""Dual RealSense sensor robot profile.

Example composite body with a head-mounted D405 and a wrist-mounted D435i.
This is a perception-only configuration with no actuators.
"""

from rosclaw.runtime.eurdf_loader import (
    RobotBenchmarkProfile,
    RobotCapabilityProfile,
    RobotCompleteProfile,
    RobotEmbodimentProfile,
    RobotSafetyProfile,
    RobotSemanticProfile,
    RobotSimulationProfile,
)

EMBODIMENT = RobotEmbodimentProfile(
    robot_id="realsense_dual",
    name="Dual RealSense D405 + D435i",
    vendor="Intel RealSense",
    version="1.0",
    description="Composite perception body with head RGB-D (D405) and wrist RGB-D+IMU (D435i).",
    dof=0,
    links=[
        {"name": "base_link", "type": "base", "mass": 0.0},
        {"name": "head_mount", "type": "link", "mass": 0.0},
        {"name": "head_realsense_d405_mount", "type": "link", "mass": 0.0},
        {"name": "head_camera_link", "type": "link", "mass": 0.05},
        {"name": "head_camera_depth_optical_frame", "type": "optical_frame", "mass": 0.0},
        {"name": "head_camera_color_optical_frame", "type": "optical_frame", "mass": 0.0},
        {"name": "wrist_mount", "type": "link", "mass": 0.0},
        {"name": "wrist_realsense_d435i_mount", "type": "link", "mass": 0.0},
        {"name": "wrist_camera_link", "type": "link", "mass": 0.05},
        {"name": "wrist_camera_depth_optical_frame", "type": "optical_frame", "mass": 0.0},
        {"name": "wrist_camera_color_optical_frame", "type": "optical_frame", "mass": 0.0},
        {"name": "wrist_camera_accel_frame", "type": "link", "mass": 0.0},
        {"name": "wrist_camera_gyro_frame", "type": "link", "mass": 0.0},
    ],
    joints=[
        {"name": "head_mount_joint", "type": "fixed", "parent": "base_link", "child": "head_mount"},
        {"name": "head_camera_joint", "type": "fixed", "parent": "head_mount", "child": "head_realsense_d405_mount"},
        {"name": "head_camera_optical_joint", "type": "fixed", "parent": "head_realsense_d405_mount", "child": "head_camera_link"},
        {"name": "head_depth_optical_joint", "type": "fixed", "parent": "head_camera_link", "child": "head_camera_depth_optical_frame"},
        {"name": "head_color_optical_joint", "type": "fixed", "parent": "head_camera_link", "child": "head_camera_color_optical_frame"},
        {"name": "wrist_mount_joint", "type": "fixed", "parent": "base_link", "child": "wrist_mount"},
        {"name": "wrist_camera_joint", "type": "fixed", "parent": "wrist_mount", "child": "wrist_realsense_d435i_mount"},
        {"name": "wrist_camera_optical_joint", "type": "fixed", "parent": "wrist_realsense_d435i_mount", "child": "wrist_camera_link"},
        {"name": "wrist_depth_optical_joint", "type": "fixed", "parent": "wrist_camera_link", "child": "wrist_camera_depth_optical_frame"},
        {"name": "wrist_color_optical_joint", "type": "fixed", "parent": "wrist_camera_link", "child": "wrist_camera_color_optical_frame"},
        {"name": "wrist_accel_joint", "type": "fixed", "parent": "wrist_camera_link", "child": "wrist_camera_accel_frame"},
        {"name": "wrist_gyro_joint", "type": "fixed", "parent": "wrist_camera_link", "child": "wrist_camera_gyro_frame"},
    ],
    sensors=[
        {"name": "head_color_camera", "type": "camera", "parent_link": "head_camera_color_optical_frame", "resolution": [1280, 720], "fov": 90.0, "fps": 30},
        {"name": "head_depth_camera", "type": "depth_camera", "parent_link": "head_camera_depth_optical_frame", "resolution": [1280, 720], "fov": 90.0, "fps": 30, "min_range": 0.05, "max_range": 6.0},
        {"name": "wrist_color_camera", "type": "camera", "parent_link": "wrist_camera_color_optical_frame", "resolution": [1280, 720], "fov": 90.0, "fps": 30},
        {"name": "wrist_depth_camera", "type": "depth_camera", "parent_link": "wrist_camera_depth_optical_frame", "resolution": [1280, 720], "fov": 90.0, "fps": 30, "min_range": 0.1, "max_range": 10.0},
        {"name": "wrist_imu", "type": "imu", "parent_link": "wrist_camera_link", "noise_std": {"accel": 0.01, "gyro": 0.001}},
    ],
    actuators=[],
    metadata={
        "category": "rgbd_camera",
        "urdf_version": "1.0",
        "ros2_package": "realsense2_description",
        "composition": ["sensors/realsense/d405/default", "sensors/realsense/d435i/default"],
    },
)

SAFETY = RobotSafetyProfile(
    robot_id="realsense_dual",
    safety_level="STRICT",
    safety_limits={
        "perception_only": True,
        "depth_for_safety_requires_calibration": True,
        "visual_servo_requires_hand_eye_calibration": True,
    },
    joint_soft_limits={},
    pfl={"enabled": False},
    collision_detection={"method": "none"},
    emergency_stop={"enabled": False},
    workspace_boundaries={"type": "bounding_box", "center": [0, 0, 0], "dimensions": [0.5, 0.5, 0.5]},
    interaction={"hand_guiding": False, "human_collaboration": False},
    environment={"real_robot_execution_allowed": False, "sandbox_required": True, "perception_only": True, "no_actuation": True},
)

CAPABILITY = RobotCapabilityProfile(
    robot_id="realsense_dual",
    capabilities=[
        {"id": "rgb_camera", "name": "rgb_observation", "category": "perception", "skill_type": "programmed", "risk": "low"},
        {"id": "depth_camera", "name": "depth_observation", "category": "perception", "skill_type": "programmed", "risk": "medium", "calibration_required": True},
        {"id": "stereo_infrared", "name": "infrared_observation", "category": "perception", "skill_type": "programmed", "risk": "medium"},
        {"id": "imu", "name": "imu_observation", "category": "perception", "skill_type": "programmed", "risk": "low"},
        {"id": "dual_rgbd", "name": "dual_rgbd_observation", "category": "perception", "skill_type": "composed", "risk": "medium", "calibration_required": True},
        {"id": "hand_eye_visual_servo", "name": "hand_eye_visual_servo", "category": "manipulation", "skill_type": "composed", "risk": "high", "calibration_required": True, "sandbox_required": True},
    ],
    skill_registry={
        "forbidden_capabilities": [
            {"id": "depth_collision_avoidance_without_calibration", "reason": "uncalibrated depth safety risk", "severity": "critical"},
            {"id": "visual_servo_without_hand_eye_calibration", "reason": "unverified camera-to-end-effector transform", "severity": "critical"},
        ],
    },
    precondition_checks={
        "camera_info_available": {"check": "camera_info topics publishing", "description": "Camera intrinsics must be available"},
        "tf_available": {"check": "camera frames in tf tree", "description": "Camera extrinsics must be resolved"},
    },
)

SIMULATION = RobotSimulationProfile(
    robot_id="realsense_dual",
    backends={
        "mujoco": {"model_file": "model/model_mujoco.urdf", "status": "experimental"},
        "ros2": {"model_file": "model/model.urdf", "status": "validated"},
        "rviz": {"model_file": "model/model.urdf", "status": "validated"},
    },
)

SEMANTIC = RobotSemanticProfile(
    robot_id="realsense_dual",
    semantic_version="1.0",
    functional_regions=[
        {"name": "base", "link": "base_link", "tags": ["base", "foundation"], "description": "Common reference frame", "affordances": ["attach"]},
        {"name": "head_camera", "link": "head_camera_link", "tags": ["head", "rgbd_camera"], "description": "Head-mounted short-range RGB-D camera"},
        {"name": "wrist_camera", "link": "wrist_camera_link", "tags": ["wrist", "rgbd_camera", "imu"], "description": "Wrist-mounted RGB-D camera with IMU"},
    ],
    grasp_points=[],
    visual_features=[
        {"name": "head_sensor", "type": "rectangular_sensor", "links": ["head_camera_link"], "visual_cues": ["black_rectangular_housing"], "llm_description": "Head-mounted RealSense D405"},
        {"name": "wrist_sensor", "type": "rectangular_sensor", "links": ["wrist_camera_link"], "visual_cues": ["black_rectangular_housing"], "llm_description": "Wrist-mounted RealSense D435i with IMU"},
    ],
    task_descriptions={
        "head_rgb_observation": {"relevant_links": ["head_camera_color_optical_frame"], "description": "Capture head color images"},
        "wrist_depth_observation": {"relevant_links": ["wrist_camera_depth_optical_frame"], "description": "Capture wrist depth images", "safety": "Requires hand-eye calibration"},
    },
    semantic_tags=["rgbd_camera", "depth_camera", "sensor", "perception", "intel_realsense", "dual_camera", "hand_eye", "experimental"],
)

BENCHMARK = RobotBenchmarkProfile(
    robot_id="realsense_dual",
    kinematic_benchmarks=[],
    dynamic_benchmarks=[],
    simulation_benchmarks={},
    task_benchmarks=[],
    safety_benchmarks=[],
    baseline_hardware={},
)

REALSENSE_DUAL_PROFILE = RobotCompleteProfile(
    robot_id="realsense_dual",
    name="Dual RealSense D405 + D435i",
    vendor="Intel RealSense",
    version="1.0",
    description="Composite perception body with head RGB-D (D405) and wrist RGB-D+IMU (D435i).",
    identity={"robot_class": "perception_only_camera"},
    embodiment=EMBODIMENT,
    safety=SAFETY,
    capability=CAPABILITY,
    simulation=SIMULATION,
    semantic=SEMANTIC,
    benchmark=BENCHMARK,
)
