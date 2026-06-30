"""Intel RealSense D405 — sensor robot profile.

Short-range RGB-D camera imported from realsense-ros.
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
    robot_id="realsense_d405",
    name="Intel RealSense D405",
    vendor="Intel RealSense",
    version="1.0",
    description="Short-range RGB-D camera with nominal extrinsics from realsense-ros.",
    dof=0,
    links=[
        {"name": "realsense_d405_mount", "type": "base", "mass": 0.0},
        {"name": "camera_bottom_screw_frame", "type": "link", "mass": 0.0},
        {"name": "camera_link", "type": "link", "mass": 0.05},
        {"name": "camera_depth_frame", "type": "link", "mass": 0.0},
        {"name": "camera_depth_optical_frame", "type": "optical_frame", "mass": 0.0},
        {"name": "camera_infra1_frame", "type": "link", "mass": 0.0},
        {"name": "camera_infra1_optical_frame", "type": "optical_frame", "mass": 0.0},
        {"name": "camera_infra2_frame", "type": "link", "mass": 0.0},
        {"name": "camera_infra2_optical_frame", "type": "optical_frame", "mass": 0.0},
        {"name": "camera_color_frame", "type": "link", "mass": 0.0},
        {"name": "camera_color_optical_frame", "type": "optical_frame", "mass": 0.0},
    ],
    joints=[
        {"name": "camera_joint", "type": "fixed", "parent": "realsense_d405_mount", "child": "camera_bottom_screw_frame"},
        {"name": "camera_link_joint", "type": "fixed", "parent": "camera_bottom_screw_frame", "child": "camera_link"},
        {"name": "camera_depth_joint", "type": "fixed", "parent": "camera_link", "child": "camera_depth_frame"},
        {"name": "camera_depth_optical_joint", "type": "fixed", "parent": "camera_depth_frame", "child": "camera_depth_optical_frame"},
        {"name": "camera_infra1_joint", "type": "fixed", "parent": "camera_link", "child": "camera_infra1_frame"},
        {"name": "camera_infra1_optical_joint", "type": "fixed", "parent": "camera_infra1_frame", "child": "camera_infra1_optical_frame"},
        {"name": "camera_infra2_joint", "type": "fixed", "parent": "camera_link", "child": "camera_infra2_frame"},
        {"name": "camera_infra2_optical_joint", "type": "fixed", "parent": "camera_infra2_frame", "child": "camera_infra2_optical_frame"},
        {"name": "camera_color_joint", "type": "fixed", "parent": "camera_link", "child": "camera_color_frame"},
        {"name": "camera_color_optical_joint", "type": "fixed", "parent": "camera_color_frame", "child": "camera_color_optical_frame"},
    ],
    sensors=[
        {"name": "color_camera", "type": "camera", "parent_link": "camera_color_optical_frame", "resolution": [1280, 720], "fov": 90.0, "fps": 30},
        {"name": "depth_camera", "type": "depth_camera", "parent_link": "camera_depth_optical_frame", "resolution": [1280, 720], "fov": 90.0, "fps": 30, "min_range": 0.05, "max_range": 6.0},
        {"name": "infrared_left", "type": "camera", "parent_link": "camera_infra1_optical_frame", "resolution": [1280, 720], "fov": 90.0, "fps": 30},
        {"name": "infrared_right", "type": "camera", "parent_link": "camera_infra2_optical_frame", "resolution": [1280, 720], "fov": 90.0, "fps": 30},
    ],
    actuators=[],
    metadata={
        "category": "rgbd_camera",
        "urdf_version": "1.0",
        "ros2_package": "realsense2_description",
        "asset_id": "sensors/realsense/d405/default",
    },
)

SAFETY = RobotSafetyProfile(
    robot_id="realsense_d405",
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
    workspace_boundaries={"type": "bounding_box", "center": [0, 0, 0], "dimensions": [0.1, 0.1, 0.1]},
    interaction={"hand_guiding": False, "human_collaboration": False},
    environment={"real_robot_execution_allowed": False, "sandbox_required": True, "perception_only": True, "no_actuation": True},
)

CAPABILITY = RobotCapabilityProfile(
    robot_id="realsense_d405",
    capabilities=[
        {"id": "rgb_camera", "name": "rgb_observation", "category": "perception", "skill_type": "programmed", "risk": "low"},
        {"id": "depth_camera", "name": "depth_observation", "category": "perception", "skill_type": "programmed", "risk": "medium", "calibration_required": True},
        {"id": "stereo_infrared", "name": "infrared_observation", "category": "perception", "skill_type": "programmed", "risk": "medium"},
    ],
    skill_registry={
        "forbidden_capabilities": [
            {"id": "depth_collision_avoidance_without_calibration", "reason": "uncalibrated depth safety risk", "severity": "critical"},
            {"id": "visual_servo_without_hand_eye_calibration", "reason": "unverified camera-to-end-effector transform", "severity": "critical"},
        ],
    },
    precondition_checks={
        "camera_info_available": {"check": "camera_info topic publishing", "description": "Camera intrinsics must be available"},
        "tf_available": {"check": "camera frame in tf tree", "description": "Camera extrinsics must be resolved"},
    },
)

SIMULATION = RobotSimulationProfile(
    robot_id="realsense_d405",
    backends={
        "mujoco": {"model_file": "model/model_mujoco.urdf", "status": "experimental"},
        "ros2": {"model_file": "model/model.urdf", "status": "validated"},
        "rviz": {"model_file": "model/model.urdf", "status": "validated"},
    },
)

SEMANTIC = RobotSemanticProfile(
    robot_id="realsense_d405",
    semantic_version="1.0",
    functional_regions=[
        {"name": "mount", "link": "realsense_d405_mount", "tags": ["mount", "base", "attachment"], "description": "Mechanical mount to robot body", "affordances": ["attach"]},
        {"name": "camera_body", "link": "camera_link", "tags": ["camera_body", "sensor_housing"], "description": "Main camera housing"},
        {"name": "color_optical", "link": "camera_color_optical_frame", "tags": ["color_sensor", "optical_frame"], "description": "Color image optical center"},
        {"name": "depth_optical", "link": "camera_depth_optical_frame", "tags": ["depth_sensor", "optical_frame"], "description": "Depth image optical center"},
    ],
    grasp_points=[],
    visual_features=[
        {"name": "camera_housing", "type": "rectangular_sensor", "links": ["camera_link"], "visual_cues": ["black_rectangular_housing"], "llm_description": "Small rectangular RGB-D camera housing"},
        {"name": "lens_assembly", "type": "lens_cluster", "links": ["camera_color_frame", "camera_depth_frame"], "visual_cues": ["lenses", "infrared_projector"], "llm_description": "Cluster of color/depth lenses on the camera face"},
    ],
    task_descriptions={
        "rgb_observation": {"relevant_links": ["camera_color_optical_frame"], "description": "Capture color images"},
        "depth_observation": {"relevant_links": ["camera_depth_optical_frame"], "description": "Capture depth images", "safety": "Requires calibration for metric use"},
    },
    semantic_tags=["rgbd_camera", "depth_camera", "sensor", "perception", "intel_realsense", "experimental"],
)

BENCHMARK = RobotBenchmarkProfile(
    robot_id="realsense_d405",
    kinematic_benchmarks=[],
    dynamic_benchmarks=[],
    simulation_benchmarks={},
    task_benchmarks=[],
    safety_benchmarks=[],
    baseline_hardware={},
)

REALSENSE_D405_PROFILE = RobotCompleteProfile(
    robot_id="realsense_d405",
    name="Intel RealSense D405",
    vendor="Intel RealSense",
    version="1.0",
    description="Short-range RGB-D camera with nominal extrinsics from realsense-ros.",
    identity={"robot_class": "perception_only_camera"},
    embodiment=EMBODIMENT,
    safety=SAFETY,
    capability=CAPABILITY,
    simulation=SIMULATION,
    semantic=SEMANTIC,
    benchmark=BENCHMARK,
)
