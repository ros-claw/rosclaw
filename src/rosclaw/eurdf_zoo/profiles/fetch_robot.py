"""Fetch Mobile Manipulator — Robot Profile Definition.

Mobile base + 7-DOF arm with RGB-D camera and laser scanner.
"""

from rosclaw.runtime.eurdf_loader import (
    RobotEmbodimentProfile,
    RobotSafetyProfile,
    RobotCapabilityProfile,
    RobotSimulationProfile,
    RobotSemanticProfile,
    RobotBenchmarkProfile,
    RobotCompleteProfile,
)

EMBODIMENT = RobotEmbodimentProfile(
    robot_id="fetch_robot",
    name="Fetch",
    vendor="Fetch Robotics",
    version="1.0",
    description="Mobile manipulation platform with differential drive base, torso lift, and 7-DOF arm",
    dof=8,
    links=[
        {"name": "base_link", "type": "base", "mass": 20.0},
        {"name": "torso_lift_link", "type": "link", "mass": 5.0, "description": "Telescoping torso lift"},
        {"name": "shoulder_pan_link", "type": "link", "mass": 2.5},
        {"name": "shoulder_lift_link", "type": "link", "mass": 2.5},
        {"name": "upperarm_roll_link", "type": "link", "mass": 2.0},
        {"name": "elbow_flex_link", "type": "link", "mass": 1.5},
        {"name": "forearm_roll_link", "type": "link", "mass": 1.2},
        {"name": "wrist_flex_link", "type": "link", "mass": 0.8},
        {"name": "wrist_roll_link", "type": "link", "mass": 0.5},
        {"name": "gripper_link", "type": "end_effector_mount", "mass": 0.3},
        {"name": "head_pan_link", "type": "link", "mass": 0.5},
        {"name": "head_tilt_link", "type": "link", "mass": 0.3},
        {"name": "head_camera_link", "type": "link", "mass": 0.2},
    ],
    joints=[
        {"name": "torso_lift_joint", "type": "prismatic", "parent": "base_link", "child": "torso_lift_link", "axis": [0, 0, 1], "limits": {"lower": 0.0, "upper": 0.386, "velocity": 0.1, "effort": 100.0}},
        {"name": "shoulder_pan_joint", "type": "revolute", "parent": "torso_lift_link", "child": "shoulder_pan_link", "axis": [0, 0, 1], "limits": {"lower": -1.6056, "upper": 1.6056, "velocity": 1.256, "effort": 50.0}},
        {"name": "shoulder_lift_joint", "type": "revolute", "parent": "shoulder_pan_link", "child": "shoulder_lift_link", "axis": [0, 1, 0], "limits": {"lower": -1.221, "upper": 1.518, "velocity": 1.256, "effort": 50.0}},
        {"name": "upperarm_roll_joint", "type": "revolute", "parent": "shoulder_lift_link", "child": "upperarm_roll_link", "axis": [1, 0, 0], "limits": {"lower": -3.14159, "upper": 3.14159, "velocity": 1.256, "effort": 30.0}},
        {"name": "elbow_flex_joint", "type": "revolute", "parent": "upperarm_roll_link", "child": "elbow_flex_link", "axis": [0, 1, 0], "limits": {"lower": -2.251, "upper": 2.251, "velocity": 1.256, "effort": 30.0}},
        {"name": "forearm_roll_joint", "type": "revolute", "parent": "elbow_flex_link", "child": "forearm_roll_link", "axis": [1, 0, 0], "limits": {"lower": -3.14159, "upper": 3.14159, "velocity": 1.256, "effort": 20.0}},
        {"name": "wrist_flex_joint", "type": "revolute", "parent": "forearm_roll_link", "child": "wrist_flex_link", "axis": [0, 1, 0], "limits": {"lower": -2.16, "upper": 2.16, "velocity": 1.256, "effort": 15.0}},
        {"name": "wrist_roll_joint", "type": "revolute", "parent": "wrist_flex_link", "child": "wrist_roll_link", "axis": [1, 0, 0], "limits": {"lower": -3.14159, "upper": 3.14159, "velocity": 1.256, "effort": 15.0}},
    ],
    sensors=[
        {"name": "base_laser", "type": "laser", "parent_link": "base_link", "range": [0.1, 25.0], "angle": [-1.57, 1.57], "resolution": 0.25, "frequency": 10},
        {"name": "head_camera_rgbd", "type": "depth_camera", "parent_link": "head_camera_link", "resolution": [640, 480], "fov": 58.0, "fps": 30},
        {"name": "base_imu", "type": "imu", "parent_link": "base_link"},
        {"name": "wheel_encoders", "type": "encoder", "parent_link": "base_link", "resolution": 0.001},
    ],
    actuators=[
        {"name": "torso_lift_motor", "type": "electric_motor", "joint": "torso_lift_joint", "max_current": 10.0, "gear_ratio": 50.0, "control_mode": ["position", "velocity"]},
        {"name": "shoulder_pan_motor", "type": "electric_motor", "joint": "shoulder_pan_joint", "max_current": 8.0, "gear_ratio": 80.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "shoulder_lift_motor", "type": "electric_motor", "joint": "shoulder_lift_joint", "max_current": 8.0, "gear_ratio": 80.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "upperarm_roll_motor", "type": "electric_motor", "joint": "upperarm_roll_joint", "max_current": 5.0, "gear_ratio": 60.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "elbow_flex_motor", "type": "electric_motor", "joint": "elbow_flex_joint", "max_current": 5.0, "gear_ratio": 60.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "forearm_roll_motor", "type": "electric_motor", "joint": "forearm_roll_joint", "max_current": 3.0, "gear_ratio": 40.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "wrist_flex_motor", "type": "electric_motor", "joint": "wrist_flex_joint", "max_current": 2.0, "gear_ratio": 30.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "wrist_roll_motor", "type": "electric_motor", "joint": "wrist_roll_joint", "max_current": 2.0, "gear_ratio": 30.0, "control_mode": ["position", "velocity", "torque"]},
    ],
    metadata={"urdf_version": "1.0", "ros2_package": "fetch_description", "max_base_speed": 1.0, "battery_capacity": "21Ah"},
)

SAFETY = RobotSafetyProfile(
    robot_id="fetch_robot",
    safety_level="MODERATE",
    safety_limits={
        "joint_limits": {"torso_lift_joint": {"lower": 0.0, "upper": 0.386}},
        "velocity_limits": {"torso_lift_joint": 0.1},
        "force_limits": {"max_tcp_force": 100.0, "max_tcp_torque": 5.0, "max_joint_effort": 100.0},
    },
    joint_soft_limits={"torso_lift_joint": {"lower": 0.02, "upper": 0.366, "k_position": 100.0, "k_velocity": 50.0}},
    pfl={"enabled": True, "max_tcp_force": 80.0, "max_tcp_torque": 4.0, "max_static_force": 30.0},
    collision_detection={"method": "force_threshold", "threshold_force": 30.0, "threshold_torque": 2.0, "reaction_time_ms": 5.0, "reaction_strategy": "stop_and_retract"},
    emergency_stop={"category": 2, "stop_time_ms": 300.0, "stop_distance_m": 0.4, "dual_channel": True},
    workspace_boundaries={"type": "enclosure", "fenceless": True},
    interaction={"hand_guiding": False, "speed_slider_max": 0.5, "protective_stop_on_contact": True},
    environment={"ambient_temp_min": 5.0, "ambient_temp_max": 40.0, "humidity_max": 85.0, "ip_rating": "IP20"},
)

CAPABILITY = RobotCapabilityProfile(
    robot_id="fetch_robot",
    capabilities=[
        {"id": "fetch_navigate", "name": "navigate", "category": "navigation", "skill_type": "programmed"},
        {"id": "fetch_pick_and_place", "name": "pick_and_place", "category": "manipulation", "skill_type": "programmed", "constraints": {"max_payload": 2.0, "max_reach": 1.2, "requires_gripper": True}},
        {"id": "fetch_approach_table", "name": "approach_table", "category": "manipulation", "skill_type": "composed"},
        {"id": "fetch_inspect_object", "name": "inspect_object", "category": "perception", "skill_type": "programmed"},
        {"id": "fetch_hand_over", "name": "hand_over", "category": "interaction", "skill_type": "composed", "constraints": {"requires_human_detection": True}},
    ],
    skill_registry={
        "fetch_navigation": {"version": "1.0", "skills": ["navigate", "explore", "dock"]},
        "fetch_manipulation": {"version": "1.0", "skills": ["pick_and_place", "approach_table", "hand_over"]},
    },
    precondition_checks={
        "gripper_attached": {"check": "gripper_link.child_links > 0", "description": "Gripper must be attached"},
        "object_in_reach": {"check": "distance <= 1.2", "description": "Object within reach"},
        "object_mass_leq_2kg": {"check": "object.mass <= 2.0", "description": "Object must not exceed 2kg"},
    },
)

SIMULATION = RobotSimulationProfile(
    robot_id="fetch_robot",
    backends={
        "mujoco": {"model_file": "robot.mjcf.xml", "timestep": 0.002, "integrator": "implicitfast"},
        "isaac": {"model_file": "robot.urdf", "stage_units": "meters", "physics_dt": 0.00833},
        "gazebo": {"model_file": "robot.urdf", "physics_engine": "ode", "max_step_size": 0.001},
    },
)

SEMANTIC = RobotSemanticProfile(
    robot_id="fetch_robot",
    semantic_version="1.0",
    functional_regions=[
        {"name": "mobile_base", "link": "base_link", "tags": ["mobile", "differential_drive", "foundation"], "description": "Differential drive mobile base", "affordances": ["move", "rotate", "navigate"]},
        {"name": "torso", "link": "torso_lift_link", "tags": ["lift", "adjustable_height"], "description": "Telescoping torso lift", "affordances": ["raise", "lower"]},
        {"name": "gripper", "link": "gripper_link", "tags": ["end_effector", "gripper", "manipulation"], "description": "Parallel jaw gripper", "affordances": ["grasp", "release"]},
        {"name": "head", "links": ["head_pan_link", "head_tilt_link"], "tags": ["head", "pan_tilt", "perception"], "description": "Pan-tilt head with RGB-D camera", "affordances": ["look", "scan"]},
    ],
    grasp_points=[],
    visual_features=[
        {"name": "mobile_base_visible", "type": "cylindrical_base", "links": ["base_link"], "visual_cues": ["white_cylinder", "wheels"], "llm_description": "White cylindrical mobile base with two drive wheels"},
        {"name": "arm_visible", "type": "articulated_arm", "links": ["shoulder_pan_link", "shoulder_lift_link", "elbow_flex_link", "wrist_flex_link"], "visual_cues": ["white_tubes", "joints"], "llm_description": "White articulated arm with multiple joints"},
        {"name": "head_visible", "type": "pan_tilt_head", "links": ["head_pan_link", "head_tilt_link", "head_camera_link"], "visual_cues": ["white_head", "camera_lens"], "llm_description": "Pan-tilt head with camera mounted on front"},
    ],
    task_descriptions={
        "pick_and_place": {"relevant_links": ["wrist_flex_link", "wrist_roll_link", "gripper_link"], "workspace_hint": "Objects within 1.2m of robot base", "collision_warning": "Watch base laser scanner when moving"},
        "navigate": {"relevant_links": ["base_link"], "description": "Robot drives to target location", "sensors": ["base_laser"]},
    },
    semantic_tags=["mobile_manipulator", "indoor", "navigation", "manipulation", "7dof_arm", "differential_drive", "research", "service_robot", "american"],
)

BENCHMARK = RobotBenchmarkProfile(
    robot_id="fetch_robot",
    kinematic_benchmarks=[
        {"name": "ik_solve_rate", "description": "IK success rate for 7-DOF arm", "target": 0.99, "timeout_ms": 10.0},
        {"name": "fk_accuracy", "description": "FK accuracy vs CAD", "target": 1.0e-5, "unit": "m"},
    ],
    dynamic_benchmarks=[
        {"name": "trajectory_tracking_error", "description": "RMSE of arm joint tracking", "target": 0.002, "unit": "rad"},
        {"name": "base_odometry_drift", "description": "Position drift over 100m", "target": 0.5, "unit": "m"},
    ],
    simulation_benchmarks={
        "mujoco": [{"name": "real_time_factor", "target": 1.0, "tolerance": 0.05}],
    },
    task_benchmarks=[
        {"name": "office_navigation_10", "description": "Navigate to 10 waypoints", "tasks": ["navigate"], "metrics": [{"name": "success_rate", "target": 0.90}, {"name": "collision_count", "target": 0}]},
        {"name": "tabletop_pick_5", "description": "Pick 5 objects", "tasks": ["pick_and_place"], "metrics": [{"name": "success_rate", "target": 0.90}]},
    ],
    safety_benchmarks=[
        {"name": "emergency_stop_latency", "description": "Time from trigger to full stop", "target": 300.0, "unit": "ms"},
        {"name": "collision_force_at_stop", "description": "Contact force when stop triggers", "target": 80.0, "unit": "N"},
    ],
    baseline_hardware={"cpu": "Intel i7-12700", "gpu": "NVIDIA RTX 3080", "ram": "32GB DDR4", "os": "Ubuntu 22.04", "ros_distro": "humble"},
)

FETCH_ROBOT_PROFILE = RobotCompleteProfile(
    robot_id="fetch_robot",
    name="Fetch",
    vendor="Fetch Robotics",
    version="1.0",
    description="Mobile manipulation platform with differential drive base, torso lift, and 7-DOF arm",
    embodiment=EMBODIMENT,
    safety=SAFETY,
    capability=CAPABILITY,
    simulation=SIMULATION,
    semantic=SEMANTIC,
    benchmark=BENCHMARK,
)
