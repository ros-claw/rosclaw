"""Unitree Go2 Quadruped — Robot Profile Definition.

Agile 12-DOF quadruped robot with LIDAR and depth cameras.
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
    robot_id="unitree_go2",
    name="Go2",
    vendor="Unitree",
    version="1.1",
    description="Agile 12-DOF quadruped with LIDAR, depth cameras, and onboard computing",
    dof=12,
    links=[
        {"name": "base", "type": "base", "mass": 8.0},
        {"name": "trunk", "type": "link", "mass": 6.0},
        {"name": "fr_hip", "type": "link", "mass": 0.5},
        {"name": "fr_thigh", "type": "link", "mass": 1.0},
        {"name": "fr_calf", "type": "link", "mass": 0.8},
        {"name": "fr_foot", "type": "end_effector", "mass": 0.1},
        {"name": "fl_hip", "type": "link", "mass": 0.5},
        {"name": "fl_thigh", "type": "link", "mass": 1.0},
        {"name": "fl_calf", "type": "link", "mass": 0.8},
        {"name": "fl_foot", "type": "end_effector", "mass": 0.1},
        {"name": "rr_hip", "type": "link", "mass": 0.5},
        {"name": "rr_thigh", "type": "link", "mass": 1.0},
        {"name": "rr_calf", "type": "link", "mass": 0.8},
        {"name": "rr_foot", "type": "end_effector", "mass": 0.1},
        {"name": "rl_hip", "type": "link", "mass": 0.5},
        {"name": "rl_thigh", "type": "link", "mass": 1.0},
        {"name": "rl_calf", "type": "link", "mass": 0.8},
        {"name": "rl_foot", "type": "end_effector", "mass": 0.1},
    ],
    joints=[
        {"name": "fr_hip_joint", "type": "revolute", "parent": "trunk", "child": "fr_hip", "axis": [1, 0, 0], "limits": {"lower": -0.8, "upper": 0.8, "velocity": 20.0, "effort": 23.7}},
        {"name": "fr_thigh_joint", "type": "revolute", "parent": "fr_hip", "child": "fr_thigh", "axis": [0, 1, 0], "limits": {"lower": -1.05, "upper": 4.19, "velocity": 20.0, "effort": 23.7}},
        {"name": "fr_calf_joint", "type": "revolute", "parent": "fr_thigh", "child": "fr_calf", "axis": [0, 1, 0], "limits": {"lower": -2.7, "upper": -0.92, "velocity": 20.0, "effort": 45.0}},
        {"name": "fl_hip_joint", "type": "revolute", "parent": "trunk", "child": "fl_hip", "axis": [1, 0, 0], "limits": {"lower": -0.8, "upper": 0.8, "velocity": 20.0, "effort": 23.7}},
        {"name": "fl_thigh_joint", "type": "revolute", "parent": "fl_hip", "child": "fl_thigh", "axis": [0, 1, 0], "limits": {"lower": -1.05, "upper": 4.19, "velocity": 20.0, "effort": 23.7}},
        {"name": "fl_calf_joint", "type": "revolute", "parent": "fl_thigh", "child": "fl_calf", "axis": [0, 1, 0], "limits": {"lower": -2.7, "upper": -0.92, "velocity": 20.0, "effort": 45.0}},
        {"name": "rr_hip_joint", "type": "revolute", "parent": "trunk", "child": "rr_hip", "axis": [1, 0, 0], "limits": {"lower": -0.8, "upper": 0.8, "velocity": 20.0, "effort": 23.7}},
        {"name": "rr_thigh_joint", "type": "revolute", "parent": "rr_hip", "child": "rr_thigh", "axis": [0, 1, 0], "limits": {"lower": -1.05, "upper": 4.19, "velocity": 20.0, "effort": 23.7}},
        {"name": "rr_calf_joint", "type": "revolute", "parent": "rr_thigh", "child": "rr_calf", "axis": [0, 1, 0], "limits": {"lower": -2.7, "upper": -0.92, "velocity": 20.0, "effort": 45.0}},
        {"name": "rl_hip_joint", "type": "revolute", "parent": "trunk", "child": "rl_hip", "axis": [1, 0, 0], "limits": {"lower": -0.8, "upper": 0.8, "velocity": 20.0, "effort": 23.7}},
        {"name": "rl_thigh_joint", "type": "revolute", "parent": "rl_hip", "child": "rl_thigh", "axis": [0, 1, 0], "limits": {"lower": -1.05, "upper": 4.19, "velocity": 20.0, "effort": 23.7}},
        {"name": "rl_calf_joint", "type": "revolute", "parent": "rl_thigh", "child": "rl_calf", "axis": [0, 1, 0], "limits": {"lower": -2.7, "upper": -0.92, "velocity": 20.0, "effort": 45.0}},
    ],
    sensors=[
        {"name": "lidar", "type": "lidar", "parent_link": "trunk", "range": [0.1, 30.0], "resolution": 0.25, "frequency": 10},
        {"name": "front_camera", "type": "camera", "parent_link": "trunk", "resolution": [1920, 1080], "fov": 120.0, "fps": 30},
        {"name": "depth_camera", "type": "depth_camera", "parent_link": "trunk", "resolution": [640, 480], "fov": 87.0, "fps": 30},
        {"name": "trunk_imu", "type": "imu", "parent_link": "trunk"},
        {"name": "foot_force_sensors", "type": "force", "parent_link": "*foot", "range": [0, 500], "resolution": 0.1},
    ],
    actuators=[
        {"name": "fr_hip_actuator", "type": "electric_motor", "joint": "fr_hip_joint", "max_current": 15.0, "gear_ratio": 7.75, "control_mode": ["position", "velocity", "torque"]},
        {"name": "fr_thigh_actuator", "type": "electric_motor", "joint": "fr_thigh_joint", "max_current": 15.0, "gear_ratio": 7.75, "control_mode": ["position", "velocity", "torque"]},
        {"name": "fr_calf_actuator", "type": "electric_motor", "joint": "fr_calf_joint", "max_current": 20.0, "gear_ratio": 16.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "fl_hip_actuator", "type": "electric_motor", "joint": "fl_hip_joint", "max_current": 15.0, "gear_ratio": 7.75, "control_mode": ["position", "velocity", "torque"]},
        {"name": "fl_thigh_actuator", "type": "electric_motor", "joint": "fl_thigh_joint", "max_current": 15.0, "gear_ratio": 7.75, "control_mode": ["position", "velocity", "torque"]},
        {"name": "fl_calf_actuator", "type": "electric_motor", "joint": "fl_calf_joint", "max_current": 20.0, "gear_ratio": 16.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "rr_hip_actuator", "type": "electric_motor", "joint": "rr_hip_joint", "max_current": 15.0, "gear_ratio": 7.75, "control_mode": ["position", "velocity", "torque"]},
        {"name": "rr_thigh_actuator", "type": "electric_motor", "joint": "rr_thigh_joint", "max_current": 15.0, "gear_ratio": 7.75, "control_mode": ["position", "velocity", "torque"]},
        {"name": "rr_calf_actuator", "type": "electric_motor", "joint": "rr_calf_joint", "max_current": 20.0, "gear_ratio": 16.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "rl_hip_actuator", "type": "electric_motor", "joint": "rl_hip_joint", "max_current": 15.0, "gear_ratio": 7.75, "control_mode": ["position", "velocity", "torque"]},
        {"name": "rl_thigh_actuator", "type": "electric_motor", "joint": "rl_thigh_joint", "max_current": 15.0, "gear_ratio": 7.75, "control_mode": ["position", "velocity", "torque"]},
        {"name": "rl_calf_actuator", "type": "electric_motor", "joint": "rl_calf_joint", "max_current": 20.0, "gear_ratio": 16.0, "control_mode": ["position", "velocity", "torque"]},
    ],
    metadata={"urdf_version": "1.0", "ros2_package": "unitree_go2_description", "payload_capacity": 3.0, "max_speed": 3.5},
)

SAFETY = RobotSafetyProfile(
    robot_id="unitree_go2",
    safety_level="MODERATE",
    safety_limits={
        "joint_limits": {"fr_hip_joint": {"lower": -0.8, "upper": 0.8}},
        "velocity_limits": {"fr_hip_joint": 20.0},
        "force_limits": {"max_joint_effort": 45.0, "max_foot_force": 500.0},
    },
    joint_soft_limits={"fr_hip_joint": {"lower": -0.75, "upper": 0.75, "k_position": 100.0, "k_velocity": 50.0}},
    pfl={"enabled": True, "max_foot_force": 400.0, "max_body_acceleration": 2.0, "max_fall_height": 0.3},
    collision_detection={"method": "foot_force_threshold", "threshold_force": 100.0, "reaction_time_ms": 10.0, "reaction_strategy": "adjust_gait"},
    emergency_stop={"category": 2, "stop_time_ms": 500.0, "stop_distance_m": 0.5, "dual_channel": False},
    workspace_boundaries={"type": "free", "fenceless": True},
    interaction={"speed_slider_max": 1.0, "protective_stop_on_contact": True},
    environment={"ambient_temp_min": -10.0, "ambient_temp_max": 45.0, "humidity_max": 90.0, "ip_rating": "IP66"},
)

CAPABILITY = RobotCapabilityProfile(
    robot_id="unitree_go2",
    capabilities=[
        {"id": "go2_walk", "name": "walk", "category": "locomotion", "skill_type": "programmed", "constraints": {"max_slope": 30.0, "max_step_height": 0.15}},
        {"id": "go2_trot", "name": "trot", "category": "locomotion", "skill_type": "programmed", "constraints": {"max_speed": 3.5}},
        {"id": "go2_stand_up", "name": "stand_up", "category": "locomotion", "skill_type": "programmed"},
        {"id": "go2_stair_climb", "name": "stair_climb", "category": "locomotion", "skill_type": "composed", "constraints": {"max_step_height": 0.18, "max_slope": 40.0}},
        {"id": "go2_navigation", "name": "navigation", "category": "perception", "skill_type": "programmed"},
    ],
    skill_registry={
        "go2_locomotion": {"version": "1.0", "skills": ["walk", "trot", "stand_up", "stair_climb"]},
        "go2_navigation": {"version": "1.0", "skills": ["navigation", "explore"]},
    },
    precondition_checks={
        "robot_standing": {"check": "all(foot.z > 0.05)", "description": "All feet above ground"},
        "terrain_traversable": {"check": "max_slope <= 30.0", "description": "Terrain within limits"},
    },
)

SIMULATION = RobotSimulationProfile(
    robot_id="unitree_go2",
    backends={
        "mujoco": {"model_file": "robot.mjcf.xml", "timestep": 0.001, "integrator": "implicitfast"},
        "isaac": {"model_file": "robot.urdf", "stage_units": "meters", "physics_dt": 0.00833, "gpu_physics": True},
        "gazebo": {"model_file": "robot.urdf", "physics_engine": "ode", "max_step_size": 0.001},
    },
)

SEMANTIC = RobotSemanticProfile(
    robot_id="unitree_go2",
    semantic_version="1.0",
    functional_regions=[
        {"name": "trunk", "link": "trunk", "tags": ["body", "main_chassis", "central"], "description": "Main body of quadruped", "affordances": ["carry_payload"]},
        {"name": "front_right_leg", "links": ["fr_hip", "fr_thigh", "fr_calf"], "tags": ["leg", "front", "right", "3dof"], "description": "Front right leg", "affordances": ["walk", "support"]},
        {"name": "front_left_leg", "links": ["fl_hip", "fl_thigh", "fl_calf"], "tags": ["leg", "front", "left", "3dof"], "description": "Front left leg", "affordances": ["walk", "support"]},
        {"name": "rear_right_leg", "links": ["rr_hip", "rr_thigh", "rr_calf"], "tags": ["leg", "rear", "right", "3dof"], "description": "Rear right leg", "affordances": ["walk", "support"]},
        {"name": "rear_left_leg", "links": ["rl_hip", "rl_thigh", "rl_calf"], "tags": ["leg", "rear", "left", "3dof"], "description": "Rear left leg", "affordances": ["walk", "support"]},
        {"name": "head", "link": "trunk", "tags": ["sensor_mount", "perception"], "description": "Sensor head with cameras and LIDAR", "affordances": ["observe", "scan"]},
    ],
    grasp_points=[],
    visual_features=[
        {"name": "trunk_body", "type": "rectangular_body", "links": ["trunk"], "visual_cues": ["black_rectangular", "compact"], "llm_description": "Compact rectangular body of quadruped"},
        {"name": "leg_structure", "type": "articulated_leg", "links": ["fr_hip", "fr_thigh", "fr_calf"], "visual_cues": ["three_segments", "jointed"], "llm_description": "Articulated leg with three segments"},
    ],
    task_descriptions={
        "walk": {"relevant_links": ["fr_foot", "fl_foot", "rr_foot", "rl_foot"], "description": "Robot walks using alternating leg movements", "workspace_hint": "Requires flat or mildly sloped terrain"},
        "navigation": {"relevant_links": ["trunk"], "description": "Robot navigates to target while avoiding obstacles", "sensors": ["lidar", "depth_camera"]},
    },
    semantic_tags=["quadruped", "legged_robot", "12dof", "locomotion", "outdoor", "agile", "inspection", "research", "chinese"],
)

BENCHMARK = RobotBenchmarkProfile(
    robot_id="unitree_go2",
    kinematic_benchmarks=[
        {"name": "gait_stability", "description": "Stability margin during walking", "target": 0.05, "unit": "m"},
        {"name": "foot_placement_accuracy", "description": "Accuracy of foot placement", "target": 0.01, "unit": "m"},
    ],
    dynamic_benchmarks=[
        {"name": "walking_speed", "description": "Max stable walking speed", "target": 0.5, "unit": "m/s"},
        {"name": "trotting_speed", "description": "Max stable trotting speed", "target": 3.5, "unit": "m/s"},
        {"name": "power_efficiency", "description": "Power per meter", "target": 50.0, "unit": "W/(m/s)"},
    ],
    simulation_benchmarks={
        "mujoco": [{"name": "real_time_factor", "target": 1.0, "tolerance": 0.05}],
        "isaac": [{"name": "real_time_factor", "target": 1.0, "tolerance": 0.1}],
    },
    task_benchmarks=[
        {"name": "flat_terrain_walk_100m", "description": "Walk 100m on flat", "tasks": ["walk"], "metrics": [{"name": "success_rate", "target": 0.98}, {"name": "fall_count", "target": 0}]},
        {"name": "stair_climb_10_steps", "description": "Climb 10 steps", "tasks": ["stair_climb"], "metrics": [{"name": "success_rate", "target": 0.90}]},
    ],
    safety_benchmarks=[
        {"name": "fall_recovery_time", "description": "Time to recover from fall", "target": 3.0, "unit": "s"},
        {"name": "max_fall_impact_force", "description": "Impact force during fall", "target": 500.0, "unit": "N"},
    ],
    baseline_hardware={"cpu": "Intel i7-12700", "gpu": "NVIDIA RTX 3080", "ram": "32GB DDR4", "os": "Ubuntu 22.04", "ros_distro": "humble"},
)

UNITREE_GO2_PROFILE = RobotCompleteProfile(
    robot_id="unitree_go2",
    name="Go2",
    vendor="Unitree",
    version="1.1",
    description="Agile 12-DOF quadruped with LIDAR, depth cameras, and onboard computing",
    embodiment=EMBODIMENT,
    safety=SAFETY,
    capability=CAPABILITY,
    simulation=SIMULATION,
    semantic=SEMANTIC,
    benchmark=BENCHMARK,
)
