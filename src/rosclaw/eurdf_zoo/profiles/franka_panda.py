"""Franka Emika Panda — Robot Profile Definition.

Collaborative 7-DOF torque-controlled manipulator.
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

# ── RobotEmbodimentProfile ──
EMBODIMENT = RobotEmbodimentProfile(
    robot_id="franka_emika_panda",
    name="Panda",
    vendor="Franka Emika",
    version="4.0",
    description="Collaborative 7-DOF torque-controlled manipulator with 3kg payload, 855mm reach",
    dof=7,
    links=[
        {"name": "base_link", "type": "base", "mass": 2.5},
        {"name": "link1", "type": "link", "mass": 4.0},
        {"name": "link2", "type": "link", "mass": 4.0},
        {"name": "link3", "type": "link", "mass": 3.0},
        {"name": "link4", "type": "link", "mass": 2.5},
        {"name": "link5", "type": "link", "mass": 1.5},
        {"name": "link6", "type": "link", "mass": 1.2},
        {"name": "link7", "type": "link", "mass": 0.5},
        {"name": "flange", "type": "end_effector_mount", "mass": 0.1},
        {"name": "tcp", "type": "tcp", "mass": 0.0},
    ],
    joints=[
        {"name": "panda_joint1", "type": "revolute", "parent": "base_link", "child": "link1", "axis": [0, 0, 1], "limits": {"lower": -2.8973, "upper": 2.8973, "velocity": 2.1750, "effort": 87.0}},
        {"name": "panda_joint2", "type": "revolute", "parent": "link1", "child": "link2", "axis": [0, 1, 0], "limits": {"lower": -1.7628, "upper": 1.7628, "velocity": 2.1750, "effort": 87.0}},
        {"name": "panda_joint3", "type": "revolute", "parent": "link2", "child": "link3", "axis": [0, 0, 1], "limits": {"lower": -2.8973, "upper": 2.8973, "velocity": 2.1750, "effort": 87.0}},
        {"name": "panda_joint4", "type": "revolute", "parent": "link3", "child": "link4", "axis": [0, 1, 0], "limits": {"lower": -3.0718, "upper": -0.0698, "velocity": 2.1750, "effort": 87.0}},
        {"name": "panda_joint5", "type": "revolute", "parent": "link4", "child": "link5", "axis": [0, 0, 1], "limits": {"lower": -2.8973, "upper": 2.8973, "velocity": 2.6100, "effort": 12.0}},
        {"name": "panda_joint6", "type": "revolute", "parent": "link5", "child": "link6", "axis": [0, 1, 0], "limits": {"lower": -0.0175, "upper": 3.7525, "velocity": 2.6100, "effort": 12.0}},
        {"name": "panda_joint7", "type": "revolute", "parent": "link6", "child": "link7", "axis": [0, 0, 1], "limits": {"lower": -2.8973, "upper": 2.8973, "velocity": 2.6100, "effort": 12.0}},
    ],
    sensors=[
        {"name": "joint_torque_sensors", "type": "torque", "parent_link": "*", "resolution": 0.01},
        {"name": "tcp_force_sensor", "type": "force_torque", "parent_link": "link7", "frame": "tcp", "range": {"force": [0, 200], "torque": [0, 10]}},
        {"name": "hand_camera", "type": "camera", "parent_link": "link7", "resolution": [1280, 800], "fov": 70.0, "fps": 30},
        {"name": "base_imu", "type": "imu", "parent_link": "base_link"},
    ],
    actuators=[
        {"name": "panda_motor_1", "type": "electric_motor", "joint": "panda_joint1", "max_current": 10.0, "gear_ratio": 100.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "panda_motor_2", "type": "electric_motor", "joint": "panda_joint2", "max_current": 10.0, "gear_ratio": 100.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "panda_motor_3", "type": "electric_motor", "joint": "panda_joint3", "max_current": 10.0, "gear_ratio": 100.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "panda_motor_4", "type": "electric_motor", "joint": "panda_joint4", "max_current": 10.0, "gear_ratio": 100.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "panda_motor_5", "type": "electric_motor", "joint": "panda_joint5", "max_current": 3.0, "gear_ratio": 50.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "panda_motor_6", "type": "electric_motor", "joint": "panda_joint6", "max_current": 3.0, "gear_ratio": 50.0, "control_mode": ["position", "velocity", "torque"]},
        {"name": "panda_motor_7", "type": "electric_motor", "joint": "panda_joint7", "max_current": 3.0, "gear_ratio": 50.0, "control_mode": ["position", "velocity", "torque"]},
    ],
    metadata={
        "urdf_version": "1.0",
        "ros2_package": "franka_description",
        "gripper_options": ["Franka Hand", "Robotiq 2F-85"],
        "certification": ["CE", "ISO_10218_1"],
    },
)

# ── RobotSafetyProfile ──
SAFETY = RobotSafetyProfile(
    robot_id="franka_emika_panda",
    safety_level="STRICT",
    safety_limits={
        "joint_limits": {
            "panda_joint1": {"lower": -2.8973, "upper": 2.8973},
            "panda_joint2": {"lower": -1.7628, "upper": 1.7628},
        },
        "velocity_limits": {"panda_joint1": 2.1750, "panda_joint2": 2.1750},
        "force_limits": {"max_tcp_force": 200.0, "max_tcp_torque": 10.0, "max_joint_effort": 87.0},
    },
    joint_soft_limits={
        "panda_joint1": {"lower": -2.8, "upper": 2.8, "k_position": 100.0, "k_velocity": 50.0},
        "panda_joint2": {"lower": -1.7, "upper": 1.7, "k_position": 100.0, "k_velocity": 50.0},
    },
    pfl={"enabled": True, "max_tcp_force": 140.0, "max_tcp_torque": 7.0, "max_static_force": 40.0},
    collision_detection={"method": "force_threshold", "threshold_force": 40.0, "threshold_torque": 2.5, "reaction_time_ms": 2.0},
    emergency_stop={"category": 3, "stop_time_ms": 200.0, "stop_distance_m": 0.2, "dual_channel": True},
    workspace_boundaries={"type": "enclosure", "fenceless": True},
    interaction={"hand_guiding": True, "speed_slider_max": 0.25, "force_guiding_max": 10.0, "protective_stop_on_contact": True},
    environment={"ambient_temp_min": 0.0, "ambient_temp_max": 45.0, "humidity_max": 80.0, "ip_rating": "IP40"},
)

# ── RobotCapabilityProfile ──
CAPABILITY = RobotCapabilityProfile(
    robot_id="franka_emika_panda",
    capabilities=[
        {"id": "panda_pick_and_place", "name": "pick_and_place", "category": "manipulation", "skill_type": "programmed", "constraints": {"max_payload": 3.0, "max_reach": 0.855}},
        {"id": "panda_insertion", "name": "insertion", "category": "assembly", "skill_type": "composed"},
        {"id": "panda_hand_guided_teaching", "name": "hand_guided_teaching", "category": "teaching", "skill_type": "programmed"},
        {"id": "panda_peg_in_hole", "name": "peg_in_hole", "category": "assembly", "skill_type": "composed"},
        {"id": "panda_drawer_opening", "name": "drawer_opening", "category": "manipulation", "skill_type": "composed"},
    ],
    skill_registry={
        "panda_manipulation": {"version": "1.0", "skills": ["pick_and_place", "push", "slide"]},
        "panda_assembly": {"version": "1.0", "skills": ["insertion", "peg_in_hole"]},
    },
    precondition_checks={
        "gripper_attached": {"check": "tool0.child_links > 0", "description": "Gripper must be attached"},
        "object_in_reach": {"check": "distance <= 0.855", "description": "Object within reach"},
    },
)

# ── RobotSimulationProfile ──
SIMULATION = RobotSimulationProfile(
    robot_id="franka_emika_panda",
    backends={
        "mujoco": {"model_file": "robot.mjcf.xml", "timestep": 0.002, "integrator": "implicitfast"},
        "isaac": {"model_file": "robot.urdf", "stage_units": "meters", "physics_dt": 0.00833},
        "gazebo": {"model_file": "robot.urdf", "physics_engine": "ode", "max_step_size": 0.001},
    },
)

# ── RobotSemanticProfile ──
SEMANTIC = RobotSemanticProfile(
    robot_id="franka_emika_panda",
    semantic_version="1.0",
    functional_regions=[
        {"name": "base", "link": "base_link", "tags": ["mount", "stationary"], "description": "Fixed base mounting point", "affordances": []},
        {"name": "wrist_assembly", "links": ["link5", "link6", "link7"], "tags": ["wrist", "compact", "dexterous", "3dof"], "description": "Three intersecting wrist joints", "affordances": ["orient", "rotate"]},
        {"name": "tool_mount", "link": "flange", "tags": ["end_effector_mount", "tcp"], "description": "Tool mounting flange", "affordances": ["attach_tool", "attach_gripper"]},
    ],
    grasp_points=[
        {"name": "upper_arm_grasp", "link": "link2", "position": [0, 0, 0.15], "approach_direction": [0, 0, 1], "description": "For hand-guiding", "safety": {"max_force": 50.0}},
    ],
    visual_features=[
        {"name": "shoulder_joint_visible", "type": "rotary_joint", "links": ["base_link", "link1"], "visual_cues": ["white_ring"], "llm_description": "Circular joint where robot connects to base"},
    ],
    task_descriptions={
        "pick_and_place": {"relevant_links": ["link6", "link7", "flange"], "workspace_hint": "Objects within 855mm of base", "collision_warning": "Watch elbow when reaching low objects"},
    },
    semantic_tags=["collaborative_robot", "research", "7dof", "torque_control", "force_sensing", "pick_and_place", "assembly", "european", "franka_series"],
)

# ── RobotBenchmarkProfile ──
BENCHMARK = RobotBenchmarkProfile(
    robot_id="franka_emika_panda",
    kinematic_benchmarks=[
        {"name": "ik_solve_rate", "description": "Success rate of analytical IK", "target": 0.999, "timeout_ms": 5.0},
        {"name": "fk_accuracy", "description": "FK accuracy vs CAD", "target": 1.0e-6, "unit": "m"},
    ],
    dynamic_benchmarks=[
        {"name": "trajectory_tracking_error", "description": "RMSE of joint tracking", "target": 0.001, "unit": "rad"},
        {"name": "force_control_bandwidth", "description": "Force control bandwidth", "target": 1000.0, "unit": "Hz"},
    ],
    simulation_benchmarks={
        "mujoco": [{"name": "real_time_factor", "target": 1.0, "tolerance": 0.05}],
    },
    task_benchmarks=[
        {"name": "tabletop_pick_10", "description": "Pick 10 objects", "tasks": ["pick_and_place"], "metrics": [{"name": "success_rate", "target": 0.95}]},
    ],
    safety_benchmarks=[
        {"name": "pfl_response_time", "description": "PFL response time", "target": 2.0, "unit": "ms"},
    ],
    baseline_hardware={"cpu": "Intel i7-12700", "gpu": "NVIDIA RTX 3080", "ram": "32GB DDR4", "os": "Ubuntu 22.04", "ros_distro": "humble"},
)

# ── Aggregated Complete Profile ──
FRANKA_PANDA_PROFILE = RobotCompleteProfile(
    robot_id="franka_emika_panda",
    name="Panda",
    vendor="Franka Emika",
    version="4.0",
    description="Collaborative 7-DOF torque-controlled manipulator with 3kg payload, 855mm reach",
    embodiment=EMBODIMENT,
    safety=SAFETY,
    capability=CAPABILITY,
    simulation=SIMULATION,
    semantic=SEMANTIC,
    benchmark=BENCHMARK,
)
