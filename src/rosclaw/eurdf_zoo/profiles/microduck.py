"""Microduck simulation registration; game execution lives in microduck-lab.

Joint limits/mass from Pollen's 5946fd9c MJCF (SHA256 7a6fdf437f5a80c7348ad801f43f906b997a834389cb70cca5e1e8517ba38044).
This profile grants no real-hardware capability or calibration claim.
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

JOINTS = [
    {
        "name": "left_hip_yaw",
        "type": "revolute",
        "limits": [-0.4363323129985824, 0.5235987755982988],
    },
    {
        "name": "left_hip_roll",
        "type": "revolute",
        "limits": [-0.3839724354387525, 0.3839724354387525],
    },
    {
        "name": "left_hip_pitch",
        "type": "revolute",
        "limits": [-1.570796326795005, 1.5707963267947882],
    },
    {"name": "left_knee", "type": "revolute", "limits": [-1.570796326795012, 1.570796326794781]},
    {"name": "left_ankle", "type": "revolute", "limits": [-1.5707963267949019, 1.5707963267948912]},
    {"name": "neck_pitch", "type": "revolute", "limits": [-1.5707963267948966, 1.0471975511965976]},
    {"name": "head_pitch", "type": "revolute", "limits": [-1.5707963267948974, 1.5707963267948957]},
    {"name": "head_yaw", "type": "revolute", "limits": [-2.967059728390373, 2.967059728390348]},
    {"name": "head_roll", "type": "revolute", "limits": [-0.43633231299859127, 0.4363323129985735]},
    {
        "name": "right_hip_yaw",
        "type": "revolute",
        "limits": [-0.5235987755982988, 0.4363323129985824],
    },
    {
        "name": "right_hip_roll",
        "type": "revolute",
        "limits": [-0.3839724354387525, 0.3839724354387525],
    },
    {
        "name": "right_hip_pitch",
        "type": "revolute",
        "limits": [-1.5707963267949268, 1.5707963267948664],
    },
    {"name": "right_knee", "type": "revolute", "limits": [-1.570796326794932, 1.570796326794861]},
    {
        "name": "right_ankle",
        "type": "revolute",
        "limits": [-1.5707963267949054, 1.5707963267948877],
    },
]
LINKS = [
    {"name": "trunk_base", "mass": 0.199224, "type": "link", "parent": "world"},
    {"name": "yaw2roll", "mass": 0.0230406, "type": "link", "parent": "trunk_base"},
    {"name": "hip_l", "mass": 0.00618934, "type": "link", "parent": "yaw2roll"},
    {"name": "upper_leg_left", "mass": 0.0482067, "type": "link", "parent": "hip_l"},
    {"name": "leg", "mass": 0.0215844, "type": "link", "parent": "upper_leg_left"},
    {"name": "ankle_left", "mass": 0.0300246, "type": "link", "parent": "leg"},
    {"name": "neck", "mass": 0.0368414, "type": "link", "parent": "trunk_base"},
    {"name": "neck_pitch", "mass": 0.00572, "type": "link", "parent": "neck"},
    {"name": "yaw_roll_motion", "mass": 0.0486, "type": "link", "parent": "neck_pitch"},
    {"name": "jaw_soft", "mass": 0.188766, "type": "link", "parent": "yaw_roll_motion"},
    {"name": "bearing_roll", "mass": 0.0230406, "type": "link", "parent": "trunk_base"},
    {"name": "hip_l_2", "mass": 0.00618934, "type": "link", "parent": "bearing_roll"},
    {"name": "upper_leg_right", "mass": 0.0482067, "type": "link", "parent": "hip_l_2"},
    {"name": "leg_2", "mass": 0.0215844, "type": "link", "parent": "upper_leg_right"},
    {"name": "ankle_right", "mass": 0.0300251, "type": "link", "parent": "leg_2"},
]
EMBODIMENT = RobotEmbodimentProfile(
    robot_id="microduck",
    name="Pollen Microduck (simulation)",
    vendor="Pollen Robotics",
    version="1.0.0",
    description="14-axis biped, native MJCF in the external microduck-lab executor.",
    dof=14,
    joints=JOINTS,
    links=LINKS,
    sensors=[{"name": "imu", "type": "imu"}],
    actuators=[{"name": j["name"], "type": "position", "torque_limit_Nm": 0.6405} for j in JOINTS],
    metadata={
        "evidence_domain": "simulation",
        "hardware_verified": False,
        "motor_hz": 50,
        "mjcf_sha256": "7a6fdf437f5a80c7348ad801f43f906b997a834389cb70cca5e1e8517ba38044",
        "mjcf_uri": "https://github.com/pollen-robotics/microduck_rl/blob/5946fd9cdbc58956424420153e51975af3b30d77/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml",
    },
)
MICRODUCK_PROFILE = RobotCompleteProfile(
    robot_id="microduck",
    name=EMBODIMENT.name,
    vendor=EMBODIMENT.vendor,
    version="1.0.0",
    description=EMBODIMENT.description,
    embodiment=EMBODIMENT,
    identity={"robot_class": "biped", "evidence_domain": "simulation"},
    safety=RobotSafetyProfile(
        robot_id="microduck",
        safety_level="SIMULATION_ONLY",
        safety_limits={"motor_torque_Nm": 0.6405},
        environment={"real_robot_execution_allowed": False},
    ),
    capability=RobotCapabilityProfile(
        robot_id="microduck",
        capabilities=[{"id": "microduck.start_game"}],
        skill_registry={
            "forbidden_capabilities": [{"id": "real_execution", "reason": "simulation only"}]
        },
    ),
    simulation=RobotSimulationProfile(
        robot_id="microduck", backends={"mujoco": {"external_executor": "microduck-lab"}}
    ),
    semantic=RobotSemanticProfile(robot_id="microduck", semantic_version="1.0"),
    benchmark=RobotBenchmarkProfile(robot_id="microduck"),
)
