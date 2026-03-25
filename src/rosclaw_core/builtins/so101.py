"""SO101 robot definition for ROSClaw V4.

The SO101 is a 6-DOF robot arm from The Robot Studio, commonly used
with LeRobot for imitation learning.
"""

from __future__ import annotations

from pathlib import Path

from rosclaw_core.definitions.robot import (
    RobotManifest,
    RobotCapabilities,
    JointConfig,
    JointType,
    JointLimits,
)


def create_so101_manifest(
    name: str = "so101",
    calibration_dir: Path | str | None = None,
) -> RobotManifest:
    """Create a standard SO101 robot manifest.

    The SO101 has 6 joints (shoulder to wrist) with position control.
    Motors are Feetech STS3215 (6V) or similar.

    Args:
        name: Robot instance name
        calibration_dir: Path to calibration directory

    Returns:
        RobotManifest for SO101
    """
    if calibration_dir is None:
        calibration_dir = Path(f"~/.rosclaw/calibration/{name}").expanduser()
    else:
        calibration_dir = Path(calibration_dir).expanduser()

    # SO101 joint configuration
    # Motor IDs 1-6 correspond to joints from base to wrist
    joints = [
        JointConfig(
            name="shoulder_pan",
            type=JointType.REVOLUTE,
            motor_id=1,
            limits=JointLimits(
                min_position=-2.8,
                max_position=2.8,
                max_velocity=2.0,
                max_effort=5.0,
            ),
        ),
        JointConfig(
            name="shoulder_lift",
            type=JointType.REVOLUTE,
            motor_id=2,
            limits=JointLimits(
                min_position=-1.5,
                max_position=1.5,
                max_velocity=2.0,
                max_effort=5.0,
            ),
        ),
        JointConfig(
            name="elbow_flex",
            type=JointType.REVOLUTE,
            motor_id=3,
            limits=JointLimits(
                min_position=-1.5,
                max_position=1.5,
                max_velocity=2.0,
                max_effort=5.0,
            ),
        ),
        JointConfig(
            name="wrist_flex",
            type=JointType.REVOLUTE,
            motor_id=4,
            limits=JointLimits(
                min_position=-2.0,
                max_position=2.0,
                max_velocity=3.0,
                max_effort=2.0,
            ),
        ),
        JointConfig(
            name="wrist_roll",
            type=JointType.REVOLUTE,
            motor_id=5,
            limits=JointLimits(
                min_position=-2.8,
                max_position=2.8,
                max_velocity=3.0,
                max_effort=2.0,
            ),
        ),
        JointConfig(
            name="gripper",
            type=JointType.REVOLUTE,
            motor_id=6,
            limits=JointLimits(
                min_position=0.0,
                max_position=0.08,  # ~8cm gripper opening
                max_velocity=1.0,
                max_effort=2.0,
            ),
        ),
    ]

    capabilities = RobotCapabilities(
        dof=6,
        has_gripper=True,
        max_payload_kg=0.5,
        reach_m=0.35,
        precision_mm=1.0,
        max_speed_m_s=0.1,
        supports_teleop=True,
        supports_recording=True,
        supports_policy=True,
    )

    return RobotManifest(
        name=name,
        version="1.0.0",
        hardware_type="so101",
        joints=joints,
        sensors=[],  # Cameras configured separately
        capabilities=capabilities,
        calibration_dir=calibration_dir,
        ros_namespace=f"/{name}",
        safety_limits={
            "max_joint_velocity": 2.0,
            "max_cartesian_velocity": 0.1,
        },
    )


def create_so101_leader_manifest(
    name: str = "so101_leader",
    calibration_dir: Path | str | None = None,
) -> RobotManifest:
    """Create an SO101 leader (teleop) arm manifest.

    Leader arms are passive (no motors) and used for demonstration.
    They read position from motor encoders but don't send commands.

    Args:
        name: Robot instance name
        calibration_dir: Path to calibration directory

    Returns:
        RobotManifest for SO101 leader
    """
    manifest = create_so101_manifest(name, calibration_dir)
    manifest.hardware_type = "so101_leader"
    manifest.capabilities.supports_recording = False
    manifest.capabilities.supports_policy = False
    return manifest


def create_so101_follower_manifest(
    name: str = "so101_follower",
    calibration_dir: Path | str | None = None,
) -> RobotManifest:
    """Create an SO101 follower arm manifest.

    Follower arms execute commands from teleop or policy.

    Args:
        name: Robot instance name
        calibration_dir: Path to calibration directory

    Returns:
        RobotManifest for SO101 follower
    """
    manifest = create_so101_manifest(name, calibration_dir)
    manifest.hardware_type = "so101_follower"
    return manifest
