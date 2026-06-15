#!/usr/bin/env python3
"""
control_pid_arm_demo.py - ROSClaw v1.0 PID Joint Control Demo

Demonstrates closed-loop PID control on a simulated robot arm:
    1. Load a UR5e robot profile from the e-URDF zoo
    2. Initialize Runtime + EventBus
    3. Connect MuJoCoSimDriver (mock mode if no model)
    4. Use PIDController to track a target joint configuration
    5. Register/execute a skill through SkillExecutor
    6. Record practice via PracticeRecorder
    7. Persist the experience to SeekDB via MemoryInterface
    8. Query the experience back from SeekDB
"""

import sys
import time
from pathlib import Path

import numpy as np

from rosclaw.control import PIDController, PIDGains
from rosclaw.core import Event, EventBus, EventPriority, Runtime, RuntimeConfig
from rosclaw.eurdf import RobotRegistry
from rosclaw.mcp_drivers import MuJoCoSimDriver
from rosclaw.memory import MemoryInterface, SeekDBMemoryClient
from rosclaw.practice import PracticeRecorder
from rosclaw.skill_manager import SkillEntry, SkillExecutor, SkillRegistry

PROJECT_ROOT = Path(__file__).parent.parent


def main() -> int:
    print("=== ROSClaw PID Arm Control Demo ===\n")

    robot_id = "ur5e_pid_demo"
    target_joints = [0.5, -0.5, 0.3, -0.2, 0.1, 0.0]

    # 1. Load robot profile from e-URDF zoo
    registry = RobotRegistry()
    try:
        profile = registry.install("ur5e")
        print(f"1. Loaded e-URDF profile: {profile.name} (DOF={profile.embodiment.dof})")
    except Exception as exc:
        print(f"1. Could not load ur5e profile: {exc}")
        profile = None

    # 2. EventBus + Runtime
    bus = EventBus()
    bus.subscribe(
        "telemetry.joint_error",
        lambda e: print(f"   [Bus] joint_error event: {e.payload['error_norm']:.4f}"),
    )

    config = RuntimeConfig(
        robot_id=robot_id,
        safety_level="MODERATE",
        timeline_output_dir=str(PROJECT_ROOT / "practice_data"),
        seekdb_backend="memory",
    )
    runtime = Runtime(config)
    runtime.initialize()
    runtime.start()
    print("2. Runtime started")

    # 3. MuJoCoSimDriver (uses real model if present, otherwise mock)
    model_path = str(PROJECT_ROOT / "e-urdf-zoo" / "ur5e" / "robot.mjcf.xml")
    driver = MuJoCoSimDriver(robot_id=robot_id, model_path=model_path, joint_dof=6)
    driver.initialize()
    driver.start()
    print(f"3. Driver ready (connected={driver.is_connected()})")

    # 4. PID controller per joint
    pids = [PIDController(PIDGains(kp=4.0, ki=0.1, kd=0.5)) for _ in range(6)]
    for pid in pids:
        pid.set_output_limit(-2.0, 2.0)
        pid.set_integral_limit(1.0)

    dt = 0.02
    duration = 2.0
    steps = int(duration / dt)
    trajectory: list[list[float]] = []

    print("4. Running PID tracking...")
    for step in range(steps):
        current = np.array(driver.get_joint_positions(), dtype=float)
        target = np.array(target_joints, dtype=float)
        errors = target - current
        commands = []
        for i, pid in enumerate(pids):
            cmd = pid.update(float(errors[i]), dt)
            commands.append(cmd)

        # Simple mock integration: new_pos = current + cmd * dt
        new_pos = current + np.array(commands) * dt
        driver.move_joints(new_pos.tolist(), duration=dt)
        trajectory.append(new_pos.tolist())

        if step % 20 == 0:
            error_norm = float(np.linalg.norm(errors))
            bus.publish(
                Event(
                    topic="telemetry.joint_error",
                    payload={"step": step, "error_norm": error_norm, "commands": commands},
                    source="pid_arm_demo",
                    priority=EventPriority.NORMAL,
                )
            )

    final_error = np.linalg.norm(np.array(target_joints) - np.array(driver.get_joint_positions()))
    print(f"   Final joint error norm: {final_error:.4f}")

    # 5. Skill registration + execution
    skill_registry = SkillRegistry()
    skill_registry.register(
        SkillEntry(
            name="hold_joint_position",
            description="Hold joints at the target configuration",
            skill_type="programmed",
            parameters={"target": target_joints},
        )
    )
    executor = SkillExecutor(bus, skill_registry)
    executor.initialize()
    result = executor.execute("hold_joint_position", {"target": target_joints})
    print(f"5. Skill executed: {result['status']}")

    # 6. Practice recording
    recorder = PracticeRecorder(robot_id=robot_id, joint_dof=6, event_bus=bus)
    recorder.initialize()
    recorder.start()
    print("6. Practice recorder started")

    # 7. SeekDB memory: store the tracking experience
    memory = MemoryInterface(
        robot_id=robot_id,
        event_bus=bus,
        seekdb_client=SeekDBMemoryClient(),
    )
    memory.initialize()
    memory.start()
    exp_id = memory.store_experience(
        event_id=f"{robot_id}-{int(time.time())}",
        event_type="pid_tracking",
        instruction="Track target joint configuration using PID control",
        trajectory=trajectory,
        outcome="success" if final_error < 0.5 else "partial",
        duration_sec=duration,
        tags=["pid", "arm", "control"],
        metadata={"target": target_joints, "final_error": final_error},
    )
    print(f"7. Experience stored in SeekDB: {exp_id}")

    # 8. Query it back
    experiences = memory.seekdb_client.query(
        "experience_graph",
        filters={"robot_id": robot_id},
        limit=5,
    )
    print(f"8. Queried {len(experiences)} experience(s) for {robot_id}")

    # Cleanup
    memory.stop()
    recorder.stop()
    executor.stop()
    driver.stop()
    runtime.stop()
    print("\n=== PID Arm Control Demo complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
