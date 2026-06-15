#!/usr/bin/env python3
"""
hello_robot.py - ROSClaw v1.0 Quick Start Example

Demonstrates the core ROSClaw workflow:
    1. Initialize Runtime
    2. Connect a simulated robot driver
    3. Register and execute skills
    4. Record practice sessions
    5. Use the EventBus for module communication
"""

from pathlib import Path

import numpy as np

from rosclaw.core import Event, EventBus, EventPriority, Runtime, RuntimeConfig
from rosclaw.core.types import PraxisEvent, RobotState
from rosclaw.mcp_drivers import MuJoCoSimDriver
from rosclaw.practice import PracticeRecorder
from rosclaw.skill_manager import SkillEntry, SkillExecutor, SkillRegistry

PROJECT_ROOT = Path(__file__).parent.parent


def main() -> int:
    print("=== ROSClaw Hello Robot ===\n")

    # ------------------------------------------------------------------
    # 1. Create EventBus (central nervous system)
    # ------------------------------------------------------------------
    bus = EventBus()
    bus.subscribe("agent.command", lambda e: print(f"  [Bus] Command: {e.payload}"))
    print("1. EventBus created")

    # ------------------------------------------------------------------
    # 2. Initialize Runtime
    # ------------------------------------------------------------------
    config = RuntimeConfig(
        robot_id="hello_bot",
        safety_level="MODERATE",
        timeline_output_dir=str(PROJECT_ROOT / "practice_data"),
        seekdb_backend="memory",
    )
    runtime = Runtime(config)
    runtime.initialize()
    runtime.start()
    print(f"2. Runtime started: {runtime.config.robot_id}")

    # ------------------------------------------------------------------
    # 3. Connect a simulated robot driver
    # ------------------------------------------------------------------
    driver = MuJoCoSimDriver(robot_id="hello_bot", model_path="", joint_dof=6)
    driver.initialize()
    driver.start()
    print(f"3. Driver connected: positions={driver.get_joint_positions()}")

    # Move joints
    target = [0.2, -0.3, 0.1, 0.0, 0.0, 0.0]
    driver.move_joints(target, duration=0.5)
    print(f"   Moved to: positions={driver.get_joint_positions()}")

    # ------------------------------------------------------------------
    # 4. Register skills
    # ------------------------------------------------------------------
    registry = SkillRegistry()
    pick_skill = SkillEntry(
        name="pick",
        description="Pick an object",
        skill_type="programmed",
        parameters={"object": "str"},
    )
    place_skill = SkillEntry(
        name="place",
        description="Place an object",
        skill_type="programmed",
    )
    registry.register(pick_skill)
    registry.register(place_skill)
    print(f"4. Skills registered: {registry.list_skills()}")

    # ------------------------------------------------------------------
    # 5. Execute a skill
    # ------------------------------------------------------------------
    executor = SkillExecutor(bus, registry)
    executor.initialize()
    result = executor.execute("pick", {"object": "red_cube"})
    print(f"5. Skill executed: {result['status']}")

    # ------------------------------------------------------------------
    # 6. Record a practice event
    # ------------------------------------------------------------------
    recorder = PracticeRecorder(robot_id="hello_bot", event_bus=bus)
    recorder.initialize()
    recorder.start()

    evt = PraxisEvent(
        event_id="evt-001",
        event_type="success",
        timestamp=0.0,
        robot_id="hello_bot",
        agent_instruction="pick red_cube",
        cot_trace=["plan", "execute", "verify"],
        initial_state=RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6),
        ),
        final_state=None,
        trajectory=[target],
        mcap_path=None,
        error_details=None,
        duration_sec=1.2,
    )
    recorder.record_praxis_event(evt)
    print("6. Practice event recorded")

    # ------------------------------------------------------------------
    # 7. Publish an event on the bus
    # ------------------------------------------------------------------
    bus.publish(
        Event(
            topic="agent.command",
            payload={"action": "done", "result": "success"},
            source="hello_robot",
            priority=EventPriority.NORMAL,
        )
    )
    print("7. Event published to bus")

    # ------------------------------------------------------------------
    # 8. Cleanup
    # ------------------------------------------------------------------
    recorder.stop()
    driver.stop()
    runtime.stop()
    print("\n=== Hello Robot complete ===")
    return 0


if __name__ == "__main__":
    exit(main())
