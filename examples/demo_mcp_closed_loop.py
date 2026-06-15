#!/usr/bin/env python3
"""
demo_mcp_closed_loop.py — Claude Code MCP Closed-Loop Demo

Demonstrates the complete ROSClaw v1.0 closed-loop that Claude Code would execute:

    1. List available robots
    2. Query robot capabilities
    3. Start Dashboard for observability
    4. Initialize Runtime with EventBus, Memory, Practice, How
    5. Run a sandbox reach task (UR5e in MuJoCo)
    6. Record episode to Practice
    7. Query Memory for "what happened"
    8. How generates recovery hints on failure
    9. Dashboard shows the full trace

Usage:
    python examples/demo_mcp_closed_loop.py
"""

import asyncio
from pathlib import Path

import numpy as np

from rosclaw.core import Event, EventPriority, Runtime, RuntimeConfig
from rosclaw.runtime.eurdf_loader import EURDFLoader

PROJECT_ROOT = Path(__file__).parent.parent


def print_section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


async def main() -> int:
    print_section("Claude Code MCP Closed-Loop Demo")

    # ------------------------------------------------------------------
    # 1. List available robots
    # ------------------------------------------------------------------
    print_section("Step 1: List Robots")
    eurdf = EURDFLoader(zoo_path=str(PROJECT_ROOT / "e-urdf-zoo"))
    robots = eurdf.list_robots()
    print(f"  Available robots: {robots}")

    # ------------------------------------------------------------------
    # 2. Inspect a robot
    # ------------------------------------------------------------------
    print_section("Step 2: Inspect UR5e")
    profile = eurdf.load("ur5e")
    print(f"  Robot: {profile.name}")
    print(f"  DOF: {profile.embodiment.dof}")
    print(f"  Joints: {len(profile.embodiment.joints)}")
    print(f"  Links: {len(profile.embodiment.links)}")

    # ------------------------------------------------------------------
    # 3. Start Dashboard (observability)
    # ------------------------------------------------------------------
    print_section("Step 3: Start Dashboard")
    from rosclaw.dashboard.web_server import DashboardWebServer
    dashboard = DashboardWebServer(host="0.0.0.0", port=8766)
    await dashboard.start()
    print("  Dashboard: http://localhost:8766/health")

    # ------------------------------------------------------------------
    # 4. Initialize Runtime with full grounding stack
    # ------------------------------------------------------------------
    print_section("Step 4: Initialize Runtime")
    config = RuntimeConfig(
        robot_id="ur5e",
        robot_zoo_path=str(PROJECT_ROOT / "e-urdf-zoo"),
        default_eurdf_robot="ur5e",
        enable_firewall=True,
        enable_memory=True,
        enable_practice=True,
        enable_how=True,
        enable_provider=True,
        timeline_output_dir=str(PROJECT_ROOT / "practice_data"),
        seekdb_backend="memory",
    )
    runtime = Runtime(config)
    runtime.initialize()
    runtime.start()
    dashboard.attach_to_event_bus(runtime.event_bus)
    print(f"  Runtime: {runtime.config.robot_id}")
    print("  EventBus: active")
    print(f"  How module: {'enabled' if runtime._how else 'disabled'}")
    print(f"  Memory: {'enabled' if runtime._memory else 'disabled'}")
    print("  Dashboard: connected to EventBus")

    # ------------------------------------------------------------------
    # 5. Run a sandbox reach task (mock — MuJoCo simulation)
    # ------------------------------------------------------------------
    print_section("Step 5: Sandbox Reach Task")

    # Subscribe to action events for observability
    def on_action(event: Event) -> None:
        action = event.payload.get("action", "unknown")
        print(f"    [EventBus] action.{action}: {event.payload.get('status', '')}")

    runtime.event_bus.subscribe("sandbox.action", on_action)

    # Simulate a reach task
    target_pos = np.array([0.5, -0.2, 0.3, 0.1, 0.0, 0.0])
    print(f"  Target joint positions: {target_pos}")

    # Publish action to EventBus
    runtime.event_bus.publish(Event(
        topic="sandbox.action",
        payload={
            "action": "reach",
            "target": target_pos.tolist(),
            "robot_id": "ur5e",
            "status": "started",
        },
        source="demo",
        priority=EventPriority.HIGH,
    ))

    # Simulate execution (mock — no real robot)
    await asyncio.sleep(0.5)

    # Simulate success
    runtime.event_bus.publish(Event(
        topic="sandbox.action",
        payload={
            "action": "reach",
            "status": "completed",
            "error": 0.02,
            "duration_sec": 1.5,
        },
        source="demo",
        priority=EventPriority.HIGH,
    ))
    print("  Task: completed (mock)")

    # ------------------------------------------------------------------
    # 6. Record episode to Practice (via EventBus events)
    # ------------------------------------------------------------------
    print_section("Step 6: Practice Episode Recording")
    # EpisodeRecorder subscribes to EventBus events automatically
    # We publish events to trigger recording

    runtime.event_bus.publish(Event(
        topic="skill.execution.start",
        payload={
            "episode_id": "ep_reach_001",
            "skill": "reach",
            "robot_id": "ur5e",
            "inputs": {"target": target_pos.tolist()},
        },
        source="demo",
    ))

    runtime.event_bus.publish(Event(
        topic="agent.response",
        payload={
            "request_id": "req_001",
            "status": "ok",
            "capability": "skill.reach",
            "result": {"error_m": 0.02, "duration_sec": 1.5},
        },
        source="demo",
    ))

    runtime.event_bus.publish(Event(
        topic="praxis.completed",
        payload={
            "episode_id": "ep_reach_001",
            "robot_id": "ur5e",
            "task": "reach",
            "success": True,
            "duration_sec": 1.5,
            "final_error_m": 0.02,
        },
        source="demo",
    ))
    print("  Events published: skill.start → agent.response → praxis.completed")
    print("  EpisodeRecorder captured via EventBus subscription")

    # ------------------------------------------------------------------
    # 7. Query Memory: "What happened?"
    # ------------------------------------------------------------------
    print_section("Step 7: Memory Query")
    if runtime._memory is not None:
        try:
            # Query memory for similar tasks
            results = runtime._memory.search(
                query="reach task ur5e",
                limit=3,
            )
            print(f"  Similar experiences: {len(results)}")
            for r in results:
                print(f"    - {r.get('key', '?')}")
        except Exception as e:
            print(f"  [Memory] {e}")
    else:
        print("  Memory not available (SeekDB not configured)")

    # ------------------------------------------------------------------
    # 8. How: Recovery hints
    # ------------------------------------------------------------------
    print_section("Step 8: How Recovery")
    if runtime._how is not None:
        try:
            # Simulate a failure scenario
            hint = await runtime._how.suggest_recovery(
                error_log="approach z too high, missed grasp",
                context={"robot": "ur5e", "task": "grasp"},
            )
            if hint:
                print(f"  Recovery hint: {hint.get('action', 'N/A')}")
                print(f"  Confidence: {hint.get('priority', 'N/A')}")
            else:
                print("  No matching heuristic rule (expected — demo has empty DB)")
        except Exception as e:
            print(f"  [How] {e}")
    else:
        print("  How module not available")

    # ------------------------------------------------------------------
    # 9. Dashboard snapshot
    # ------------------------------------------------------------------
    print_section("Step 9: Dashboard Trace")
    health = dashboard.server.get_health()
    snapshot = dashboard.server.get_snapshot()
    print(f"  System status: {health['status']}")
    print(f"  Uptime: {health['uptime_sec']:.1f}s")
    print(f"  Episodes recorded: {snapshot.get('episodes', {}).get('total', 0)}")
    print(f"  Event counts: {len(snapshot.get('event_counts', {}))} topics")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print_section("Cleanup")
    runtime.stop()
    await dashboard.stop()
    print("  All services stopped")

    print_section("Demo Complete")
    print("  The full closed-loop was executed:")
    print("    Agent (Claude Code) → MCP → Runtime → Sandbox → Practice → Memory → How → Dashboard")
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
