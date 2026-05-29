#!/usr/bin/env python3
"""
demo_how_recovery_loop.py — Failure → Recovery → Retry → Compare → Learn

Demonstrates the complete How recovery closed-loop:
    1. Run a task that fails
    2. How generates recovery hint from heuristic rules
    3. Apply patched parameters for retry
    4. Compare failure vs retry outcomes
    5. Update rule efficacy in Memory

Usage:
    PYTHONPATH=src python examples/demo_how_recovery_loop.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asyncio

from rosclaw.core import Runtime, RuntimeConfig, EventBus, Event, EventPriority
from rosclaw.how.engine import HeuristicEngine
from rosclaw.memory.seekdb_client import SeekDBMemoryClient


def print_section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


async def main() -> int:
    print_section("How Recovery Closed-Loop Demo")

    # ------------------------------------------------------------------
    # 1. Initialize Runtime with How enabled
    # ------------------------------------------------------------------
    print_section("Step 1: Initialize Runtime + How")
    config = RuntimeConfig(
        robot_id="ur5e",
        robot_zoo_path=str(PROJECT_ROOT / "e-urdf-zoo"),
        default_eurdf_robot="ur5e",
        enable_firewall=True,
        enable_memory=True,
        enable_practice=True,
        enable_how=True,
        enable_provider=True,
        seekdb_backend="memory",
    )
    runtime = Runtime(config)
    runtime.initialize()
    runtime.start()
    print("  Runtime + How initialized")

    # ------------------------------------------------------------------
    # 2. Seed a heuristic rule for common grasp failure
    # ------------------------------------------------------------------
    print_section("Step 2: Seed Heuristic Rule")
    how: HeuristicEngine = runtime._how
    if how is None:
        print("  [ERROR] How module not available")
        return 1

    await how.seed_defaults()
    print(f"  Seeded {len(how._rule_cache)} default rules")

    # ------------------------------------------------------------------
    # 3. Simulate a failure
    # ------------------------------------------------------------------
    print_section("Step 3: Simulate Failure")
    failure_reason = "joint limit exceeded during reach"
    print(f"  Failure: {failure_reason}")

    runtime.event_bus.publish(Event(
        topic="rosclaw.sandbox.episode.failed",
        payload={
            "episode_id": "ep_fail_001",
            "robot_id": "ur5e",
            "failure_type": "joint_limit_exceeded",
            "error_log": failure_reason,
            "params": {"approach_z": 0.3, "speed": 0.5},
        },
        source="demo",
    ))

    # ------------------------------------------------------------------
    # 4. How generates recovery hint
    # ------------------------------------------------------------------
    print_section("Step 4: How Recovery Hint")
    hint = await how.suggest_recovery(failure_reason)
    if hint:
        print(f"  Rule matched: {hint['rule_id']}")
        print(f"  Condition: {hint['condition'][:60]}...")
        print(f"  Action: {hint['action'][:80]}...")
        print(f"  Priority: {hint.get('priority', 'N/A')}")
    else:
        print("  [INFO] No exact rule match — using generic fallback")
        # Use a default hint for demo
        hint = {
            "rule_id": "demo_fallback",
            "condition": failure_reason,
            "action": "reduce speed to 0.2 and lower approach_z by 2cm",
            "priority": 5,
        }
        print(f"  Fallback action: {hint['action']}")

    # ------------------------------------------------------------------
    # 5. Apply recovery and retry
    # ------------------------------------------------------------------
    print_section("Step 5: Retry with Recovery Patch")
    patched_params = {"approach_z": 0.28, "speed": 0.2}
    print(f"  Original params: {{approach_z: 0.3, speed: 0.5}}")
    print(f"  Patched params:  {patched_params}")

    # Simulate success with patched params
    runtime.event_bus.publish(Event(
        topic="rosclaw.sandbox.episode.succeeded",
        payload={
            "episode_id": "ep_retry_001",
            "robot_id": "ur5e",
            "params": patched_params,
            "reward": 0.95,
        },
        source="demo",
    ))
    print("  Retry: SUCCESS")

    # ------------------------------------------------------------------
    # 6. Record outcome and update rule efficacy
    # ------------------------------------------------------------------
    print_section("Step 6: Update Rule Efficacy")
    rule_id = hint["rule_id"]
    await how.record_outcome(rule_id, success=True)
    print(f"  Rule {rule_id}: success_count incremented")

    # ------------------------------------------------------------------
    # 7. Compare outcomes
    # ------------------------------------------------------------------
    print_section("Step 7: Compare Outcomes")
    print("  ep_fail_001:  failed  | params={approach_z: 0.3, speed: 0.5} | reward=0.0")
    print("  ep_retry_001: success | params={approach_z: 0.28, speed: 0.2} | reward=0.95")
    print("  Improvement:  +0.95 reward")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    runtime.stop()
    print_section("Demo Complete")
    print("  Closed-loop: fail → How hint → patch → retry → success → learn")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
