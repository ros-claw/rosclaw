#!/usr/bin/env python3
"""Swarm Multi-Agent Combat Task Demo — G1 + UR5 Collaborative Carry.

Demonstrates the complete Sprint 11 flow:
  1. EventBus setup
  2. EpisodeRecorder (Practice) initialization
  3. SwarmCoordinator task decomposition + auction allocation
  4. RaftLikeConsensus for shared state agreement
  5. Simulated skill execution with EventBus events
  6. PraxisEvent recording and artifact generation

Usage:
    python3 examples/demo_swarm_combat_task.py
"""

from __future__ import annotations

import time

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.practice.episode_recorder import EpisodeRecorder
from rosclaw.swarm.coordinator import SwarmCoordinator

# ──────────────────────────────────────────────────────────────────
# Scenario configuration
# ──────────────────────────────────────────────────────────────────

SCENARIO = {
    "task_id": "carry_table_001",
    "goal": "Collaboratively carry a table from zone_A to zone_B",
    "type": "collaborative_carry",
    "objects": ["table_001"],
    "from_location": "zone_A",
    "to_location": "zone_B",
    "target_position": (5.0, 0.0, 0.0),
}

AGENTS = [
    {
        "agent_id": "g1",
        "hardware_type": "Unitree_G1",
        "capabilities": ["skill.locomotion", "skill.balance", "skill.pick_and_place"],
        "position": (0.0, 0.0, 0.0),
    },
    {
        "agent_id": "ur5e",
        "hardware_type": "UR5e",
        "capabilities": ["skill.manipulation", "skill.pick_and_place", "skill.force_control"],
        "position": (1.0, 0.0, 0.0),
    },
]


def build_task() -> dict:
    """Build the high-level collaborative task."""
    return {
        "id": SCENARIO["task_id"],
        "type": "parallel_pick",
        "objects": SCENARIO["objects"],
        "target_location": SCENARIO["to_location"],
        "from_location": SCENARIO["from_location"],
        "target_position": SCENARIO["target_position"],
        "required_capabilities": ["skill.pick_and_place"],
    }


def setup_infrastructure() -> tuple[EventBus, EpisodeRecorder, SwarmCoordinator]:
    """Initialize EventBus, Practice recorder, and SwarmCoordinator."""
    print("=" * 60)
    print("🚀 Swarm Multi-Agent Combat Task Demo")
    print("=" * 60)

    # 1. EventBus
    bus = EventBus()
    print("\n[1/6] EventBus initialized")

    # 2. EpisodeRecorder for Practice logging
    recorder = EpisodeRecorder(
        robot_id="swarm_g1_ur5e",
        event_bus=bus,
        artifact_base_dir="~/.rosclaw/demo_swarm",
    )
    recorder.initialize()
    print("[2/6] EpisodeRecorder initialized")

    # 3. SwarmCoordinator wired to EventBus
    coordinator = SwarmCoordinator(event_bus=bus)
    print("[3/6] SwarmCoordinator initialized")

    return bus, recorder, coordinator


def register_agents(coordinator: SwarmCoordinator) -> None:
    """Register all swarm agents with the coordinator."""
    print("\n[4/6] Registering agents...")
    for agent in AGENTS:
        coordinator.register_agent(
            agent_id=agent["agent_id"],
            capabilities=agent["capabilities"],
            position=agent["position"],
        )
        print(f"  • {agent['agent_id']} ({agent['hardware_type']}) @ {agent['position']}")


def run_task_decomposition(coordinator: SwarmCoordinator, task: dict) -> list[dict]:
    """Decompose the high-level task into subtasks."""
    print(f"\n[5/6] Decomposing task: {task['id']}")
    subtasks = coordinator.decompose_task(task)
    print(f"  Generated {len(subtasks)} subtask(s):")
    for st in subtasks:
        print(f"    - {st['id']} | type={st['type']} | required={st.get('required_capabilities', [])}")
    return subtasks


def run_auction_allocation(coordinator: SwarmCoordinator, task: dict) -> dict:
    """Run auction-based task allocation."""
    print(f"\n[6/6] Auction allocation for: {task['id']}")
    allocation = coordinator.allocate_task(task)

    if not allocation.feasible:
        print(f"  ❌ Allocation FAILED: {allocation.reason}")
        return {}

    print(f"  ✅ Allocation SUCCESS ({len(allocation.assignments)} assignment(s)):")
    for ass in allocation.assignments:
        print(f"    - {ass['subtask_id']} → agent={ass['agent_id']} cost={ass['cost']:.2f}")
    return allocation


def simulate_execution(
    event_bus: EventBus,
    coordinator: SwarmCoordinator,
    allocation: dict,
) -> None:
    """Simulate subtask execution by publishing skill events."""
    print("\n[EXEC] Simulating subtask execution...")

    for assignment in allocation.assignments:
        agent_id = assignment["agent_id"]
        subtask_id = assignment["subtask_id"]
        episode_id = f"{SCENARIO['task_id']}_{subtask_id}"

        # skill.execution.start
        event_bus.publish(Event(
            topic="skill.execution.start",
            payload={
                "episode_id": episode_id,
                "skill_name": subtask_id,
                "agent_id": agent_id,
                "initial_state": {"position": "zone_A", "gripper": "open"},
                "parameters": {"object": SCENARIO["objects"][0]},
            },
            source="swarm_executor",
            priority=EventPriority.HIGH,
        ))
        print(f"  ▶ {subtask_id} started on {agent_id}")

        time.sleep(0.05)  # tiny delay for event ordering

        # skill.execution.complete
        event_bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "episode_id": episode_id,
                "skill_name": subtask_id,
                "agent_id": agent_id,
                "final_state": {"position": "zone_B", "gripper": "closed"},
                "result": {"success": True, "object_moved": SCENARIO["objects"][0]},
                "duration_sec": 2.5,
            },
            source="swarm_executor",
            priority=EventPriority.HIGH,
        ))
        print(f"  ✓ {subtask_id} completed on {agent_id}")

        time.sleep(0.05)


def run_consensus(
    event_bus: EventBus,
    coordinator: SwarmCoordinator,
) -> bool:
    """Run Raft-like consensus on task completion state."""
    print("\n[CONSENSUS] Reaching agreement on task completion...")

    # Each agent proposes that the task is complete
    ts = time.time()
    for agent in AGENTS:
        coordinator.propose_state(
            agent_id=agent["agent_id"],
            key="task_complete",
            value=True,
            timestamp=ts,
        )

    agreed = coordinator.get_consensus("task_complete")
    if agreed:
        print(f"  ✅ Consensus reached: task_complete={agreed}")
        return True
    else:
        print("  ❌ Consensus NOT reached")
        return False


def publish_praxis_completion(event_bus: EventBus) -> None:
    """Publish the terminal praxis.completed event."""
    print("\n[PRAXIS] Publishing completion event...")
    event_bus.publish(Event(
        topic="praxis.completed",
        payload={
            "episode_id": SCENARIO["task_id"],
            "outcome": {
                "success": True,
                "reward": 1.0,
                "description": "Table successfully carried from zone_A to zone_B",
            },
        },
        source="swarm_demo",
        priority=EventPriority.HIGH,
    ))
    print("  ✅ praxis.completed published")


def verify_artifacts(recorder: EpisodeRecorder) -> None:
    """Verify that EpisodeRecorder generated artifacts."""
    print("\n[VERIFY] Checking Practice artifacts...")
    episodes = recorder.list_episodes()
    print(f"  Recorded episodes: {len(episodes)}")
    for ep in episodes:
        print(f"    - {ep['episode_id']} | status={ep['status']} | reward={ep['reward']}")

    # Show artifact directory contents
    base_dir = recorder.artifact_base
    episodes_dir = base_dir / "episodes"
    if episodes_dir.exists():
        for ep_name in sorted(episodes_dir.iterdir()):
            if ep_name.is_dir():
                files = list(ep_name.iterdir())
                print(f"    Artifact dir {ep_name.name}: {len(files)} file(s)")
                for f in files:
                    print(f"      • {f.name}")


def print_event_summary(event_bus: EventBus) -> None:
    """Print summary of all events that traversed the bus."""
    print("\n[SUMMARY] EventBus traffic:")
    topic_counts: dict[str, int] = {}
    for ev in event_bus._event_history:
        topic_counts[ev.topic] = topic_counts.get(ev.topic, 0) + 1
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {topic:50s} {count:3d}")


def main() -> int:
    """Run the full combat task demo."""
    bus, recorder, coordinator = setup_infrastructure()

    # Register agents
    register_agents(coordinator)

    # Build and decompose task
    task = build_task()
    run_task_decomposition(coordinator, task)

    # Auction allocation
    allocation = run_auction_allocation(coordinator, task)
    if not allocation:
        print("\n❌ Demo aborted: task allocation failed")
        recorder.stop()
        return 1

    # Simulate execution (publishes skill events)
    simulate_execution(bus, coordinator, allocation)

    # Consensus
    consensus_ok = run_consensus(bus, coordinator)
    if not consensus_ok:
        print("\n❌ Demo aborted: consensus failed")
        recorder.stop()
        return 1

    # Terminal praxis event
    publish_praxis_completion(bus)

    # Allow recorder to finalize
    time.sleep(0.2)

    # Verify artifacts
    verify_artifacts(recorder)

    # Print event summary
    print_event_summary(bus)

    # Cleanup
    recorder.stop()
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
