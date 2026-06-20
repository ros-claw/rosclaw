"""ROS-specific recovery rules for the HOW HeuristicEngine.

These rules seed the heuristic_rules table with ROS / rosbridge specific
remediations. They are registered when the ROS connector is initialized in a
Runtime that has a HOW engine.
"""

from __future__ import annotations

from typing import Any

ROS_RECOVERY_RULES: list[tuple[str, str, int]] = [
    (
        "rosbridge connection refused",
        "Verify rosbridge_server is running and reachable on the configured endpoint; check firewall rules.",
        3,
    ),
    (
        "Failed to connect to ws://",
        "Ensure the rosbridge WebSocket URL is correct and the server is listening on that port.",
        3,
    ),
    (
        "topic not found",
        "Confirm the ROS topic is advertised by a running node; use ros_discover to list active topics.",
        2,
    ),
    (
        "service not found",
        "Confirm the ROS service is advertised by a running node; check service type spelling.",
        2,
    ),
    (
        "action goal topic not found",
        "Confirm the action server is running and the /goal topic is available.",
        2,
    ),
    (
        "velocity command blocked",
        "Reduce linear/angular velocity and duration to stay within robot safety defaults; use validate-capability dry-run.",
        3,
    ),
    (
        "safety contract blocked",
        "Replan with args inside the capability constraints; inspect the capability schema and safety limits.",
        3,
    ),
    (
        "publish timeout",
        "Check rosbridge latency and topic subscription count; retry with a longer timeout.",
        2,
    ),
    (
        "service call timeout",
        "Check if the service server is alive; verify service type matches the request.",
        2,
    ),
]


def seed_ros_recovery_rules(seekdb_client: Any) -> int:
    """Insert ROS-specific recovery rules into the heuristic_rules table.

    Returns the number of rules inserted.
    """
    inserted = 0
    for idx, (condition, action, priority) in enumerate(ROS_RECOVERY_RULES):
        rid = f"ros_rule_{idx:03d}_{condition.replace(' ', '_')[:40]}"
        try:
            seekdb_client.insert(
                "heuristic_rules",
                {
                    "id": rid,
                    "condition": condition,
                    "action": action,
                    "priority": priority,
                    "success_count": 0,
                    "failure_count": 0,
                },
            )
            inserted += 1
        except Exception:
            # Best-effort seeding; ignore duplicates or missing table.
            pass
    return inserted
