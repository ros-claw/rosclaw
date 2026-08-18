"""Storage helpers for Knowledge Graph.

Provides functions to seed and query the SeekDB knowledge_graph table.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("rosclaw.know.storage")


# -- Seed data for v1.0 --

# Robot capability triples: (subject, predicate, object, confidence, source)
_SEED_CAPABILITIES: list[tuple[str, str, str, float, str]] = [
    # UR5e
    ("ur5e", "has_capability", "6dof_arm", 1.0, "e_urdf"),
    ("ur5e", "has_capability", "grasp", 1.0, "e_urdf"),
    ("ur5e", "has_capability", "pick_and_place", 1.0, "e_urdf"),
    ("ur5e", "has_capability", "payload_5kg", 1.0, "e_urdf"),
    ("ur5e", "has_capability", "reach_850mm", 1.0, "e_urdf"),
    ("ur5e", "has_capability", "force_control", 0.9, "e_urdf"),
    ("ur5e", "has_capability", "tool_changer", 0.8, "e_urdf"),
    ("ur5e", "has_capability", "sort_objects", 0.7, "task_profile"),
    ("ur5e", "has_capability", "stack_blocks", 0.75, "task_profile"),
    # Panda (Franka Emika)
    ("panda", "has_capability", "7dof_arm", 1.0, "e_urdf"),
    ("panda", "has_capability", "grasp", 1.0, "e_urdf"),
    ("panda", "has_capability", "pick_and_place", 1.0, "e_urdf"),
    ("panda", "has_capability", "payload_3kg", 1.0, "e_urdf"),
    ("panda", "has_capability", "torque_sensing", 1.0, "e_urdf"),
    ("panda", "has_capability", "collision_detection", 0.95, "e_urdf"),
    ("panda", "has_capability", "impedance_control", 0.9, "e_urdf"),
    ("panda", "has_capability", "assembly", 0.85, "task_profile"),
    ("panda", "has_capability", "handover_object", 0.8, "task_profile"),
    # Unitree G1
    ("unitree_g1", "has_capability", "humanoid_walk", 1.0, "e_urdf"),
    ("unitree_g1", "has_capability", "grasp", 1.0, "e_urdf"),
    ("unitree_g1", "has_capability", "balance", 1.0, "e_urdf"),
    ("unitree_g1", "has_capability", "locomotion", 1.0, "e_urdf"),
    ("unitree_g1", "has_capability", "payload_2kg", 1.0, "e_urdf"),
    ("unitree_g1", "has_capability", "walk_to_point", 0.95, "task_profile"),
    ("unitree_g1", "has_capability", "open_door", 0.7, "task_profile"),
    # Boston Dynamics Spot
    ("spot", "has_capability", "quadruped_walk", 1.0, "e_urdf"),
    ("spot", "has_capability", "inspect_surface", 0.9, "task_profile"),
    ("spot", "has_capability", "locomotion", 1.0, "e_urdf"),
    ("spot", "has_capability", "payload_14kg", 1.0, "e_urdf"),
    ("spot", "has_capability", "balance", 0.95, "e_urdf"),
    # AgileX Piper
    ("agilex_piper", "has_capability", "6dof_arm", 1.0, "e_urdf"),
    ("agilex_piper", "has_capability", "grasp", 1.0, "e_urdf"),
    ("agilex_piper", "has_capability", "pick_and_place", 0.9, "e_urdf"),
    ("agilex_piper", "has_capability", "payload_1kg", 1.0, "e_urdf"),
    ("agilex_piper", "has_capability", "sort_objects", 0.75, "task_profile"),
]

# Symptom triples: (subject, predicate, object, confidence, source)
_SEED_SYMPTOMS: list[tuple[str, str, str, float, str]] = [
    ("ur5e", "has_symptom", "Torque_Overflow", 1.0, "curated"),
    ("ur5e", "has_symptom", "Velocity_Divergence", 1.0, "curated"),
    ("ur5e", "has_symptom", "Oscillation_Divergence", 0.9, "curated"),
    ("ur5e", "has_symptom", "Communication_Timeout", 0.8, "curated"),
    ("panda", "has_symptom", "Torque_Overflow", 1.0, "curated"),
    ("panda", "has_symptom", "Numerical_Instability", 0.9, "curated"),
    ("panda", "has_symptom", "Memory_Exhaustion", 0.8, "curated"),
    ("panda", "has_symptom", "Collision_Detected", 0.95, "curated"),
    ("unitree_g1", "has_symptom", "Velocity_Divergence", 1.0, "curated"),
    ("unitree_g1", "has_symptom", "Balance_Loss", 0.95, "curated"),
    ("unitree_g1", "has_symptom", "Torque_Overflow", 0.9, "curated"),
]


def seed_knowledge_graph(seekdb_client: Any) -> dict[str, int]:
    """Populate knowledge_graph with v1.0 seed data.

    Idempotent: safe to call multiple times (uses INSERT OR REPLACE).
    """
    counts = {"capabilities": 0, "symptoms": 0, "total": 0}

    if seekdb_client is None:
        logger.warning("[Know] No SeekDB client provided, skipping seed")
        return counts

    all_records = _SEED_CAPABILITIES + _SEED_SYMPTOMS
    for subject, predicate, obj, confidence, source in all_records:
        record_id = f"{subject}_{predicate}_{obj}"
        try:
            seekdb_client.insert(
                "knowledge_graph",
                {
                    "id": record_id,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": confidence,
                    "source": source,
                    "timestamp": time.time(),
                },
            )
            counts["total"] += 1
            if predicate == "has_capability":
                counts["capabilities"] += 1
            elif predicate == "has_symptom":
                counts["symptoms"] += 1
        except Exception as exc:
            logger.warning("[Know] Failed to insert %s: %s", record_id, exc)

    logger.info(
        "[Know] Seeded knowledge_graph: %d capabilities, %d symptoms, %d total",
        counts["capabilities"],
        counts["symptoms"],
        counts["total"],
    )
    return counts


def ingest_e_urdf_capabilities(
    robot_name: str,
    semantic_tags: list[str],
    seekdb_client: Any,
) -> int:
    """Write e-URDF semantic tags as capability triples into knowledge_graph.

    Returns number of triples written.
    """
    if seekdb_client is None or not semantic_tags:
        return 0

    count = 0
    for tag in semantic_tags:
        if not tag:
            continue
        record_id = f"{robot_name}_has_capability_{tag}"
        try:
            seekdb_client.insert(
                "knowledge_graph",
                {
                    "id": record_id,
                    "subject": robot_name,
                    "predicate": "has_capability",
                    "object": tag,
                    "confidence": 1.0,
                    "source": "e_urdf",
                    "timestamp": time.time(),
                },
            )
            count += 1
        except Exception as exc:
            logger.warning("[Know] Failed to insert capability %s: %s", tag, exc)

    logger.info("[Know] Ingested %d capabilities from e-URDF for %s", count, robot_name)
    return count


__all__ = ["seed_knowledge_graph", "ingest_e_urdf_capabilities"]
