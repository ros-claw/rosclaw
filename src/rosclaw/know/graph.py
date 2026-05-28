"""Knowledge Graph operations for v1.0.

Provides high-level graph queries over the SeekDB knowledge_graph table.
These are convenience wrappers around SeekDBClient.query().
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("rosclaw.know.graph")


def get_robot_capabilities(
    seekdb_client: Any,
    robot_id: str,
) -> list[dict[str, Any]]:
    """Query all capabilities for a robot.

    Returns list of {capability, confidence, source} dicts.
    """
    if seekdb_client is None:
        return []

    records = seekdb_client.query(
        "knowledge_graph",
        filters={"subject": robot_id, "predicate": "has_capability"},
        order_by="-confidence",
        limit=100,
    )
    return [
        {
            "capability": rec.get("object", ""),
            "confidence": rec.get("confidence", 1.0),
            "source": rec.get("source", ""),
        }
        for rec in records
    ]


def get_robot_symptoms(
    seekdb_client: Any,
    robot_id: str,
) -> list[dict[str, Any]]:
    """Query all known symptoms for a robot.

    Returns list of {symptom, confidence, source} dicts.
    """
    if seekdb_client is None:
        return []

    records = seekdb_client.query(
        "knowledge_graph",
        filters={"subject": robot_id, "predicate": "has_symptom"},
        order_by="-confidence",
        limit=100,
    )
    return [
        {
            "symptom": rec.get("object", ""),
            "confidence": rec.get("confidence", 1.0),
            "source": rec.get("source", ""),
        }
        for rec in records
    ]


def get_related_robots(
    seekdb_client: Any,
    capability: str,
) -> list[str]:
    """Find all robots that have a given capability.

    Returns list of robot IDs.
    """
    if seekdb_client is None:
        return []

    records = seekdb_client.query(
        "knowledge_graph",
        filters={"predicate": "has_capability", "object": capability},
        limit=100,
    )
    return sorted({rec.get("subject", "") for rec in records if rec.get("subject")})


def count_knowledge_facts(seekdb_client: Any) -> dict[str, int]:
    """Return counts of knowledge_graph entries by predicate type."""
    if seekdb_client is None:
        return {"total": 0, "capabilities": 0, "symptoms": 0}

    try:
        total = seekdb_client.count("knowledge_graph")
        capabilities = seekdb_client.count(
            "knowledge_graph", filters={"predicate": "has_capability"}
        )
        symptoms = seekdb_client.count(
            "knowledge_graph", filters={"predicate": "has_symptom"}
        )
        return {
            "total": total,
            "capabilities": capabilities,
            "symptoms": symptoms,
        }
    except Exception as exc:
        logger.warning("[Know] count_knowledge_facts failed: %s", exc)
        return {"total": 0, "capabilities": 0, "symptoms": 0}


__all__ = [
    "get_robot_capabilities",
    "get_robot_symptoms",
    "get_related_robots",
    "count_knowledge_facts",
]
