"""ROS Connector - KNOW knowledge graph seeding.

Publishes ROS capability triples into the knowledge_graph so that
KnowledgeInterface can answer provider-selection queries for ROS robots.
"""

from __future__ import annotations

from typing import Any


def seed_ros_capabilities(knowledge_interface: Any, robot_id: str, capabilities: list[str]) -> int:
    """Seed capabilities for a ROS robot into the knowledge graph.

    Args:
        knowledge_interface: An object exposing ``add_triple`` or a SeekDB client.
        robot_id: The robot identifier.
        capabilities: List of capability ids (e.g. ``turtlesim.base.velocity_command``).

    Returns:
        Number of triples inserted.
    """
    inserted = 0
    for cap_id in capabilities:
        try:
            if hasattr(knowledge_interface, "add_triple"):
                knowledge_interface.add_triple(
                    triple_id=f"{robot_id}_cap_{cap_id.replace('.', '_')}",
                    subject=robot_id,
                    predicate="has_capability",
                    obj=cap_id,
                    confidence=1.0,
                    source="ros_connector_discovery",
                )
            elif hasattr(knowledge_interface, "_client"):
                client = knowledge_interface._client
                client.insert("knowledge_graph", {
                    "id": f"{robot_id}_cap_{cap_id.replace('.', '_')}",
                    "subject": robot_id,
                    "predicate": "has_capability",
                    "object": cap_id,
                    "confidence": 1.0,
                    "source": "ros_connector_discovery",
                })
            inserted += 1
        except Exception:
            pass
    return inserted
