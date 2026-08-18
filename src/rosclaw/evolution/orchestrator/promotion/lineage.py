"""LineageTracker — skill evolution genealogy."""

import logging
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

logger = logging.getLogger("rosclaw.auto.promotion.lineage")


@dataclass
class LineageNode:
    """A node in the skill evolution tree."""

    skill_id: str
    parent: str = ""
    patch_id: str = ""
    experiment_id: str = ""
    result: str = ""  # improved / rejected / champion / deprecated
    metrics: dict = field(default_factory=dict)
    timestamp: str = ""
    children: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "parent": self.parent,
            "patch_id": self.patch_id,
            "experiment_id": self.experiment_id,
            "result": self.result,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "children": self.children,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LineageNode":
        return cls(
            skill_id=d["skill_id"],
            parent=d.get("parent", ""),
            patch_id=d.get("patch_id", ""),
            experiment_id=d.get("experiment_id", ""),
            result=d.get("result", ""),
            metrics=d.get("metrics", {}),
            timestamp=d.get("timestamp", ""),
            children=d.get("children", []),
        )


class LineageTracker:
    """Track skill evolution lineage across experiments."""

    def __init__(self, store_backend: Any):
        self._store = store_backend
        self._namespace = "lineage"

    def _ensure_namespace(self) -> None:
        import os

        path = os.path.join(str(self._store.base), self._namespace)
        os.makedirs(path, exist_ok=True)

    def record(
        self,
        skill_id: str,
        parent_skill: str,
        patch_id: str,
        experiment_id: str,
        result: str,
        metrics: dict | None = None,
    ) -> LineageNode:
        """Record a new lineage node."""
        self._ensure_namespace()
        from datetime import datetime

        node = LineageNode(
            skill_id=skill_id,
            parent=parent_skill,
            patch_id=patch_id,
            experiment_id=experiment_id,
            result=result,
            metrics=metrics or {},
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._store.save(self._namespace, skill_id, node.to_dict())

        # Update parent's children list
        if parent_skill:
            parent_data = self._store.load(self._namespace, parent_skill)
            if parent_data:
                parent_data.setdefault("children", []).append(skill_id)
                self._store.save(self._namespace, parent_skill, parent_data)
            else:
                # Parent not yet recorded; create root stub
                root = LineageNode(skill_id=parent_skill, children=[skill_id])
                self._store.save(self._namespace, parent_skill, root.to_dict())

        logger.info("LineageTracker: recorded %s -> %s (result=%s)", parent_skill, skill_id, result)
        return node

    def get_lineage(self, skill_id: str) -> list[LineageNode]:
        """Get full ancestry chain from root to given skill."""
        chain = []
        current = skill_id
        seen = set()
        while current and current not in seen:
            seen.add(current)
            data = self._store.load(self._namespace, current)
            if not data:
                break
            node = LineageNode.from_dict(data)
            chain.append(node)
            current = node.parent
        return list(reversed(chain))

    def get_descendants(self, skill_id: str) -> list[LineageNode]:
        """Get all descendants of a skill."""
        result = []
        queue = [skill_id]
        seen = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            data = self._store.load(self._namespace, current)
            if not data:
                continue
            node = LineageNode.from_dict(data)
            result.append(node)
            queue.extend(node.children)
        return result

    def render_tree(self, root_skill: str) -> str:
        """Render ASCII skill evolution tree."""
        lines = [f"Skill Lineage Tree: {root_skill}"]
        descendants = self.get_descendants(root_skill)
        for node in descendants:
            indent = "  " * len(self.get_lineage(node.skill_id))
            marker = (
                "✅" if node.result == "champion" else "❌" if node.result == "rejected" else "🔄"
            )
            lines.append(f"{indent}{marker} {node.skill_id} ({node.result})")
        return "\n".join(lines)
