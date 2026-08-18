"""LineageRepository (PR-DF-14 / flywheel §36-38; typed in PR-DF-17 / §8).

The typed ancestry graph over the ``lineage_edges`` table.  Answers the
acceptance questions — "why is this champion v1.7?" — by walking:

    Champion → Evaluation → Experiment → Patch → Proposal
             → MemoryInsight → Memory → Episode → Receipt

Relations are the canonical ``LineageRelation`` vocabulary (§8.2); entity
types are the canonical ``LineageEntityType`` vocabulary (§8.1).

An edge reads ``from --relation--> to`` in the child/derived → parent/source
orientation (§8.5): e.g.
``link("champion", "champ_1", "promoted_from", "evaluation", "eval_9")``.
Writes are idempotent on the 5-field key
(from_type, from_id, relation, to_type, to_id) — two entities of different
types may share an id namespace, so the type is part of identity.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

TABLE = "lineage_edges"


class LineageRepository:
    """Directed typed lineage graph on the structured store."""

    def __init__(self, structured_store: Any) -> None:
        self._store = structured_store

    # -- write ------------------------------------------------------------

    def link(
        self,
        from_type: str,
        from_id: str,
        relation: str,
        to_type: str,
        to_id: str,
        *,
        trace_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create an edge; idempotent on (from_type, from_id, relation, to_type, to_id)."""
        existing = self._store.query(
            TABLE,
            {
                "from_type": from_type,
                "from_id": from_id,
                "relation": relation,
                "to_type": to_type,
                "to_id": to_id,
            },
            limit=1,
        )
        if existing:
            return existing[0]["id"]
        edge_id = f"lin_{uuid.uuid4().hex[:16]}"
        self._store.insert(
            TABLE,
            {
                "id": edge_id,
                "from_type": from_type,
                "from_id": from_id,
                "relation": relation,
                "to_type": to_type,
                "to_id": to_id,
                "trace_id": trace_id,
                "metadata": json.dumps(metadata or {}, default=str),
                "created_at": time.time(),
            },
        )
        return edge_id

    # -- read ---------------------------------------------------------------

    def parents(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        """Edges pointing AWAY from this entity toward its sources."""
        return self._store.query(
            TABLE, {"from_type": entity_type, "from_id": entity_id}, limit=10_000
        )

    def children(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        """Edges pointing AT this entity (things derived from it)."""
        return self._store.query(TABLE, {"to_type": entity_type, "to_id": entity_id}, limit=10_000)

    def ancestors(
        self, entity_type: str, entity_id: str, *, max_depth: int = 32
    ) -> list[dict[str, Any]]:
        """BFS over parents(); cycle-safe, depth-capped."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        frontier = [(entity_type, entity_id)]
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            nxt: list[tuple[str, str]] = []
            for ftype, fid in frontier:
                for edge in self.parents(ftype, fid):
                    eid = edge["id"]
                    if eid in seen:
                        continue
                    seen.add(eid)
                    out.append(edge)
                    node = (edge.get("to_type", ""), edge.get("to_id", ""))
                    if node[1] and node not in nxt:
                        nxt.append(node)
            frontier = nxt
        return out

    def descendants(
        self, entity_type: str, entity_id: str, *, max_depth: int = 32
    ) -> list[dict[str, Any]]:
        """BFS over children(); cycle-safe, depth-capped."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        frontier = [(entity_type, entity_id)]
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            nxt: list[tuple[str, str]] = []
            for ftype, fid in frontier:
                for edge in self.children(ftype, fid):
                    eid = edge["id"]
                    if eid in seen:
                        continue
                    seen.add(eid)
                    out.append(edge)
                    node = (edge.get("from_type", ""), edge.get("from_id", ""))
                    if node[1]:
                        nxt.append(node)
            frontier = nxt
        return out

    def trace(self, entity_type: str, entity_id: str) -> dict[str, Any]:
        """The single-path ancestry chain for CLI/reporting (§38)."""
        chain = []
        current_type, current_id = entity_type, entity_id
        visited = {(current_type, current_id)}
        while True:
            edges = [
                e
                for e in self.parents(current_type, current_id)
                if (e["to_type"], e["to_id"]) not in visited
            ]
            if not edges:
                break
            edge = edges[0]
            chain.append(
                {
                    "relation": edge["relation"],
                    "to_type": edge["to_type"],
                    "to_id": edge["to_id"],
                }
            )
            visited.add((edge["to_type"], edge["to_id"]))
            current_type, current_id = edge["to_type"], edge["to_id"]
        return {"root": {"type": entity_type, "id": entity_id}, "chain": chain}

    def trace_graph(
        self,
        entity_type: str,
        entity_id: str,
        *,
        max_depth: int = 16,
        max_nodes: int = 500,
    ) -> dict[str, Any]:
        """The full ancestry DAG rooted at this entity (§11).

        Unlike :meth:`trace` (first-parent chain), this walks every parent
        branch.  Returns nodes (with type/id/depth) and edges (with relation)
        so callers can render trees, JSON, or feed a visualizer.  Cycle-safe;
        caps prevent pathological fan-out.
        """
        root_key = f"{entity_type}:{entity_id}"
        nodes: dict[str, dict[str, Any]] = {
            root_key: {"type": entity_type, "id": entity_id, "depth": 0}
        }
        edges: list[dict[str, Any]] = []
        seen_edges: set[str] = set()
        frontier = [(entity_type, entity_id, 0)]
        truncated = False
        while frontier:
            ftype, fid, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for edge in self.parents(ftype, fid):
                eid = edge["id"]
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)
                to_type, to_id = edge["to_type"], edge["to_id"]
                to_key = f"{to_type}:{to_id}"
                edges.append(
                    {
                        "from": f"{ftype}:{fid}",
                        "relation": edge["relation"],
                        "to": to_key,
                    }
                )
                if len(nodes) >= max_nodes:
                    truncated = True
                    break
                if to_key not in nodes:
                    nodes[to_key] = {"type": to_type, "id": to_id, "depth": depth + 1}
                    frontier.append((to_type, to_id, depth + 1))
            if truncated:
                break
        return {
            "root": root_key,
            "nodes": list(nodes.values()),
            "edges": edges,
            "truncated": truncated,
        }
