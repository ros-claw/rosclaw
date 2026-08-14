"""LineageRepository (PR-DF-14 / flywheel §36-38).

The typed ancestry graph over the ``lineage_edges`` table.  Answers the
acceptance questions — "why is this champion v1.7?" — by walking:

    Champion → Evaluation → Experiment → Patch → Proposal
             → MemoryInsight → Memory → Episode → Receipt

Relations (§36): derived_from / observed_in / supported_by / proposed_from /
patched_by / evaluated_by / promoted_from / supersedes / recovered_by /
generated_from.

An edge reads ``from --relation--> to``: e.g.
``link("champion", "champ_1", "promoted_from", "evaluation", "eval_9")``.
Writes are idempotent on (from, relation, to).
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
        """Create an edge; idempotent on (from, relation, to)."""
        existing = self._store.query(
            TABLE,
            {"from_id": from_id, "relation": relation, "to_id": to_id},
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
        return self._store.query(TABLE, {"from_id": entity_id}, limit=10_000)

    def children(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        """Edges pointing AT this entity (things derived from it)."""
        return self._store.query(TABLE, {"to_id": entity_id}, limit=10_000)

    def ancestors(self, entity_type: str, entity_id: str, *, max_depth: int = 32) -> list[dict[str, Any]]:
        """BFS over parents(); cycle-safe, depth-capped."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        frontier = [entity_id]
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            nxt: list[str] = []
            for fid in frontier:
                for edge in self.parents(entity_type, fid):
                    eid = edge["id"]
                    if eid in seen:
                        continue
                    seen.add(eid)
                    out.append(edge)
                    to_id = edge.get("to_id")
                    if to_id and to_id not in (nxt or []):
                        nxt.append(to_id)
            frontier = nxt
        return out

    def descendants(self, entity_type: str, entity_id: str, *, max_depth: int = 32) -> list[dict[str, Any]]:
        """BFS over children(); cycle-safe, depth-capped."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        frontier = [entity_id]
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            nxt = []
            for fid in frontier:
                for edge in self.children(entity_type, fid):
                    eid = edge["id"]
                    if eid in seen:
                        continue
                    seen.add(eid)
                    out.append(edge)
                    from_id = edge.get("from_id")
                    if from_id:
                        nxt.append(from_id)
            frontier = nxt
        return out

    def trace(self, entity_type: str, entity_id: str) -> dict[str, Any]:
        """The formatted ancestry chain for CLI/reporting (§38)."""
        chain = []
        current_type, current_id = entity_type, entity_id
        visited = {current_id}
        while True:
            edges = [e for e in self.parents(current_type, current_id) if e["to_id"] not in visited]
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
            visited.add(edge["to_id"])
            current_type, current_id = edge["to_type"], edge["to_id"]
        return {"root": {"type": entity_type, "id": entity_id}, "chain": chain}
