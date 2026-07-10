"""Query interface for ROSClaw Practice closed-loop data.

``PracticeQuery`` provides a unified read API across the local Practice Catalog
(L1) and the SeekDB Knowledge Plane (L2). It is used by
``rosclaw practice query`` and by agent/runtime consumers that need to ask
"Why did this episode fail?" or "Which candidates are ready for promotion?".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rosclaw.memory.seekdb_client import SeekDBClient, SeekDBMemoryClient
from rosclaw.practice.storage.catalog import PracticeCatalog
from rosclaw.practice.storage.layout import PracticeLayout

logger = logging.getLogger("rosclaw.practice.query")


class PracticeQuery:
    """Query practice episodes, failures, cognition, candidates, and interventions."""

    def __init__(
        self,
        data_root: Path | str,
        *,
        seekdb_client: SeekDBClient | None = None,
        layout: PracticeLayout | None = None,
    ):
        self._data_root = Path(data_root)
        self._layout = layout or PracticeLayout(self._data_root)
        self._client = seekdb_client or SeekDBMemoryClient()
        self._owns_connection = not self._client.is_connected()
        if self._owns_connection:
            self._client.connect()

    def close(self) -> None:
        if self._owns_connection:
            self._client.disconnect()

    def __enter__(self) -> PracticeQuery:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Catalog (L1) queries
    # ------------------------------------------------------------------

    def list_episodes(
        self,
        body_id: str | None = None,
        skill_id: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query recorded practice episodes from the local catalog or SeekDB.

        The catalog v2 ``practice_episodes`` table is authoritative when present.
        If it is empty or unavailable, fall back to the SeekDB ``episodes`` table
        so that ingested sessions remain queryable without catalog v2 writes.
        """
        catalog = PracticeCatalog(self._layout.catalog_db_path)
        try:
            rows = catalog.list_episodes(
                body_id=body_id,
                skill_id=skill_id,
                outcome=outcome,
                limit=limit,
            )
            if rows:
                return rows
        except Exception as exc:  # noqa: BLE001
            logger.debug("Catalog episode query failed: %s", exc)
        finally:
            catalog.close()

        # Fallback: SeekDB episodes table stores body_id/skill_id in JSON metadata.
        filters: dict[str, Any] = {}
        if outcome:
            filters["outcome"] = outcome
        candidates = self._client.query(
            "episodes",
            filters=filters if filters else None,
            order_by="-started_at",
            limit=limit * 4,
        )
        results: list[dict[str, Any]] = []
        for ep in candidates:
            meta = ep.get("metadata") or {}
            if isinstance(meta, str):
                import json

                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            if body_id and meta.get("body_id") != body_id:
                continue
            if skill_id and meta.get("skill_id") != skill_id:
                continue
            merged = dict(ep)
            merged.update(meta)
            merged.setdefault("episode_id", ep.get("id"))
            merged.setdefault("body_id", meta.get("body_id"))
            merged.setdefault("skill_id", meta.get("skill_id"))
            results.append(merged)
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # SeekDB (L2) queries
    # ------------------------------------------------------------------

    def list_failures(
        self,
        body_id: str | None = None,
        failure_type: str | None = None,
        robot_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query distilled failure records from SeekDB."""
        filters: dict[str, Any] = {}
        if failure_type:
            filters["failure_type"] = failure_type
        if robot_id:
            filters["robot_id"] = robot_id
        if body_id:
            filters["body_id"] = body_id
        return self._client.query(
            "failures",
            filters=filters if filters else None,
            order_by="-timestamp",
            limit=limit,
        )

    def list_how_interventions(
        self,
        failure_type: str | None = None,
        failure_id: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query how-intervention records from SeekDB.

        ``failure_type`` filters by looking up the linked failure record and
        checking its ``failure_type`` field.
        """
        filters: dict[str, Any] = {}
        if failure_id:
            filters["failure_id"] = failure_id
        if outcome:
            filters["outcome"] = outcome

        interventions = self._client.query(
            "how_interventions",
            filters=filters if filters else None,
            order_by="-timestamp",
            limit=limit,
        )

        if failure_type:
            failure_ids = {i.get("failure_id") for i in interventions if i.get("failure_id")}
            failures: dict[str, dict[str, Any]] = {}
            for fid in failure_ids:
                if not isinstance(fid, str):
                    continue
                rows = self._client.query("failures", filters={"id": fid}, limit=1)
                if rows:
                    failures[fid] = rows[0]
            interventions = [
                i
                for i in interventions
                if failures.get(i.get("failure_id") or "", {}).get("failure_type") == failure_type
            ]
        return interventions

    def list_body_cognition(
        self,
        body_id: str | None = None,
        cognition_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query distilled body-cognition records from SeekDB."""
        filters: dict[str, Any] = {}
        if body_id:
            filters["body_id"] = body_id
        if cognition_type:
            filters["cognition_type"] = cognition_type
        return self._client.query(
            "body_cognition",
            filters=filters if filters else None,
            order_by="-timestamp",
            limit=limit,
        )

    def list_sim2real_deltas(
        self,
        body_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query sim2real delta records from SeekDB."""
        filters: dict[str, Any] = {"body_id": body_id} if body_id else {}
        return self._client.query(
            "sim2real_deltas",
            filters=filters if filters else None,
            order_by="-timestamp",
            limit=limit,
        )

    def list_candidates(
        self,
        skill_id: str | None = None,
        status: str | None = None,
        policy_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query candidate policy records from SeekDB."""
        filters: dict[str, Any] = {}
        if skill_id:
            filters["skill_id"] = skill_id
        if status:
            filters["status"] = status
        if policy_id:
            filters["policy_id"] = policy_id
        return self._client.query(
            "skill_candidates",
            filters=filters if filters else None,
            order_by="-timestamp",
            limit=limit,
        )

    def list_promotion_results(
        self,
        candidate_id: str | None = None,
        policy_id: str | None = None,
        passed: bool | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query promotion gate results from SeekDB."""
        filters: dict[str, Any] = {}
        if candidate_id:
            filters["candidate_id"] = candidate_id
        if policy_id:
            filters["policy_id"] = policy_id
        if passed is not None:
            filters["passed"] = 1 if passed else 0
        return self._client.query(
            "promotion_results",
            filters=filters if filters else None,
            order_by="-timestamp",
            limit=limit,
        )

    def explain_failure(
        self,
        failure_id: str,
    ) -> dict[str, Any] | None:
        """Return a failure plus its linked how-interventions."""
        failures = self._client.query("failures", filters={"id": failure_id}, limit=1)
        if not failures:
            return None
        failure = dict(failures[0])
        failure["interventions"] = self._client.query(
            "how_interventions",
            filters={"failure_id": failure_id},
            order_by="-timestamp",
            limit=100,
        )
        return failure

    def explain_episode(
        self,
        episode_id: str,
    ) -> dict[str, Any]:
        """Return a snapshot of everything SeekDB knows about an episode."""
        return {
            "episode_id": episode_id,
            "failures": self._client.query(
                "failures", filters={"episode_id": episode_id}, order_by="-timestamp", limit=100
            ),
            "how_interventions": self._client.query(
                "how_interventions",
                filters={"episode_id": episode_id},
                order_by="-timestamp",
                limit=100,
            ),
            "body_cognition": self._client.query(
                "body_cognition",
                filters={"episode_id": episode_id},
                order_by="-timestamp",
                limit=10,
            ),
            "sim2real_deltas": self._client.query(
                "sim2real_deltas",
                filters={"episode_id": episode_id},
                order_by="-timestamp",
                limit=100,
            ),
            "skill_candidates": self._client.query(
                "skill_candidates",
                filters={"episode_id": episode_id},
                order_by="-timestamp",
                limit=100,
            ),
            "promotion_results": self._client.query(
                "promotion_results",
                filters={"episode_id": episode_id},
                order_by="-timestamp",
                limit=100,
            ),
        }
