"""SeekDB ingestion for distilled ROSClaw Practice results.

``SeekDBIngestor`` takes the output of ``PracticeDistiller`` (or distill a
practice on demand) and writes the structured knowledge into the SeekDB
Knowledge Plane. All writes are idempotent: repeating an ingestion with the
same episode/failure/candidate/promotion IDs updates the existing records.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.memory.seekdb_client import SeekDBClient, SeekDBMemoryClient
from rosclaw.practice.distiller import DistillationResult, PracticeDistiller
from rosclaw.practice.storage.catalog import PracticeCatalog
from rosclaw.practice.storage.layout import PracticeLayout

logger = logging.getLogger("rosclaw.practice.seekdb_ingestor")


@dataclass
class IngestionReport:
    """Result of ingesting one practice session into SeekDB."""

    practice_id: str
    episode_id: str | None
    success: bool = True
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def total_records(self) -> int:
        return sum(self.table_counts.values())


def _iso_to_timestamp(value: str | float | int | None) -> float:
    """Best-effort conversion of an ISO timestamp to Unix seconds."""
    if value is None:
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)
    try:
        from datetime import UTC, datetime

        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=UTC).timestamp()
    except Exception:
        return time.time()


class SeekDBIngestor:
    """Ingest distilled practice results into SeekDB.

    Usage:
        ingestor = SeekDBIngestor("/data/rosclaw/practice")
        report = ingestor.ingest_practice("prac_...")
        ingestor.close()
    """

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

    def __enter__(self) -> SeekDBIngestor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ingest_practice(
        self,
        practice_id: str,
        *,
        distillation_result: DistillationResult | None = None,
        robot_id: str | None = None,
        task_id: str | None = None,
    ) -> IngestionReport:
        """Ingest one practice session and its distilled knowledge into SeekDB."""
        catalog = PracticeCatalog(self._layout.catalog_db_path)
        try:
            practice = catalog.get_practice(practice_id)
            if practice is None:
                raise ValueError(f"practice {practice_id} not found in catalog")

            session_id = practice.get("session_id") or practice_id
            episode_id = practice.get("episode_id")
            robot_id = robot_id or practice.get("robot_id") or "unknown"
            task_id = task_id or practice.get("task_id") or practice.get("task_name")

            if distillation_result is None:
                distiller = PracticeDistiller(self._data_root)
                distillation_result = distiller.distill(practice_id, write_artifacts=False)

            report = IngestionReport(
                practice_id=practice_id,
                episode_id=episode_id or distillation_result.episode_id,
            )

            # Episode summary
            try:
                self._ingest_episode_summary(practice, robot_id, task_id)
                report.table_counts["episodes"] = 1
            except Exception as exc:
                logger.warning("Failed to ingest episode summary: %s", exc)
                report.errors.append(f"episodes: {exc}")

            # Failures
            for payload in distillation_result.failures:
                try:
                    self.ingest_failure(
                        payload,
                        robot_id,
                        episode_id,
                        task_id,
                        body_id=distillation_result.body_cognition.get("body_id"),
                    )
                    report.table_counts["failures"] = report.table_counts.get("failures", 0) + 1
                except Exception as exc:
                    logger.warning("Failed to ingest failure: %s", exc)
                    report.errors.append(f"failure {payload.get('failure_id')}: {exc}")

            # How interventions
            for payload in distillation_result.how_interventions:
                try:
                    self.ingest_how_intervention(payload, robot_id, episode_id, task_id)
                    report.table_counts["how_interventions"] = (
                        report.table_counts.get("how_interventions", 0) + 1
                    )
                except Exception as exc:
                    logger.warning("Failed to ingest how_intervention: %s", exc)
                    report.errors.append(
                        f"how_intervention {payload.get('intervention_id')}: {exc}"
                    )

            # Body cognition
            if distillation_result.body_cognition:
                try:
                    self.ingest_body_cognition(
                        distillation_result.body_cognition,
                        robot_id,
                        session_id,
                        episode_id,
                    )
                    report.table_counts["body_cognition"] = 1
                except Exception as exc:
                    logger.warning("Failed to ingest body_cognition: %s", exc)
                    report.errors.append(f"body_cognition: {exc}")

            # Sim2real deltas
            for payload in distillation_result.sim2real_deltas:
                try:
                    self.ingest_sim2real_delta(payload, robot_id, episode_id)
                    report.table_counts["sim2real_deltas"] = (
                        report.table_counts.get("sim2real_deltas", 0) + 1
                    )
                except Exception as exc:
                    logger.warning("Failed to ingest sim2real_delta: %s", exc)
                    report.errors.append(f"sim2real_delta {payload.get('delta_id')}: {exc}")

            # Candidate policies
            for payload in distillation_result.candidates:
                try:
                    self.ingest_candidate_policy(payload, robot_id, episode_id)
                    report.table_counts["skill_candidates"] = (
                        report.table_counts.get("skill_candidates", 0) + 1
                    )
                except Exception as exc:
                    logger.warning("Failed to ingest candidate policy: %s", exc)
                    report.errors.append(f"candidate {payload.get('candidate_id')}: {exc}")

            # Promotion results
            for payload in distillation_result.promotion_results:
                try:
                    self.ingest_promotion_result(payload, robot_id, episode_id)
                    report.table_counts["promotion_results"] = (
                        report.table_counts.get("promotion_results", 0) + 1
                    )
                except Exception as exc:
                    logger.warning("Failed to ingest promotion result: %s", exc)
                    report.errors.append(f"promotion {payload.get('promotion_id')}: {exc}")

            # Candidate policies may be promoted/rejected by promotion results in
            # the same episode. Update their status after promotions are ingested.
            for payload in distillation_result.promotion_results:
                candidate_id = payload.get("candidate_id")
                if not candidate_id:
                    continue
                status = "promoted" if payload.get("passed") else "rejected"
                try:
                    self._client.update("skill_candidates", candidate_id, {"status": status})
                except Exception as exc:
                    logger.warning("Failed to update candidate status: %s", exc)

            # Mark catalog as committed
            try:
                catalog.update_practice(practice_id, {"seekdb_committed": 1})
            except Exception as exc:
                logger.warning("Failed to update catalog seekdb_committed: %s", exc)
                report.errors.append(f"catalog_commit: {exc}")

            if report.errors:
                report.success = any(
                    count > 0 for count in report.table_counts.values()
                ) and not all(err.startswith("catalog_commit") for err in report.errors)
            return report
        finally:
            catalog.close()

    def _ingest_episode_summary(
        self,
        practice: dict[str, Any],
        robot_id: str,
        task_id: str | None,
    ) -> str:
        episode_id = practice.get("episode_id") or practice.get("practice_id")
        started_at = practice.get("start_time")
        ended_at = practice.get("end_time")
        outcome = practice.get("outcome", "unknown")
        record = {
            "id": episode_id,
            "task_id": task_id,
            "robot_id": robot_id,
            "started_at": _iso_to_timestamp(started_at) if started_at else time.time(),
            "ended_at": _iso_to_timestamp(ended_at) if ended_at else None,
            "outcome": outcome,
            "artifact_uri": practice.get("events_jsonl_path"),
            "metadata": {
                "practice_id": practice.get("practice_id"),
                "session_id": practice.get("session_id"),
                "robot_type": practice.get("robot_type"),
                "task_name": practice.get("task_name"),
                "skill_id": practice.get("skill_id"),
                "duration_ms": practice.get("duration_ms"),
                "reward": practice.get("reward"),
            },
        }
        return self._client.insert("episodes", record)

    def ingest_failure(
        self,
        payload: dict[str, Any],
        robot_id: str,
        episode_id: str | None,
        task_id: str | None = None,
        body_id: str | None = None,
    ) -> str:
        record = {
            "id": payload.get("failure_id") or f"fail_{int(time.time() * 1000)}",
            "episode_id": episode_id,
            "task_id": task_id,
            "robot_id": robot_id,
            "body_id": body_id or payload.get("body_id"),
            "failure_type": payload.get("failure_type", "unknown"),
            "root_cause": payload.get("description", "") or payload.get("root_cause", ""),
            "timestamp": _iso_to_timestamp(payload.get("timestamp")),
            "recovery_hint": (
                payload.get("suggested_fix", {}).get("description")
                if isinstance(payload.get("suggested_fix"), dict)
                else None
            )
            or payload.get("recovery_hint", ""),
            "metadata": payload,
        }
        return self._client.insert("failures", record)

    def ingest_how_intervention(
        self,
        payload: dict[str, Any],
        robot_id: str,
        episode_id: str | None,
        task_id: str | None = None,
    ) -> str:
        record = {
            "id": payload.get("intervention_id") or f"how_{int(time.time() * 1000)}",
            "failure_id": payload.get("failure_id"),
            "episode_id": payload.get("episode_id") or episode_id,
            "robot_id": robot_id,
            "task_id": task_id,
            "intervention_type": payload.get("intervention_type")
            or payload.get("description", "").split(" ")[0].lower()
            or "unknown",
            "description": payload.get("description", ""),
            "action_taken": payload.get("action_taken", {}),
            "outcome": payload.get("outcome", "pending"),
            "timestamp": _iso_to_timestamp(payload.get("timestamp")),
            "metadata": payload,
        }
        return self._client.insert("how_interventions", record)

    def ingest_body_cognition(
        self,
        cognition: dict[str, Any],
        robot_id: str,
        session_id: str | None,
        episode_id: str | None,
    ) -> str:
        body_id = cognition.get("body_id") or "unknown"
        cognition_type = cognition.get("cognition_type") or "body_model"
        cognition_id = f"cog:{body_id}:{episode_id or session_id or 'unknown'}:{cognition_type}"
        metadata = dict(cognition.get("metadata", {}))
        if cognition.get("cognition_id"):
            metadata["source_cognition_id"] = cognition["cognition_id"]
        record = {
            "id": cognition_id,
            "body_id": body_id,
            "robot_id": robot_id,
            "episode_id": episode_id,
            "session_id": session_id,
            "cognition_type": cognition_type,
            "data": cognition.get("data")
            or {
                "known_traits": cognition.get("known_traits", []),
                "force_model": cognition.get("force_model", {}),
                "thermal_limits": cognition.get("thermal_limits", {}),
                "contact_distribution": cognition.get("contact_distribution", {}),
            },
            "timestamp": _iso_to_timestamp(
                cognition.get("updated_at") or cognition.get("timestamp")
            ),
            "metadata": metadata,
        }
        return self._client.insert("body_cognition", record)

    def ingest_sim2real_delta(
        self,
        payload: dict[str, Any],
        robot_id: str,
        episode_id: str | None,
    ) -> str:
        record = {
            "id": payload.get("delta_id") or f"delta_{int(time.time() * 1000)}",
            "body_id": payload.get("body_id") or "unknown",
            "robot_id": robot_id,
            "episode_id": episode_id,
            "dofs": payload.get("dofs", []),
            "sim_value": payload.get("sim_value", {}),
            "real_value": payload.get("real_value", {}),
            "delta": payload.get("delta", {}),
            "unit": payload.get("unit", ""),
            "timestamp": _iso_to_timestamp(payload.get("timestamp")),
            "metadata": payload.get("metadata", {}),
        }
        return self._client.insert("sim2real_deltas", record)

    def ingest_candidate_policy(
        self,
        payload: dict[str, Any],
        robot_id: str,
        episode_id: str | None,
    ) -> str:
        status = "candidate"
        promoted = payload.get("promoted")
        if promoted is True:
            status = "promoted"
        elif promoted is False:
            status = "rejected"
        record = {
            "id": payload.get("candidate_id") or f"cand_{int(time.time() * 1000)}",
            "skill_id": payload.get("skill_id"),
            "robot_id": robot_id,
            "episode_id": episode_id,
            "policy_id": payload.get("policy_id"),
            "policy_type": payload.get("policy_type", "unknown"),
            "policy_params": payload.get("policy_params", {}),
            "metrics": payload.get("metrics", {}),
            "status": status,
            "evidence_refs": payload.get("evidence_refs", []),
            "timestamp": _iso_to_timestamp(payload.get("timestamp")),
            "metadata": payload.get("metadata", {}),
        }
        return self._client.insert("skill_candidates", record)

    def ingest_promotion_result(
        self,
        payload: dict[str, Any],
        robot_id: str,
        episode_id: str | None,
    ) -> str:
        record = {
            "id": payload.get("promotion_id") or f"promo_{int(time.time() * 1000)}",
            "candidate_id": payload.get("candidate_id"),
            "policy_id": payload.get("policy_id"),
            "robot_id": robot_id,
            "episode_id": episode_id,
            "gate_name": payload.get("gate_name", "unknown"),
            "passed": 1 if payload.get("passed") else 0,
            "metrics": payload.get("metrics", {}),
            "failures": payload.get("failures", []),
            "evidence_refs": payload.get("evidence_refs", []),
            "promoted_policy_ref": payload.get("promoted_policy_ref"),
            "timestamp": _iso_to_timestamp(payload.get("timestamp")),
            "metadata": payload.get("metadata", {}),
        }
        return self._client.insert("promotion_results", record)

    def ingest_artifact_ref(
        self,
        artifact_id: str,
        artifact_type: str,
        uri: str,
        episode_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store an artifact reference in SeekDB's artifacts table."""
        record = {
            "id": artifact_id,
            "episode_id": episode_id,
            "artifact_type": artifact_type,
            "uri": uri,
            "size_bytes": None,
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        return self._client.insert("artifacts", record)
