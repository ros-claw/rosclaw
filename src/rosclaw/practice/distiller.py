"""Practice distillation: turn raw events into structured knowledge.

``PracticeDistiller`` reads a practice session's events and produces distilled
artifacts such as body cognition, failure summaries, how-interventions,
candidate policies, and promotion results. These artifacts can then be ingested
into SeekDB by ``SeekDBIngestor``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.practice.artifact_store import ArtifactStore
from rosclaw.practice.ids import generate_asset_id, generate_candidate_id
from rosclaw.practice.storage.catalog import PracticeCatalog
from rosclaw.practice.storage.layout import PracticeLayout

logger = logging.getLogger("rosclaw.practice.distiller")


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class DistillationResult:
    practice_id: str
    session_id: str
    episode_id: str | None
    body_cognition: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    how_interventions: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    promotion_results: list[dict[str, Any]] = field(default_factory=list)
    sim2real_deltas: list[dict[str, Any]] = field(default_factory=list)
    artifact_refs: dict[str, str] = field(default_factory=dict)


class PracticeDistiller:
    """Distill raw practice events into knowledge artifacts."""

    def __init__(self, data_root: Path | str):
        self._data_root = Path(data_root)
        self._layout = PracticeLayout(self._data_root)
        self._artifact_store = ArtifactStore(self._data_root, layout=self._layout)

    def distill(
        self,
        practice_id: str,
        *,
        body_id: str | None = None,
        write_artifacts: bool = True,
    ) -> DistillationResult:
        """Distill one practice session."""
        catalog = PracticeCatalog(self._layout.catalog_db_path)
        try:
            practice = catalog.get_practice(practice_id)
            if practice is None:
                raise ValueError(f"practice {practice_id} not found in catalog")

            session_id = practice.get("session_id") or practice_id
            episode_id = practice.get("episode_id")
            events_path = Path(practice.get("events_jsonl_path", ""))
            events = self._load_events(events_path)

            result = self._distill_events(
                practice_id, session_id, episode_id, events, body_id=body_id
            )

            if write_artifacts:
                self._write_distilled_artifacts(result)

            return result
        finally:
            catalog.close()

    @staticmethod
    def _load_events(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSONL line: %s", e)
        return events

    def _distill_events(
        self,
        practice_id: str,
        session_id: str,
        episode_id: str | None,
        events: list[dict[str, Any]],
        body_id: str | None,
    ) -> DistillationResult:
        result = DistillationResult(
            practice_id=practice_id,
            session_id=session_id,
            episode_id=episode_id,
        )

        contact_distribution: dict[str, int] = {}
        force_nets: dict[str, list[float]] = {}
        temperatures: dict[str, list[float]] = {}
        body_id = body_id or self._infer_body_id(events)

        for ev in events:
            event_type = ev.get("event_type")
            payload = ev.get("payload", {})

            if event_type == "physical_feedback_event":
                primary = payload.get("primary_event", "unknown")
                contact_distribution[primary] = contact_distribution.get(primary, 0) + 1
                for dof, value in (payload.get("force_net") or {}).items():
                    if value is not None:
                        force_nets.setdefault(dof, []).append(value)
                for dof, value in (payload.get("temperature") or {}).items():
                    if value is not None:
                        temperatures.setdefault(dof, []).append(value)

            elif event_type == "contact_event":
                contact_distribution[payload.get("event_type", "unknown")] = (
                    contact_distribution.get(payload.get("event_type", "unknown"), 0) + 1
                )

            elif event_type == "failure_event":
                result.failures.append(payload)

            elif event_type == "how_intervention_event":
                result.how_interventions.append(payload)

            elif event_type == "candidate_policy_event":
                result.candidates.append(payload)

            elif event_type == "promotion_result_event":
                result.promotion_results.append(payload)

            elif event_type == "sim2real_delta_event":
                result.sim2real_deltas.append(payload)

        result.body_cognition = self._build_body_cognition(
            body_id, contact_distribution, force_nets, temperatures
        )
        return result

    @staticmethod
    def _infer_body_id(events: list[dict[str, Any]]) -> str | None:
        for ev in events:
            body_id = ev.get("body_id")
            if body_id:
                return body_id
            payload = ev.get("payload", {})
            body_id = payload.get("body_id")
            if body_id:
                return body_id
        return None

    @staticmethod
    def _build_body_cognition(
        body_id: str | None,
        contact_distribution: dict[str, int],
        force_nets: dict[str, list[float]],
        temperatures: dict[str, list[float]],
    ) -> dict[str, Any]:
        force_model: dict[str, Any] = {}
        for dof, values in force_nets.items():
            if values:
                force_model[dof] = {
                    "mean": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                    "count": len(values),
                }

        thermal_limits: dict[str, float] = {}
        for dof, values in temperatures.items():
            if values:
                thermal_limits[dof] = max(values)

        known_traits: list[str] = []
        if "no_contact" in contact_distribution:
            known_traits.append("no_contact_region")
        if "desired_contact" in contact_distribution:
            known_traits.append("desired_contact_region")
        if "over_contact" in contact_distribution:
            known_traits.append("over_contact_risk")
        if "temperature_limited" in contact_distribution:
            known_traits.append("thermal_limit")

        return {
            "cognition_id": f"cog_{_utc_now_iso()}",
            "body_id": body_id,
            "schema_version": "rosclaw.body.cognition.v1",
            "updated_at": _utc_now_iso(),
            "known_traits": known_traits,
            "force_model": force_model,
            "thermal_limits": thermal_limits,
            "contact_distribution": contact_distribution,
        }

    def _write_distilled_artifacts(self, result: DistillationResult) -> None:
        session_id = result.session_id
        episode_id = result.episode_id

        if result.body_cognition:
            record = self._artifact_store.write_yaml(
                f"body_cognition_{result.body_cognition['cognition_id']}",
                result.body_cognition,
                session_id=session_id,
                episode_id=episode_id,
                artifact_type="body_cognition",
            )
            result.artifact_refs["body_cognition"] = record.path

        if result.failures:
            record = self._artifact_store.write_yaml(
                f"failures_{generate_asset_id()}",
                {"failures": result.failures},
                session_id=session_id,
                episode_id=episode_id,
                artifact_type="failures",
            )
            result.artifact_refs["failures"] = record.path

        if result.how_interventions:
            record = self._artifact_store.write_yaml(
                f"how_interventions_{generate_asset_id()}",
                {"interventions": result.how_interventions},
                session_id=session_id,
                episode_id=episode_id,
                artifact_type="how_interventions",
            )
            result.artifact_refs["how_interventions"] = record.path

        if result.candidates:
            record = self._artifact_store.write_yaml(
                f"candidates_{generate_candidate_id()}",
                {"candidates": result.candidates},
                session_id=session_id,
                episode_id=episode_id,
                artifact_type="candidates",
            )
            result.artifact_refs["candidates"] = record.path

        if result.promotion_results:
            record = self._artifact_store.write_yaml(
                f"promotion_results_{generate_asset_id()}",
                {"promotion_results": result.promotion_results},
                session_id=session_id,
                episode_id=episode_id,
                artifact_type="promotion_results",
            )
            result.artifact_refs["promotion_results"] = record.path

        if result.sim2real_deltas:
            record = self._artifact_store.write_yaml(
                f"sim2real_deltas_{generate_asset_id()}",
                {"deltas": result.sim2real_deltas},
                session_id=session_id,
                episode_id=episode_id,
                artifact_type="sim2real_deltas",
            )
            result.artifact_refs["sim2real_deltas"] = record.path
