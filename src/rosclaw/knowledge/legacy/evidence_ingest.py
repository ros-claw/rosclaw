"""EvidenceTrace ingest for Know module.

Subscribes to How and Auto events, updates pattern_metrics and
bridge_index with validated evidence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rosclaw.schemas import EvidenceTrace

logger = logging.getLogger("rosclaw.know.evidence_ingest")


class EvidenceIngestor:
    """Ingest EvidenceTrace into Know assets."""

    def __init__(
        self,
        assets_dir: str = "./know_assets",
        event_bus: Any | None = None,
    ):
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._bus = event_bus
        self._pattern_metrics_path = self.assets_dir / "pattern_metrics.json"
        self._bridge_index_path = self.assets_dir / "bridge_index.json"
        self._pattern_metrics: dict[str, Any] = {}
        self._load_pattern_metrics()
        if self._bus is not None:
            self._bus.subscribe("rosclaw.how.evidence.generated", self._on_how_evidence)
            self._bus.subscribe("rosclaw.auto.experiment.completed", self._on_auto_experiment)

    def _load_pattern_metrics(self) -> None:
        if self._pattern_metrics_path.exists():
            try:
                self._pattern_metrics = json.loads(self._pattern_metrics_path.read_text())
            except Exception as exc:
                logger.warning("Failed to load pattern_metrics: %s", exc)
                self._pattern_metrics = {}

    def _save_pattern_metrics(self) -> None:
        try:
            self._pattern_metrics_path.write_text(
                json.dumps(self._pattern_metrics, indent=2, ensure_ascii=False)
            )
        except Exception as exc:
            logger.warning("Failed to save pattern_metrics: %s", exc)

    def _on_how_evidence(self, event: Any) -> None:
        """Handle rosclaw.how.evidence.generated event."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        trace = EvidenceTrace.from_dict(payload)
        self.ingest(trace)

    def _on_auto_experiment(self, event: Any) -> None:
        """Handle rosclaw.auto.experiment.completed event."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        # Convert auto experiment result into EvidenceTrace shape
        trace = EvidenceTrace(
            injection_id=payload.get("experiment_id", ""),
            pattern_id=payload.get("patch_id", ""),
            run_id=payload.get("run_id", ""),
            task_name=payload.get("task_id", ""),
            pre_score=payload.get("baseline_metrics", {}).get("success_rate", 0.0),
            post_score_1=payload.get("candidate_metrics", {}).get("success_rate", 0.0),
            post_score_3=payload.get("candidate_metrics", {}).get("success_rate", 0.0),
            used_hint=False,
            verifier_status="valid" if payload.get("decision") == "promote" else "invalid",
            source="rosclaw-auto",
        )
        self.ingest(trace)

    def ingest(self, trace: EvidenceTrace) -> None:
        """Ingest a single EvidenceTrace and update pattern_metrics."""
        pattern_id = trace.pattern_id or "unknown"
        entry = self._pattern_metrics.setdefault(
            pattern_id,
            {
                "uses": 0,
                "total_delta": 0.0,
                "successes": 0,
                "failures": 0,
            },
        )
        entry["uses"] += 1
        entry["total_delta"] += trace.score_delta
        if trace.verifier_status == "valid" or trace.score_delta > 0:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
        self._save_pattern_metrics()
        logger.info("Ingested evidence for pattern=%s delta=%.3f", pattern_id, trace.score_delta)

    def get_pattern_ranking(self) -> list[tuple[str, float]]:
        """Return patterns ranked by average delta per use."""
        ranked = []
        for pid, data in self._pattern_metrics.items():
            uses = data.get("uses", 1)
            avg_delta = data.get("total_delta", 0.0) / uses
            ranked.append((pid, avg_delta))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
