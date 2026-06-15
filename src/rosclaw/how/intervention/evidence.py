"""EvidenceTrace generation and collection for How interventions."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from rosclaw.schemas import EvidenceTrace

logger = logging.getLogger("rosclaw.how.intervention.evidence")


class EvidenceCollector:
    """Collects evidence traces from How interventions."""

    def __init__(self, event_bus: Any | None = None, seekdb_client: Any | None = None):
        self._bus = event_bus
        self._seekdb = seekdb_client
        self._traces: list[EvidenceTrace] = []

    def record(
        self,
        injection_id: str,
        pattern_id: str,
        run_id: str,
        task_name: str,
        pre_score: float,
        post_score_1: float,
        post_score_3: float,
        code_diff_summary: list[str] | None = None,
        used_hint: bool = False,
        verifier_status: str = "",
        objective_direction: str = "maximize",
    ) -> EvidenceTrace:
        """Record a new evidence trace."""
        trace = EvidenceTrace(
            injection_id=injection_id,
            pattern_id=pattern_id,
            run_id=run_id,
            task_name=task_name,
            pre_score=pre_score,
            post_score_1=post_score_1,
            post_score_3=post_score_3,
            code_diff_summary=code_diff_summary or [],
            used_hint=used_hint,
            verifier_status=verifier_status,
            objective_direction=objective_direction,  # type: ignore[arg-type]
        )
        self._traces.append(trace)

        # Write to SeekDB
        if self._seekdb is not None:
            try:
                self._seekdb.insert("evidence_traces", trace.to_dict())
            except Exception as exc:
                logger.warning("SeekDB evidence write failed: %s", exc)

        # Publish event
        if self._bus is not None:
            try:
                from rosclaw.core.event_bus import Event, EventPriority
                self._bus.publish(Event(
                    topic="rosclaw.how.evidence.generated",
                    payload=trace.to_dict(),
                    source="rosclaw-how",
                    priority=EventPriority.NORMAL,
                ))
            except Exception as exc:
                logger.warning("EventBus evidence publish failed: %s", exc)

        logger.info("Evidence recorded: injection=%s pattern=%s delta=%.3f",
                    injection_id, pattern_id, trace.score_delta)
        return trace

    def list_traces(self, run_id: str | None = None, task_name: str | None = None) -> list[EvidenceTrace]:
        traces = list(self._traces)
        if run_id:
            traces = [t for t in traces if t.run_id == run_id]
        if task_name:
            traces = [t for t in traces if t.task_name == task_name]
        return traces


def generate_injection_id() -> str:
    """Generate a unique injection_id for How interventions."""
    return f"inj_{uuid.uuid4().hex[:8]}"
