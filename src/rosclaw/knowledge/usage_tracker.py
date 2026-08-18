"""KnowledgeUsageTracker (PR-DF-10 / flywheel §29-31).

Closes the ReferencePack → Advice → Action → Receipt → Feedback loop that
previously only existed in CLI/test paths:

    facade.reference_pack() / facade.advise()   (events observed here)
        ↓  agent acts
    rosclaw.critic.judgment                      (verified outcome)
        ↓
    KnowledgeUsageFeedback  (conservative verdict, §31)
        ↓  facade.feedback()

Verdict discipline (§31 — conservative by construction):
  * ``useful``  — knowledge was presented AND the task verified SUCCESS;
  * ``unknown`` — task failed or the outcome is unattributable.  Automatic
    feedback NEVER emits ``misleading``/``stale``/``incompatible``: those
    require verifier attribution, human feedback, or explicit counterfactual
    evidence, none of which an automated tracker can assert.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from .feedback_adapter import build_usage_feedback

logger = logging.getLogger("rosclaw.knowledge.usage_tracker")

_PACK_TOPIC = "know.reference_pack.created"
_ADVICE_TOPIC = "how.advice.created"
_OUTCOME_TOPIC = "rosclaw.critic.judgment"


class KnowledgeUsageTracker:
    """Track pack/advice usage and emit conservative feedback on outcomes."""

    def __init__(
        self,
        event_bus: Any,
        facade: Any,
        *,
        ttl_s: float = 3600.0,
        max_tracked: int = 1000,
        usage_ledger: Any = None,
    ) -> None:
        self._bus = event_bus
        self._facade = facade
        self._ttl_s = ttl_s
        self._max_tracked = max_tracked
        # PR-DF-21 (phase-II §32): durable usage observation.  None keeps
        # the DF-10 in-memory-only behavior (no data plane present).
        self._ledger = usage_ledger
        self._open: dict[str, dict[str, Any]] = {}
        self._subscribed = False

    # -- lifecycle ------------------------------------------------------

    def subscribe(self) -> None:
        if self._bus is None or self._subscribed:
            return
        self._bus.subscribe(_PACK_TOPIC, self._on_pack)
        self._bus.subscribe(_ADVICE_TOPIC, self._on_advice)
        self._bus.subscribe(_OUTCOME_TOPIC, self._on_outcome)
        self._subscribed = True
        logger.info("KnowledgeUsageTracker subscribed")

    def unsubscribe(self) -> None:
        if self._bus is None or not self._subscribed:
            return
        self._bus.unsubscribe(_PACK_TOPIC, self._on_pack)
        self._bus.unsubscribe(_ADVICE_TOPIC, self._on_advice)
        self._bus.unsubscribe(_OUTCOME_TOPIC, self._on_outcome)
        self._subscribed = False

    # -- tracking ---------------------------------------------------------

    def _on_pack(self, event: Any) -> None:
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return
        pack_id = payload.get("reference_pack_id")
        if not pack_id:
            return
        if len(self._open) >= self._max_tracked:
            oldest = min(self._open, key=lambda k: self._open[k]["created_at"])
            del self._open[oldest]
        self._open[pack_id] = {
            "reference_pack_id": pack_id,
            "knowledge_unit_ids": list(payload.get("knowledge_unit_ids") or []),
            "advice_id": None,
            "created_at": time.time(),
            "trace_id": str(payload.get("trace_id") or getattr(event, "trace_id", "") or ""),
            "robot_id": str(payload.get("robot_id") or ""),
            "body_id": str(payload.get("body_id") or ""),
            "task_id": str(payload.get("task_id") or ""),
            "skill_id": str(payload.get("skill_id") or ""),
        }
        if self._ledger is not None:
            for unit_id in self._open[pack_id]["knowledge_unit_ids"] or ["unknown_unit"]:
                self._ledger.record(
                    "presented",
                    reference_pack_id=pack_id,
                    knowledge_unit_id=unit_id,
                    trace_id=self._open[pack_id]["trace_id"],
                    robot_id=self._open[pack_id]["robot_id"],
                    body_id=self._open[pack_id]["body_id"],
                    task_id=self._open[pack_id]["task_id"],
                    skill_id=self._open[pack_id]["skill_id"],
                )

    def _on_advice(self, event: Any) -> None:
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return
        pack_id = payload.get("reference_pack_id")
        advice_id = payload.get("advice_id")
        if pack_id and pack_id in self._open and advice_id:
            self._open[pack_id]["advice_id"] = advice_id
            usage = self._open[pack_id]
            if self._ledger is not None:
                for unit_id in usage["knowledge_unit_ids"] or ["unknown_unit"]:
                    self._ledger.record(
                        "used",
                        reference_pack_id=pack_id,
                        knowledge_unit_id=unit_id,
                        advice_id=advice_id,
                        trace_id=usage["trace_id"],
                        robot_id=usage["robot_id"],
                        body_id=usage["body_id"],
                        task_id=usage["task_id"],
                        skill_id=usage["skill_id"],
                    )

    # -- feedback ---------------------------------------------------------

    def _on_outcome(self, event: Any) -> None:
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return
        status = str(payload.get("status", "UNKNOWN"))
        episode_id = payload.get("episode_id")
        practice_id = payload.get("practice_id")
        verdict = "useful" if status == "SUCCESS" else "unknown"
        now = time.time()
        for pack_id, usage in list(self._open.items()):
            if now - usage["created_at"] > self._ttl_s:
                del self._open[pack_id]
                continue
            unit_ids = usage["knowledge_unit_ids"] or ["unknown_unit"]
            for unit_id in unit_ids:
                self._submit(usage, unit_id, verdict, episode_id, practice_id)
                if self._ledger is not None:
                    self._ledger.record(
                        verdict,
                        reference_pack_id=pack_id,
                        knowledge_unit_id=unit_id,
                        advice_id=usage.get("advice_id"),
                        trace_id=usage["trace_id"],
                        practice_id=practice_id,
                        episode_id=episode_id,
                        robot_id=usage["robot_id"],
                        body_id=usage["body_id"],
                        task_id=usage["task_id"],
                        skill_id=usage["skill_id"],
                    )
            del self._open[pack_id]

    def _submit(
        self,
        usage: dict[str, Any],
        unit_id: str,
        verdict: str,
        episode_id: Any,
        practice_id: Any,
    ) -> None:
        try:
            feedback = build_usage_feedback(
                reference_pack_id=usage["reference_pack_id"],
                knowledge_unit_id=unit_id,
                advice_id=usage.get("advice_id"),
                context_hash=hashlib.sha256(
                    f"{usage['reference_pack_id']}:{episode_id}:{unit_id}".encode()
                ).hexdigest(),
                verdict=verdict,  # type: ignore[arg-type]
                presented=True,
                used_by_agent=True,
                receipt_ref=str(episode_id) if episode_id else None,
                practice_ref=str(practice_id) if practice_id else None,
                origin="verifier",
                reason=None if verdict == "useful" else "outcome unattributed (auto feedback is conservative)",
            )
            created = self._facade.feedback(feedback)
            logger.info(
                "Knowledge usage feedback: pack=%s unit=%s verdict=%s created=%s",
                usage["reference_pack_id"],
                unit_id,
                verdict,
                created,
            )
        except Exception as exc:  # noqa: BLE001 — feedback must never break execution
            logger.info("Knowledge usage feedback failed (non-fatal): %s", exc)

    @property
    def tracked_count(self) -> int:
        return len(self._open)
