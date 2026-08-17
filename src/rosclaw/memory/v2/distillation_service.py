"""MemoryDistillationService (PR-DF-16B / Phase-II §5).

Closes the loop the flywheel was missing: a NORMAL practice session close
now automatically becomes Memory 2.0 items — body / intervention / skill /
failure / episodic — not just critic judgments.

    practice.session.finished        (enriched payload, all files final)
        ↓ put_nowait (never blocks Practice, §5.6)
    single worker (order + idempotency + audit over throughput, §5.7)
        ↓ distill_session_dir (existing extractors, §5.11)
        ↓ quality policy (§5.10: WARN -> quality*0.7; FAIL -> quarantine
           except safety-critical facts kept with evidence_quality=degraded)
    WriteGate -> MemoryRepository -> StructuredStore (-> projection)
        ↓ ledger row (memory_distillation_runs) + lineage edges
        (memory --derived_from--> episode --observed_in--> practice)

The ledger answers "why did this session not become memory" (§5.9).
subscribe() reconciles queued/failed ledger rows from a previous process
(§T7); distillation is idempotent by repository content-hash + ledger key
(practice_id, source_manifest_hash).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from rosclaw.core.event_topics import EventTopics

from .distill import build_candidates, load_session_events
from .models import MemoryStatus

logger = logging.getLogger("rosclaw.memory.v2.distillation_service")

LEDGER_TABLE = "memory_distillation_runs"
_SAFETY_CRITICAL = ("collision", "overcurrent", "overtemperature", "estop", "e-stop", "emergency")


class MemoryDistillationService:
    """Session-finished -> queued -> distilled -> ledgered -> linked."""

    def __init__(
        self,
        event_bus: Any,
        repository: Any,
        gate: Any,
        lineage: Any | None = None,
        store: Any | None = None,
    ) -> None:
        self._bus = event_bus
        self._repo = repository
        self._gate = gate
        self._lineage = lineage
        self._store = store if store is not None else getattr(repository, "_client", None)
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._subscribed = False

    # -- lifecycle ------------------------------------------------------

    def subscribe(self) -> None:
        if self._subscribed:
            return
        self._bus.subscribe(EventTopics.PRACTICE_SESSION_FINISHED, self._on_session_finished)
        self._subscribed = True
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="memory-distillation-worker"
        )
        self._worker.start()
        self._reconcile_pending()
        logger.info("MemoryDistillationService subscribed (worker started)")

    def unsubscribe(self, *, drain_timeout: float = 10.0) -> None:
        if not self._subscribed:
            return
        self._bus.unsubscribe(EventTopics.PRACTICE_SESSION_FINISHED, self._on_session_finished)
        self._subscribed = False
        self.drain(timeout=drain_timeout)
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None

    def drain(self, timeout: float = 10.0) -> bool:
        """Wait until every queued session has been fully processed.

        ``queue.empty()`` is not enough: the worker dequeues a payload
        *before* distilling it, so an empty queue can still mean a session
        is mid-write (observed as a CI-only flake where drain returned
        before memory_items were stored).
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.05)
        return False

    # -- intake ---------------------------------------------------------

    def _on_session_finished(self, event: Any) -> None:
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return
        try:
            self._queue.put_nowait(payload)  # return immediately (§5.6)
        except queue.Full:
            logger.warning("distillation queue full — session %s dropped", payload.get("practice_id"))

    def _reconcile_pending(self) -> None:
        """Restart path (T7): queued/failed ledger rows re-enqueue."""
        if self._store is None:
            return
        try:
            for status in ("queued", "failed", "running"):
                for row in self._store.query(LEDGER_TABLE, {"status": status}, limit=1000):
                    session_dir = row.get("session_dir")
                    if not session_dir:
                        continue
                    logger.info("reconcile: re-enqueue distillation %s", row.get("practice_id"))
                    self._queue.put_nowait(
                        {
                            "practice_id": row.get("practice_id"),
                            "session_dir": session_dir,
                            "manifest_hash": row.get("source_manifest_hash"),
                            "_reconciled": True,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            logger.info("distillation reconcile unavailable: %s", exc)

    # -- worker -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._work(payload)
            except Exception as exc:  # noqa: BLE001 — one bad session never kills the worker
                logger.warning("distillation work failed: %s", exc)
            finally:
                self._queue.task_done()

    def _work(self, payload: dict[str, Any]) -> None:
        practice_id = payload.get("practice_id")
        session_dir = payload.get("session_dir")
        if not practice_id or not session_dir:
            return
        manifest_hash = payload.get("manifest_hash") or ""
        ledger_id = f"{practice_id}:{manifest_hash or 'nohash'}"
        started = time.time()
        self._ledger_write(
            ledger_id,
            {
                "practice_id": practice_id,
                "session_id": payload.get("session_id"),
                "episode_id": payload.get("episode_id"),
                "source_manifest_hash": manifest_hash,
                "session_dir": session_dir,
                "status": "running",
                "started_at": started,
                "attempt": self._next_attempt(ledger_id),
            },
        )
        try:
            context, events = load_session_events(session_dir)
            candidates = build_candidates(context, events)
            fact_verify = payload.get("fact_verify") or {}
            candidates = self._apply_quality_policy(candidates, fact_verify)
            result = self._apply(candidates)
            self._link(context, result)
            self._ledger_write(
                ledger_id,
                {
                    "practice_id": practice_id,
                    "status": "completed",
                    "candidate_count": len(candidates),
                    "stored_count": len(result["stored"]),
                    "merged_count": len(result["merged"]),
                    "updated_count": len(result["updated"]),
                    "ignored_count": result["ignored"],
                    "quarantined_count": result["quarantined"],
                    "finished_at": time.time(),
                    "last_error": None,
                },
            )
            logger.info(
                "distilled %s: %d candidates -> %d stored / %d merged / %d quarantined",
                practice_id,
                len(candidates),
                len(result["stored"]),
                len(result["merged"]),
                result["quarantined"],
            )
        except Exception as exc:  # noqa: BLE001 — T8: corrupt session -> ledger error, never crash
            logger.warning("distillation failed for %s: %s", practice_id, exc)
            self._ledger_write(
                ledger_id,
                {
                    "practice_id": practice_id,
                    "status": "failed",
                    "finished_at": time.time(),
                    "last_error": str(exc)[:500],
                },
            )

    # -- quality policy (§5.10) ------------------------------------------

    def _apply_quality_policy(self, candidates: list[Any], fact_verify: dict[str, Any]) -> list[Any]:
        passed = fact_verify.get("passed")
        warnings = int(fact_verify.get("warnings") or 0)
        if passed is True and warnings == 0:
            return candidates
        if passed is True:
            for c in candidates:
                c.quality_score = round(c.quality_score * 0.7, 4)
            return candidates
        # FAIL: quarantine everything except safety-critical facts, which are
        # kept with a degraded-evidence marker (never silently dropped).
        out = []
        for c in candidates:
            text = f"{c.title} {c.document}".lower()
            if any(k in text for k in _SAFETY_CRITICAL):
                c.metadata = {**c.metadata, "evidence_quality": "degraded"}
                out.append(c)
            else:
                c.status = MemoryStatus.QUARANTINED.value
                out.append(c)
        return out

    # -- gate application (mirrors distill_events, §5.11 reuse) -----------

    def _apply(self, candidates: list[Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "stored": [],
            "merged": [],
            "updated": [],
            "ignored": 0,
            "quarantined": 0,
            "items": [],
        }
        for candidate in candidates:
            try:
                self._apply_one(candidate, result)
            except ValueError as ev:
                # active-without-evidence etc. — quarantine, never crash (T8)
                logger.info("candidate quarantined on store error: %s", ev)
                candidate.status = "quarantined"
                import contextlib

                with contextlib.suppress(Exception):
                    self._repo.store(candidate)
                result["quarantined"] += 1
        return result

    def _apply_one(self, candidate: Any, result: dict[str, Any]) -> None:
        # Evidence guarantee (§5.4): attach the session-level evidence ref
        # when the extractor left the candidate bare.
        if not candidate.evidence_refs and not candidate.artifact_refs:
            session_ref = (candidate.metadata or {}).get("session_evidence_ref")
            if session_ref:
                candidate.evidence_refs = [session_ref]
        if candidate.status == MemoryStatus.QUARANTINED.value:
            self._repo.store(candidate)
            result["quarantined"] += 1
            result["items"].append(candidate)
            return
        decision = self._gate.evaluate(candidate)
        if decision.decision == "STORE":
            result["stored"].append(self._repo.store(candidate))
            result["items"].append(candidate)
        elif decision.decision == "MERGE" and decision.target_memory_id:
            if self._repo.merge_into(decision.target_memory_id, candidate):
                result["merged"].append(decision.target_memory_id)
                result["items"].append(candidate)
        elif decision.decision == "UPDATE" and decision.target_memory_id:
            result["updated"].append(self._repo.supersede(decision.target_memory_id, candidate))
            result["items"].append(candidate)
        elif decision.decision == "QUARANTINE":
            candidate.status = "quarantined"
            self._repo.store(candidate)
            result["quarantined"] += 1
            result["items"].append(candidate)
        else:
            result["ignored"] += 1

    # -- lineage (phase-II §6): derived -> source; never guess receipts ----

    def _link(self, context: Any, result: dict[str, Any]) -> None:
        if self._lineage is None:
            return
        episode_id = getattr(context, "episode_id", None)
        practice_id = getattr(context, "practice_id", None)
        try:
            for item in result["items"]:
                if episode_id:
                    self._lineage.link("memory", item.memory_id, "derived_from", "episode", episode_id)
            if episode_id and practice_id:
                self._lineage.link("episode", episode_id, "observed_in", "practice", practice_id)
        except Exception as exc:  # noqa: BLE001
            logger.info("lineage link failed (non-fatal): %s", exc)

    # -- ledger -------------------------------------------------------------

    def _next_attempt(self, ledger_id: str) -> int:
        if self._store is None:
            return 1
        try:
            rows = self._store.query(LEDGER_TABLE, {"id": ledger_id}, limit=1)
            return int(rows[0].get("attempt") or 0) + 1 if rows else 1
        except Exception:  # noqa: BLE001
            return 1

    def _ledger_write(self, ledger_id: str, fields: dict[str, Any]) -> None:
        if self._store is None:
            return
        now = time.time()
        try:
            rows = self._store.query(LEDGER_TABLE, {"id": ledger_id}, limit=1)
            base = dict(rows[0]) if rows else {}
            base.pop("_indices", None)
            base.update(fields)
            base["id"] = ledger_id
            base.setdefault("created_at", now)
            base["updated_at"] = now
            self._store.insert(LEDGER_TABLE, base)
        except Exception as exc:  # noqa: BLE001
            logger.info("ledger write failed (non-fatal): %s", exc)
