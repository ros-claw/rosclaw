"""DataReconciler (PR-DF-19 / phase-II §17-§26).

Phase I proved the Runtime survives store outages.  DF-19 proves the
other half: **when the store comes back, the data comes back.**

Recovery sources (§18-§20), per fault class:

* Structured Store down → Practice raw artifacts (events.jsonl /
  episode.json / manifest) are the original truth; the catalog's §21
  reconcile ledger (fact_ingested / memory_distilled /
  reconcile_required) records what still owes the store.
* Retrieval Store down → the structured SoT is fine; a projection
  rebuild/catch-up restores parity (lag is an ops signal, never loss).
* Evolution spool → LocalStore cache entries missing-or-newer are
  upserted into the structured store and marked synced (never deleted —
  the cache stays a cache).

Every step is idempotent (§24): fact ingest dedups by entity ids,
distillation dedups by content hash at the write gate, lineage links
dedup on the 5-field key, projection upserts by memory id.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from rosclaw.practice.storage.catalog import (
    reconcile_catalog_path,
    update_reconcile_fields,
)

logger = logging.getLogger("rosclaw.storage.reconciler")

LEDGER_TABLE = "memory_distillation_runs"
_EVOLUTION_TABLE = "evolution_records"
_SPOOL_SYNC_NAMESPACE = "_reconcile"


def _manifest_hash(session_dir: Path) -> str:
    """The manifest hash the distillation ledger keys on (best-effort)."""
    manifest = session_dir / "manifest.yaml"
    if not manifest.exists():
        return ""
    try:
        text = manifest.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith(("manifest_hash:", "sha256:")):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except OSError:
        return ""


class DataReconciler:
    """Catch-up engine for the four fault classes (§18)."""

    def __init__(
        self,
        *,
        structured_store: Any,
        data_root: str | Path,
        retrieval_store: Any | None = None,
        evolution_cache: Any | None = None,
        lineage: Any | None = None,
    ) -> None:
        self._store = structured_store
        self._data_root = Path(data_root)
        self._retrieval_store = retrieval_store
        self._evolution_cache = evolution_cache
        self._lineage = lineage

    # ------------------------------------------------------------------
    # catalog access (read-only, raw sqlite — no batch writers)
    # ------------------------------------------------------------------

    def _catalog_rows(self, practice_id: str | None = None) -> list[dict[str, Any]]:
        catalog_path = reconcile_catalog_path(self._data_root)
        if not catalog_path.exists():
            return []
        conn = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            if practice_id:
                rows = conn.execute(
                    "SELECT * FROM practices WHERE practice_id = ?", (practice_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM practices").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def pending_practices(self) -> list[dict[str, Any]]:
        """Catalog rows that still owe the store work (§20 discovery)."""
        out = []
        for row in self._catalog_rows():
            fact_done = bool(row.get("fact_ingested")) or bool(row.get("seekdb_committed"))
            memory_done = bool(row.get("memory_distilled"))
            if row.get("reconcile_required") or not fact_done or not memory_done:
                out.append(row)
        return out

    # ------------------------------------------------------------------
    # §22: reconcile_practice — fact ingest + memory distill for one session
    # ------------------------------------------------------------------

    def reconcile_practice(
        self, practice_id: str, *, dry_run: bool = False
    ) -> dict[str, Any]:
        rows = self._catalog_rows(practice_id)
        if not rows:
            return {"practice_id": practice_id, "processed": False, "error": "not in catalog"}
        row = rows[0]
        session_dir = self._data_root / "sessions" / practice_id
        result: dict[str, Any] = {
            "practice_id": practice_id,
            "processed": False,
            "dry_run": dry_run,
            "fact_ingested_before": bool(row.get("fact_ingested") or row.get("seekdb_committed")),
            "memory_distilled_before": bool(row.get("memory_distilled")),
            "steps": {},
        }
        catalog_path = reconcile_catalog_path(self._data_root)

        # -- fact ingest (idempotent by entity ids) ----------------------
        if result["fact_ingested_before"]:
            result["steps"]["fact_ingest"] = "already_done"
        elif dry_run:
            result["steps"]["fact_ingest"] = "would_run"
        else:
            try:
                from rosclaw.practice.seekdb_ingestor import PracticeFactIngestor

                with PracticeFactIngestor(self._data_root, seekdb_client=self._store) as ing:
                    report = ing.ingest_practice(practice_id)
                result["steps"]["fact_ingest"] = {
                    "success": report.success,
                    "tables": report.table_counts,
                    "errors": report.errors,
                }
                if not report.success:
                    update_reconcile_fields(
                        catalog_path,
                        practice_id,
                        {"reconcile_required": 1, "fact_ingest_error": "; ".join(report.errors)[:400]},
                    )
            except Exception as exc:  # noqa: BLE001
                result["steps"]["fact_ingest"] = {"success": False, "error": str(exc)}
                update_reconcile_fields(
                    catalog_path,
                    practice_id,
                    {"reconcile_required": 1, "fact_ingest_error": str(exc)[:400]},
                )

        # -- memory distill (idempotent by content hash at the gate) -----
        if result["memory_distilled_before"]:
            result["steps"]["memory_distill"] = "already_done"
        elif not session_dir.exists():
            result["steps"]["memory_distill"] = "skipped_no_session_dir"
        elif dry_run:
            result["steps"]["memory_distill"] = "would_run"
        else:
            try:
                self._distill_session(practice_id, session_dir, row)
                ledger = self._distill_ledger_status(practice_id, session_dir)
                result["steps"]["memory_distill"] = ledger
            except Exception as exc:  # noqa: BLE001
                result["steps"]["memory_distill"] = {"status": "failed", "error": str(exc)}

        result["processed"] = True
        if not dry_run:
            after = self._catalog_rows(practice_id)
            if after:
                result["reconcile_required_after"] = bool(after[0].get("reconcile_required"))
        return result

    def _distill_session(
        self, practice_id: str, session_dir: Path, catalog_row: dict[str, Any]
    ) -> None:
        """Run the DF-16B distillation work path synchronously for one session."""
        from rosclaw.memory.v2.distillation_service import MemoryDistillationService
        from rosclaw.memory.v2.gate import MemoryWriteGate
        from rosclaw.memory.v2.repository import MemoryRepository

        repository = MemoryRepository(self._store)
        gate = MemoryWriteGate(repository)
        service = MemoryDistillationService(None, repository, gate, lineage=self._lineage)
        service._work(
            {
                "practice_id": practice_id,
                "session_id": catalog_row.get("session_id"),
                "episode_id": catalog_row.get("episode_id"),
                "session_dir": str(session_dir),
                "manifest_hash": _manifest_hash(session_dir),
                "fact_verify": self._fact_verify(practice_id),
                "_reconciled": True,
            }
        )

    def _fact_verify(self, practice_id: str) -> dict[str, Any]:
        """Recompute the close-time fact verdict (verifier is idempotent, DF-04).

        An empty ``fact_verify`` payload would trip the quality policy's FAIL
        branch and quarantine every candidate — reconstruct the real verdict
        from the raw session instead.
        """
        try:
            from rosclaw.practice.verifier import PracticeVerifier

            report = PracticeVerifier(self._data_root).verify(practice_id)
            return {
                "passed": report.passed,
                "errors": sum(1 for i in report.issues if i.level == "error"),
                "warnings": sum(1 for i in report.issues if i.level == "warning"),
            }
        except Exception as exc:  # noqa: BLE001
            logger.info("fact re-verify unavailable for %s: %s", practice_id, exc)
            return {}

    def _distill_ledger_status(self, practice_id: str, session_dir: Path) -> dict[str, Any]:
        """Read the ledger row this reconcile just wrote (status verdict)."""
        manifest_hash = _manifest_hash(session_dir) or "nohash"
        try:
            rows = self._store.query(
                LEDGER_TABLE, {"id": f"{practice_id}:{manifest_hash}"}, limit=1
            )
            if rows:
                return {
                    "status": rows[0].get("status"),
                    "stored": rows[0].get("stored_count"),
                    "merged": rows[0].get("merged_count"),
                    "error": rows[0].get("last_error"),
                }
        except Exception as exc:  # noqa: BLE001
            return {"status": "unknown", "error": str(exc)}
        return {"status": "unknown"}

    # ------------------------------------------------------------------
    # §22: reconcile_memory — every practice that owes distillation/ingest
    # ------------------------------------------------------------------

    def reconcile_memory(self, *, dry_run: bool = False, limit: int = 1000) -> dict[str, Any]:
        pending = self.pending_practices()[:limit]
        results = [
            self.reconcile_practice(row["practice_id"], dry_run=dry_run) for row in pending
        ]
        processed = sum(1 for r in results if r.get("processed"))
        still_required = sum(1 for r in results if r.get("reconcile_required_after"))
        return {
            "pending_before": len(pending),
            "processed": processed,
            "still_required": still_required,
            "dry_run": dry_run,
            "practices": results,
        }

    # ------------------------------------------------------------------
    # §22 / fault class B: reconcile_projection — retrieval catch-up
    # ------------------------------------------------------------------

    def reconcile_projection(self, *, dry_run: bool = False) -> dict[str, Any]:
        if self._retrieval_store is None:
            return {"skipped": "no retrieval store"}
        from rosclaw.memory.v2.repository import MemoryRepository
        from rosclaw.storage.seekdb_projection import MemoryRetrievalProjection

        repository = MemoryRepository(self._store)
        projection = MemoryRetrievalProjection(self._retrieval_store)
        before = projection.status(repository)
        result: dict[str, Any] = {
            "lag_before": before.get("lag"),
            "source_count": before.get("source_count"),
            "projection_count_before": before.get("projection_count"),
            "dry_run": dry_run,
        }
        if dry_run:
            result["action"] = "would_rebuild" if before.get("lag") else "noop"
            return result
        if before.get("lag"):
            rebuild = projection.rebuild(repository)
            after = projection.status(repository)
            result["rebuilt"] = rebuild.get("rebuilt")
            result["lag_after"] = after.get("lag")
        else:
            result["action"] = "noop"
            result["lag_after"] = before.get("lag", 0)
        return result

    # ------------------------------------------------------------------
    # §22/§25: reconcile_evolution_spool — LocalStore cache -> structured
    # ------------------------------------------------------------------

    def reconcile_evolution_spool(self, *, dry_run: bool = False) -> dict[str, Any]:
        cache = self._evolution_cache
        if cache is None:
            return {"skipped": "no evolution cache"}
        base = Path(cache.base)
        stats: dict[str, Any] = {
            "namespaces": {},
            "upserted": 0,
            "skipped_current": 0,
            "errors": 0,
            "dry_run": dry_run,
        }
        namespaces = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []
        for namespace in namespaces:
            if namespace == _SPOOL_SYNC_NAMESPACE:
                continue
            ns_stats = {"upserted": 0, "current": 0, "errors": 0}
            for key in cache.list_keys(namespace):
                try:
                    outcome = self._reconcile_spool_entry(namespace, key, dry_run=dry_run)
                    ns_stats["upserted" if outcome == "upserted" else outcome] += 1
                    if outcome == "upserted":
                        stats["upserted"] += 1
                    elif outcome == "current":
                        stats["skipped_current"] += 1
                    else:
                        stats["errors"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.info("spool reconcile failed for %s/%s: %s", namespace, key, exc)
                    ns_stats["errors"] += 1
                    stats["errors"] += 1
            stats["namespaces"][namespace] = ns_stats
        return stats

    def _reconcile_spool_entry(self, namespace: str, key: str, *, dry_run: bool) -> str:
        """Upsert one cache entry when missing/newer in the store; mark synced."""
        cache = self._evolution_cache
        data = cache.load(namespace, key)
        if data is None:
            return "current"
        cache_mtime = (Path(cache.base) / namespace / f"{key}.json").stat().st_mtime
        row_id = f"{namespace}:{key}"
        store_updated_at = 0.0
        try:
            rows = self._store.query(_EVOLUTION_TABLE, {"id": row_id}, limit=1)
            if rows:
                store_updated_at = float(rows[0].get("updated_at") or 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.info("spool compare failed for %s: %s", row_id, exc)
            return "errors"
        synced = cache.load(_SPOOL_SYNC_NAMESPACE, row_id) or {}
        already_synced = float(synced.get("synced_at") or 0.0) >= cache_mtime and store_updated_at > 0
        if already_synced or store_updated_at >= cache_mtime:
            return "current"
        if dry_run:
            return "upserted"
        # Upsert missing/newer (§25), verify, mark synced — never delete.
        self._store.insert(
            _EVOLUTION_TABLE,
            {
                "id": row_id,
                "namespace": namespace,
                "key": key,
                "data": json.dumps(data, default=str),
                "updated_at": time.time(),
            },
        )
        rows = self._store.query(_EVOLUTION_TABLE, {"id": row_id}, limit=1)
        if not rows:
            return "errors"
        cache.save(_SPOOL_SYNC_NAMESPACE, row_id, {"synced_at": time.time(), "namespace": namespace})
        return "upserted"

    # ------------------------------------------------------------------
    # one-shot: everything
    # ------------------------------------------------------------------

    def reconcile_all(
        self, *, practice_id: str | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        if practice_id:
            memory = self.reconcile_practice(practice_id, dry_run=dry_run)
        else:
            memory = self.reconcile_memory(dry_run=dry_run)
        projection = self.reconcile_projection(dry_run=dry_run)
        spool = self.reconcile_evolution_spool(dry_run=dry_run)
        return {
            "dry_run": dry_run,
            "memory": memory,
            "projection": projection,
            "evolution_spool": spool,
        }
