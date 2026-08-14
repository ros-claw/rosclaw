"""EvolutionRepository (PR-DF-12 / flywheel §32-35).

The Evolution module's canonical state lives in the Structured Store
(``evolution_records`` table); the legacy ``LocalStore`` JSONL tree is
demoted to a *cache / offline spool*:

* ``save`` writes the structured store first, then mirrors into the cache;
  if the store is down the record spools to the cache and the failure is
  logged — never silently lost, never blocking the engine.
* ``load`` / ``iterate`` / ``list_keys`` read the store first and fall
  back to the cache, so an offline SeekDB/SQLite degrades to pre-DF-12
  behavior (ADR-0009 §4).

The repository deliberately implements the ``LocalStore`` interface
(``save/load/list_keys/iterate/delete``) so ``AutoEngine`` swaps it in
without internal changes.  Records are stored as
``{id: "<namespace>:<key>", namespace, key, data, updated_at}`` — the data
JSON keeps full provenance (trace/evidence/refs) written by the engine.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger("rosclaw.evolution.repository")

TABLE = "evolution_records"


class EvolutionRepository:
    """Structured-store-backed Evolution state with a LocalStore spool."""

    def __init__(self, structured_store: Any, cache: Any) -> None:
        self._store = structured_store
        self._cache = cache  # LocalStore — cache / offline spool

    @property
    def base(self):
        """The spool root — keeps LocalStore introspection compat."""
        return self._cache.base

    @staticmethod
    def _row_id(namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    # -- LocalStore-compatible interface ---------------------------------

    def save(self, namespace: str, key: str, data: dict) -> None:
        record = {
            "id": self._row_id(namespace, key),
            "namespace": namespace,
            "key": key,
            "data": json.dumps(data, default=str),
            "updated_at": time.time(),
        }
        try:
            self._store.insert(TABLE, record)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "evolution store write failed (%s/%s) — spooled to local cache: %s",
                namespace,
                key,
                exc,
            )
        try:
            self._cache.save(namespace, key, data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("evolution cache write failed (%s/%s): %s", namespace, key, exc)

    def load(self, namespace: str, key: str) -> dict | None:
        try:
            rows = self._store.query(TABLE, {"id": self._row_id(namespace, key)}, limit=1)
            if rows:
                return json.loads(rows[0]["data"])
            # not in the store — maybe it only exists in the spool
        except Exception as exc:  # noqa: BLE001
            logger.info("evolution store read failed (%s/%s), using cache: %s", namespace, key, exc)
        return self._cache.load(namespace, key)

    def list_keys(self, namespace: str) -> list[str]:
        try:
            rows = self._store.query(TABLE, {"namespace": namespace}, limit=100_000)
            keys = [r.get("key") for r in rows if r.get("key")]
            if keys:
                return keys
        except Exception as exc:  # noqa: BLE001
            logger.info("evolution store list failed (%s), using cache: %s", namespace, exc)
        return self._cache.list_keys(namespace)

    def iterate(self, namespace: str) -> Iterator[dict]:
        for key in self.list_keys(namespace):
            data = self.load(namespace, key)
            if data is not None:
                yield data

    def delete(self, namespace: str, key: str) -> bool:
        deleted = False
        try:
            deleted = bool(self._store.delete(TABLE, self._row_id(namespace, key)))
        except Exception as exc:  # noqa: BLE001
            logger.info("evolution store delete failed (%s/%s): %s", namespace, key, exc)
        cache_deleted = self._cache.delete(namespace, key)
        return deleted or cache_deleted

    # -- observability -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Namespace counts for storage status/doctor (PR-DF-15 feeds)."""
        try:
            rows = self._store.query(TABLE, limit=500_000)
            counts: dict[str, int] = {}
            for row in rows:
                ns = row.get("namespace", "?")
                counts[ns] = counts.get(ns, 0) + 1
            return {"backend": type(self._store).__name__, "namespaces": counts}
        except Exception as exc:  # noqa: BLE001
            return {"backend": type(self._store).__name__, "error": str(exc)}
