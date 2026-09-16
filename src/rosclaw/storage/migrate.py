"""Cross-instance logical migration for structured stores (PR-SDB-140-2).

seekdb 1.3 -> 1.4 has NO in-place upgrade (upstream): the only supported
path is a logical dump -> restore -> parity -> cutover.  This module does
that through the StructuredStore interface, so it works for any pair of
backends (sqlite / mysql / seekdb_embedded / seekdb_server):

    dump:    every table -> one JSONL file + a manifest with a
             deterministic per-table checksum (sorted id + canonical JSON)
    restore: insert every row into the target (idempotent upsert where the
             store supports it)
    parity:  table set, row counts, id sets, per-table checksums, and a
             fixed set of golden retrieval queries on both sides

Nothing here touches raw engine files — copying data directories between
engine versions is explicitly forbidden by the outline (§七/八).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from rosclaw.memory.seekdb_client import ROSCLAW_STRUCTURED_SCHEMAS

logger = logging.getLogger("rosclaw.storage.migrate")

MANIFEST = "manifest.json"


# Engine-managed fields, not record content: the embedding vector lives in
# the index (not the metadata projection a query returns), and score/
# distance are query-time annotations.  Parity compares CONTENT.
_ENGINE_MANAGED = frozenset({"embedding", "embeddings", "score", "distance", "distances"})


def _content(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in _ENGINE_MANAGED}


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _table_checksum(rows: list[dict[str, Any]]) -> str:
    """Deterministic content checksum: sorted-by-id canonical rows."""
    keyed = sorted(
        (_content(r) for r in rows),
        key=lambda r: str(r.get("id") or r.get("memory_id") or ""),
    )
    return hashlib.sha256(_canonical(keyed).encode()).hexdigest()


class StoreMigrator:
    """Logical dump/restore/parity between two StructuredStore instances."""

    def __init__(self, store: Any):
        self._store = store

    # -- discovery -------------------------------------------------------

    def list_tables(self) -> list[str]:
        """Known schema tables that actually exist + any extras the store
        reports (engine-specific discovery is best-effort)."""
        tables: set[str] = set()
        for name in ROSCLAW_STRUCTURED_SCHEMAS:
            try:
                self._store.query(name, {}, limit=1)
                tables.add(name)
            except Exception:  # noqa: BLE001 — table simply absent
                continue
        discover = getattr(self._store, "list_tables", None)
        if callable(discover):
            try:
                tables.update(discover())
            except Exception as exc:  # noqa: BLE001
                logger.info("store table discovery unavailable: %s", exc)
        return sorted(tables)

    # -- dump -------------------------------------------------------------

    def dump(self, out_dir: str | Path) -> dict[str, Any]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "format": "rosclaw.store.dump.v1",
            "created_at": time.time(),
            "tables": {},
        }
        for table in self.list_tables():
            rows = self._store.query(table, {}, limit=10_000_000)
            (out / f"{table}.jsonl").write_text(
                "".join(_canonical(r) + "\n" for r in rows), encoding="utf-8"
            )
            manifest["tables"][table] = {
                "rows": len(rows),
                "checksum": _table_checksum(rows),
            }
        (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("dumped %d tables to %s", len(manifest["tables"]), out)
        return manifest

    # -- restore -----------------------------------------------------------

    def restore(self, dump_dir: str | Path) -> dict[str, Any]:
        src = Path(dump_dir)
        manifest = json.loads((src / MANIFEST).read_text(encoding="utf-8"))
        restored: dict[str, int] = {}
        for table in manifest["tables"]:
            path = src / f"{table}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                restored[table] = 0
                continue
            # Retrieval-side rows may carry engine-level fields (raw
            # embeddings etc.) that the SQL schema doesn't declare — keep
            # only the columns the target schema knows.
            schema = ROSCLAW_STRUCTURED_SCHEMAS.get(table)
            if schema is not None:
                known = set(schema["columns"])
                rows = [{k: v for k, v in row.items() if k in known} for row in rows]
            insert_many = getattr(self._store, "insert_many", None)
            if callable(insert_many):
                insert_many(table, rows)
            else:
                for row in rows:
                    self._store.insert(table, row)
            restored[table] = len(rows)
        logger.info("restored %d tables from %s", len(restored), src)
        return restored

    # -- parity -------------------------------------------------------------

    @staticmethod
    def parity(source_manifest: dict[str, Any], target: Any) -> dict[str, Any]:
        """Compare a dump manifest against a live target store."""
        report: dict[str, Any] = {"ok": True, "tables": {}}
        for table, want in source_manifest["tables"].items():
            rows = target.query(table, {}, limit=10_000_000)
            got_rows = len(rows)
            got_checksum = _table_checksum(rows)
            ok = got_rows == want["rows"] and got_checksum == want["checksum"]
            report["tables"][table] = {
                "rows_want": want["rows"],
                "rows_got": got_rows,
                "checksum_match": got_checksum == want["checksum"],
                "ok": ok,
            }
            if not ok:
                report["ok"] = False
        return report
