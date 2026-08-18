# DF-19 — Data Reconciliation (implementation note)

Phase-II P0/P1 per `seekdb优化v2.md` §17–§26. Phase I proved the Runtime
survives store outages; DF-19 proves the other half: **when the store
comes back, the data comes back.**

## What landed

### §21 PracticeCatalog reconcile ledger

Seven new columns on `practices` (schema + `_LEGACY_TABLES` auto-migration):
`fact_ingested`, `memory_distilled`, `last_fact_ingest_at`,
`last_memory_distill_at`, `fact_ingest_error`, `memory_distill_error`,
`reconcile_required`. New module helpers `reconcile_catalog_path()` and
`update_reconcile_fields()` — raw-sqlite, best-effort, never raising; the
`reconcile_required="auto"` update clears the flag only when BOTH sides
are done (SET-clause CASE sees OLD row values in SQLite, so the in-flight
flags are folded in as parameters — caught by test).

### Writers mark the ledger

- `MemoryDistillationService._work` marks `memory_distilled` on success
  and `reconcile_required=1` + error on failure (§20: no fake "memory
  stored" while the store is down — the ledger records the debt instead).
- `PracticeFactIngestor` marks `fact_ingested` alongside the existing
  `seekdb_committed`, and auto-clears the flag.

### §22 DataReconciler — `src/rosclaw/storage/reconciler.py`

- `reconcile_practice(id)`: fact ingest (PracticeFactIngestor, idempotent
  by entity ids) + memory distill (the DF-16B `_work` path, idempotent by
  content hash at the write gate).  The close-time fact-verify verdict is
  **recomputed** with the idempotent PracticeVerifier — an empty
  fact_verify would trip the quality policy's FAIL branch and quarantine
  everything (caught by test).
- `reconcile_memory()`: discovers `reconcile_required=1 OR
  (fact_ingested=0 AND seekdb_committed=0) OR memory_distilled=0` and
  processes each.  Legacy rows already committed count as fact-done.
- `reconcile_projection()`: lag check vs source watermark, rebuild to
  parity (fault class B).
- `reconcile_evolution_spool()` (§25): enumerate LocalStore cache, upsert
  missing/newer into `evolution_records`, verify, mark synced in a
  `_reconcile` namespace.  Local copies are never deleted.

### §23 CLI — `rosclaw data reconcile`

`--practice ID`, `--all` (default), `--dry-run`, `--json`, `--data-root`,
`--spool-path`, plus the usual backend overrides.  Exit 1 when anything
remains unreconciled.

## §26 fault E2E — `tests/storage/test_data_reconciler.py` (10 tests)

- Case 1: retrieval down → lag = source → rebuild → lag 0 (dry-run
  writes nothing).
- Case 2: structured down → no fake memory, `reconcile_required=1` →
  recovery → processed, flags cleared, and 3× reconcile = 0 duplicates
  (Case 3 folded in per §24).
- Case 4: crash between practice close and distill → reconciler
  discovers the pending practice and finishes it.
- §25: spool upserts missing/newer, sync marks written, local copies
  kept, second run a noop.
- Catalog columns, dry-run, projection/spool skips, CLI JSON run against
  a real SQLite store.

Full affected suites (storage/practice/memory-v2/e2e/runtime-coverage/
flywheel-recorder): **787 passed**.
