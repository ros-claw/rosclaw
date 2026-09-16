# SeekDB 1.4.0 Qualification — 02: Migration Report

**Date:** 2026-09-15 · **Path:** embedded 1.3 (pylibseekdb) → server 1.4.0
(deb install) · **Tooling:** `scripts/seekdb/migrate_1_3_to_1_4.sh` +
`src/rosclaw/storage/migrate.py`

## Method (per outline §七/八 — logical migration, never raw files)

```
old 1.3 embedded dir ──dump──▶ JSONL + manifest (per-table sha256 over
                               sorted-id canonical rows, engine-managed
                               fields excluded)
                    ──restore─▶ fresh 1.4.0 server database
                    ──parity──▶ table set / row counts / content checksums
```

Cutover stays a deliberate operator step; the script never edits config.

## Live run (this Jetson)

```
[1/3] dump:    416 rows, 37 tables   (embedded:/tmp/bench_emb_1#bench)
[2/3] restore: 416 rows              (server:127.0.0.1:2881#rosclaw_migrated12)
[3/3] parity:  PASS (all 37 tables checksum-match)
```

Deterministic checksums: two dumps of the same source produce identical
per-table checksums (tests/storage/test_store_migrator.py).

## Findings that shaped the tooling

1. **pylibseekdb atexit overrides exit codes to 0** — the migration verdict
   only survives via `os._exit()`; otherwise a FAILED parity exits 0 and
   the shell script would have printed "migration complete" over a
   traceback (it did, before the fix).
2. **Retrieval rows are not SQL rows**: embedded corpus rows lack
   NOT NULL columns the SQL schema declares (`memory_type`), and carry
   engine-level fields (`embedding`).  Restore filters to the target
   schema's known columns; parity compares content minus engine-managed
   fields (embedding/score/distance live in the index, not the row).
3. **Schema-validated targets need real probe tables** — the doctor's
   first probe table (`doctor_probe`) was rejected by name validation;
   the probe now writes and deletes a real `memory_items` row.
4. URL schemes: `sqlite:` / `mysql:` / `embedded:<path>#<db>` /
   `server:<host>:<port>#<db>`.

## NOT done here (per plan)

- The 1.3-era OceanBase container dump (its data volume was not part of
  this repo's state); the same tooling accepts it via `mysql://…:2882`.
- Golden retrieval-query parity across engines (BM25/vector/hybrid answer
  identity) — scheduled in PR-3 with the multilingual corpus.
