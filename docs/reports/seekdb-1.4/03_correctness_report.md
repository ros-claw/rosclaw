# SeekDB 1.4.0 Qualification — 03: Correctness Report (C lane)

**Date:** 2026-09-15 · **Lane C:** Engine 1.4.0 + pyseekdb **1.4.0.post1**
(scratch venv, Python 3.12 — the #251-validated interpreter) · **Result:**
**PROMOTED** — `pyseekdb==1.4.0.post1` is the production pin as of this PR.

## Evidence

| Suite | Result |
|---|---|
| `repro_pyseekdb_140.py` (the #251 surface, embedded) | PASS — T1/T2/T4 all OK |
| Regression matrix T0–T15, post1's bundled 1.4 embedded engine | 16/16 |
| tests/storage + test_seekdb + test_seekdb_indexes + test_memory_vector | 273/273 |
| D negative control: pyseekdb **1.4.0 exactly** | RuntimeError fail-fast at connect ✔ |
| Server-mode benchmark (1.4.0 engine, post1 SDK) | no regression, mostly faster |
| zh/en golden corpus recall (vs SDK 1.3.0 on same engine) | identical |

Environment note: the 273-suite run's single initial failure was a scratch
venv missing fastapi (env gap, not an SDK regression); closing the gap made
it 5/5.

## Why promote (not just qualify)

1. The C matrix passed in full — the outline's promotion condition.
2. post1 bundles the 1.4-series embedded engine binding
   (pylibseekdb 1.4.0.post1) — staying on 1.3.0 would keep the 1.3-era
   engine in-process while the server fleet runs 1.4.
3. post1 is measurably faster on the same 1.4.0 engine (see 04).
4. The pin stays exact and the change is one-line reversible.

## What did NOT change

- `pyseekdb==1.4.0` (no post1) remains known-bad and fails fast
  (`ROSCLAW_ALLOW_KNOWN_BAD_SEEKDB=1` is a lab-only escape hatch).
- Engine 1.3.x deployments remain supported (SDK 1.3.0 stays VALIDATED).
