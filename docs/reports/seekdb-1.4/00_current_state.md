# SeekDB 1.4.0 Qualification — 00: Current State Audit

**Date:** 2026-09-14 · **Branch base:** `main@4553ffa6` (post DF-25) ·
**Auditor:** local Claude Code · **Scope:** full repo (src/ tests/ validation/
docs/ scripts/ CI), per the 1.4.0 upgrade outline §二

## 1. Dependency pins (current production truth)

| Where | Pin | Note |
|---|---|---|
| `pyproject.toml` `[project.optional-dependencies] seekdb` | `pyseekdb==1.3.0` | "validated on Jetson aarch64" |
| `pyproject.toml` `dev` | `pyseekdb==1.3.0` | second pin site — must stay in sync |
| `validation/ty1200/configs/seekdb/requirements-lock.txt` | `pyseekdb==1.3.0`, `pylibseekdb==1.3.0.post3`, `seekdb==0.0.1.dev5`, `seekdb-lib==0.0.1.dev5` | full engine combo lock (2026-08-02 实测) |
| local `.venv` (this Jetson) | pyseekdb **1.3.0**, pylibseekdb **1.3.0** (metadata), no `seekdb` pkg | uv-managed, Python 3.13 |

Known-bad marking exists only as a **comment** in the lock file and as a
`logger.error` in `seekdb_native._warn_on_unvalidated_pyseekdb` — no
fail-fast (outline §五 requires one).

## 2. Backend architecture today

```
StoreFactory (storage/factory.py)
  create_structured_store: memory | sqlite | mysql | seekdb_embedded | seekdb_server
    ├─ seekdb_embedded → SeekDBEmbeddedRetrievalStore (seekdb_native.py:718)
    │                    pylibseekdb, engine IN-PROCESS
    └─ seekdb_server   → SeekDBServerRetrievalStore (seekdb_native.py:729)
                         mysql://…:2881 via SeekDBMySQLClient (memory/seekdb_client.py)
Structured (SQL) side: SQLiteStructuredStore / SeekDBSQLStore (MySQL) / InMemory
Retrieval projection: seekdb_projection.py / versioned_projection.py → refresh_index
```

Legacy assumptions to retire or gate behind capabilities:
- `seekdb_native.py:43` `_EMBEDDED_PROCESS_TARGET` — one embedded
  path/database target **per process** (pylibseekdb limitation).  1.4's
  local-runtime removes it; must remain scoped to `legacy_embedded`.
- Deployment mode inferred from `path is not None` / `host is not None`
  (outline §五: make it explicit).
- `validation/ty1200/tests/test_seekdb_live.py` docstring still says
  "embedded = REAL engine **in-process**" — stale under 1.4 architecture.

## 3. Version-warning / fail-fast surface

`seekdb_native._warn_on_unvalidated_pyseekdb`:
- validated = {1.3.0, 1.4.0.post1}; known_incompatible = {1.4.0}
- known-bad → `logger.error` only (outline §五: must `raise` unless
  `ROSCLAW_ALLOW_KNOWN_BAD_SEEKDB=1`).

`refresh_index` (`seekdb_native.py:408`): swallows ALL exceptions into
`logger.debug` — a 1.4 refresh-timeout regression would stay invisible
(outline §九/十三: explicit refresh must be `strict=True` by default on
write→searchable paths; only background maintenance may swallow).

Callers: `seekdb_projection.py:44,135` (write→searchable — strict),
`versioned_projection.py:207,374`, `versioned_collections.py:325`
(direct pyseekdb call), `seekdb_native.py:405` (post-upsert).

## 4. Config / CLI surface

- Env: `ROSCLAW_SEEKDB_URL`, `ROSCLAW_SEEKDB_FALLBACK_DIR`.
- rosclaw.yaml: `runtime.seekdb_backend|seekdb_url|seekdb_path`,
  `storage.retrieval.*` (Config v2, DF-18).
- CLI: `rosclaw db status|doctor|reconcile` (`storage/cli.py`) — doctor has
  no engine/SDK/binding version or capability output yet (outline §二十一).

## 5. Tests / benchmarks touching SeekDB

- `tests/test_seekdb.py`, `tests/test_seekdb_indexes.py`, plus broad
  memory/knowledge/practice coverage via InMemory/SQLite fakes.
- `tests/memory/v2/test_cli.py` opens the real embedded store (one-path
  collision with the DF-20 harness observed during DF-25 — fixed there by
  null-on-failure; same pattern needed anywhere embedded opens).
- `validation/ty1200/scripts/repro_pyseekdb_140.py` — historical #251
  repro (T0 unfiltered, T1 BM25+filters, T2 dual-leg RRF, T3 query+where,
  T4 KNN+where, T5 $and).  **Keep as history**; promote to
  `tests/integration/seekdb/test_seekdb_regression_matrix.py` (T0–T15).
- `validation/ty1200/tests/test_seekdb_live.py` — live engine tests;
  contains the stale "1.4.0 embedded: Collection.query returns empty"
  note (re-test on 1.4.0.post1 / Engine 1.4, keep issue reference).
- `validation/ty1200/benchmarks/seekdb_benchmark.py` — **not a SeekDB
  benchmark**: local path measures `sqlite3.connect`, server path measures
  connect/counts only.  To be rewritten in PR-2 (outline §十二).
- `validation/data_flywheel/` + `validation/golden_flywheel/` — the DF-20/25
  harnesses (embedded lane + honest fallback).

## 6. CI

- `.github/workflows/data-flywheel-gate.yml` — installs
  `.[dev,seekdb,embedding,knowledge]`; Gates A–F.  No version-matrix lane
  yet; regression matrix lands there in PR-1.
- Main CI (ci.yml) — same extra set.

## 7. Deployment scripts / docs

- `scripts/` has NO `seekdb/` install/start/stop/doctor/migrate tooling
  (PR-2 creates it, pinned 1.4.0 + SHA256).
- Docs with embedded/in-process assumptions to annotate as legacy-1.3:
  `docs/practice/SEEKDB_INTEGRATION.md`, `docs/guides/seekdb-multilingual-index.md`,
  README/deployment docs (outline §二十二 — annotate, never rewrite
  history).

## 8. Upstream facts (verified 2026-09-14)

- `oceanbase/seekdb` **v1.4.0** released 2026-08-26/27, still Latest.
- Release assets: el7/el9 RPM (x86_64+aarch64), debian12/13,
  ubuntu22.04/24.04 DEB (amd64+**arm64**), macOS pkg.  No container tag
  beyond `latest` on quay.io (local image `quay.io/oceanbase/seekdb:latest`
  is the 2026-05-25 arm64 build = 1.3-era; no `1.4.0` tag exists).
- Official compatible SDK: **pyseekdb 1.3.0**.  `pyseekdb` PyPI latest is
  **1.4.0.post1** (1.4.0 = known-bad, oceanbase/pyseekdb#251).
- `seekdb` PyPI package (new bindings): only **1.4.0.dev2** — PR-4 must
  treat it as experimental.
- **No in-place upgrade 1.3 → 1.4** — logical migration only.
- 1.4 embedded = separate background process over MySQL protocol via
  seekdb-bindings; multi-process shared instance becomes possible.

## 9. Local platform

This machine **is** the edge target: Jetson-class aarch64
(`Linux 6.17.0-1021-nvidia`), Ubuntu userland, glibc 2.39, uv 0.11.0,
Docker with the 1.3-era arm64 image.  The outline's "4×A6000 x86_64
workstation" is not this machine — x86_64 reference numbers come from CI
runners; this Jetson produces the edge/aarch64 validation (§十八/十九).

## 10. Target compatibility matrix (outline §一/十五)

| Lane | Engine | SDK | Status |
|---|---|---|---|
| A baseline | 1.3.x | pyseekdb 1.3.0 | current production |
| B primary | **1.4.0** | pyseekdb 1.3.0 | PR-2 target |
| C extended | **1.4.0** | pyseekdb 1.4.0.post1 | PR-3 candidate |
| D negative | any | pyseekdb **1.4.0** | forbidden — fail fast |
| local-runtime | 1.4.0 | seekdb 1.4.0.dev2 (experimental) | PR-4 |
