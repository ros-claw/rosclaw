# SeekDB 1.4.0 Qualification — 06: Concurrency + Fault Injection

**Date:** 2026-09-15 · **Tool:** `validation/seekdb/scripts/concurrency_matrix.py`
· **Target:** live 1.4.0 server (deb install, this Jetson) ·
**Machine-readable:** `concurrency_smoke.json` shape documented below

## Multi-process shared instance (outline §十五/十八)

1.4's architectural point: multiple applications share ONE local instance
over the MySQL protocol.  Proven with real OS processes (multiprocessing),
unique-marked writer rows, sequential lanes on one database:

| lane | written | final count | lost | duplicates | crash | errors | verdict |
|---|---:|---:|---:|---:|---|---|---|
| 1w1r | 778 | 778 | 0 | 0 | none | none | PASS |
| 1w4r | 751 | 751 | 0 | 0 | none | none | PASS |
| 2w8r | 1558 | 1558 | 0 | 0 | none | none | PASS |

(20s/20ms-period smoke per lane; the 10min/60min soak lanes use the same
tool with `--duration-s`.)

## Fault injection (outline §十六)

| fault | result |
|---|---|
| SIGKILL engine mid-write → restart via `start_1_4.sh` | marker row persisted; engine READY again; **PASS** |
| connect while engine down | `OperationalError 2003` (connection refused) — fails LOUDLY at connect, no silent fallback ✔ |
| stale pid file after re-exec | status/doctor resolve the live pid by port ✔ |

ROSClaw fail-closed behavior verified: a down server raises at
`store.connect()`; nothing silently degrades to another backend.
`ROSCLAW_SEEKDB_MODE=local_runtime` on a platform without the bindings
wheel raises `LocalRuntimeUnavailableError` immediately.

## Known platform boundary

The new `seekdb` bindings (1.4.0.dev2) ship **x86_64-only wheels** — the
local_runtime path is constructible on x86_64 only; aarch64 deployments
use `server` (this qualification) or `legacy_embedded` until upstream
ships an arm64 wheel.
