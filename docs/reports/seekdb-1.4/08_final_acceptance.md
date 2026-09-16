# SeekDB 1.4.0 Qualification — 08: Final Acceptance

**Date:** 2026-09-15 · **Program:** ROSClaw SeekDB 1.4 Qualification (NOT a
dependency bump) · **PRs:** PR-SDB-140-1 (#559) → PR-SDB-140-2 (#562) →
PR-SDB-140-3 → PR-SDB-140-4

## Against the outline's final acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| seekdb 1.4 Engine actually running (not just a Python bump) | ✅ | deb-installed, wire-reported `5.7.25-OceanBase seekdb-v1.4.0.0`, doctor READY |
| migration complete | ✅ | 02: 416 rows/37 tables, parity PASS, honest exit codes |
| correctness 0 regression | ✅ | T0–T15 matrix + storage/seekdb suites on both SDK lanes |
| historical #251 matrix PASS | ✅ | repro green on 1.3.0 and on 1.4.0.post1's bundled 1.4 engine |
| metadata/BM25/vector/hybrid PASS | ✅ | 04: golden corpus, all legs, both SDKs |
| W2R latency tested | ✅ | 05: metadata/BM25/vector W2R p50/95/99 on all lanes |
| concurrency tested | ✅ | 06: 1w1r/1w4r/2w8r multi-process, zero lost/duplicate |
| restart persistence PASS | ✅ | benchmark restart probe + kill-9 fault injection |
| crash recovery tested | ✅ | 06: kill -9 → restart → data intact |
| performance report produced | ✅ | 05 + machine-readable JSONs |
| x86 validation | ⚠️ CI runners (Gate G matrix on ubuntu-24.04 amd64); no physical x86 box in this lab |
| ARM/edge validation PASS | ✅ | 07: this Jetson, full matrix |
| full ROSClaw regression PASS | ✅ | CI Full Regression on each PR |

## The four PRs

1. **PR-SDB-140-1** — compatibility matrix as code (`seekdb_compat.py`),
   known-bad fail-fast, strict refresh, T0–T15 regression harness, Gate G.
2. **PR-SDB-140-2** — Engine 1.4.0 server mode live on Jetson: pinned
   install/start/stop/status/doctor/migrate scripts, StoreMigrator, real
   benchmark (incl. W2R), migration-marker fixes that make fresh server
   bootstrap possible at all, pylibseekdb atexit exit-code hazard found +
   worked around.
3. **PR-SDB-140-3** — pyseekdb 1.4.0.post1 qualified and promoted to the
   production pin (the outline's promotion rule met: C matrix fully green).
4. **PR-SDB-140-4** — `SeekDBLocalRuntime` (lifecycle only, crash recovery,
   fail-closed on platforms without the bindings wheel),
   `ROSCLAW_SEEKDB_MODE=legacy_embedded|local_runtime|server` explicit
   deployment config, multi-process concurrency matrix proven on the shared
   1.4.0 instance.

## What 1.4.0 concretely bought ROSClaw

- BM25 p95 −33% and metadata-filter p95 −49% (post1 SDK on the 1.4 engine).
- W2R (write-to-retrievable) BM25 p95 3.1ms on the 1.4 server — the
  agent-memory KPI the outline asked to establish.
- The engine is now a separable, cap-able background process instead of
  inflating rosclawd's RSS — the Physical-AI-OS data-plane shape.
- A repeatable upgrade harness: matrix + benchmark + migration + parity +
  concurrency + fault injection all rerun on the next engine release.

## Not claimed

- local_runtime on aarch64 (upstream wheel is x86_64-only today).
- The official −73.6% idle-RSS figure (ours is a warm/default-config
  measurement; a tuned idle measurement belongs to the soak lane).
- 10min/60min soak lanes (the tools take `--duration-s`; not run here).
