# SeekDB 1.4.0 Qualification — 05: Performance 1.3 vs 1.4

**Date:** 2026-09-15 · **Platform:** Jetson-class aarch64, Ubuntu 24.04 ·
**Tool:** `validation/seekdb/benchmarks/seekdb_benchmark.py` (real engine,
both modes; `--control-sqlite` exists but produces no SeekDB numbers) ·
**Machine-readable:** `baseline_1_3_server.json`,
`engine_1_4_sdk_1_3_server.json`, `baseline_1_3_embedded.json` (this dir)

Fixed conditions per outline §十七: same host, same corpus (400 rows of the
ROSClaw zh/en workload), same SDK (pyseekdb 1.3.0), same query set,
query-reps 200 for the firmed numbers.

| metric | 1.3 embedded | 1.3 server | 1.4 server | 1.3srv→1.4srv Δ |
|---|---:|---:|---:|---:|
| startup (s) | 3.509 | 1.127 | 1.118 | −0.8% |
| single upsert p95 (ms) | 66.3 | 65.9 | 66.0 | +0.2% |
| single write throughput (rows/s) | 14.1 | 14.2 | 14.3 | +0.7% |
| batch 256 p95 (ms) | 17794.6 | 17129.1 | 17687.9 | +3.3% |
| metadata filter p95 (ms) | 0.30 | 1.21 | 1.24 | +2.1% |
| BM25 p95 (ms) | 8.51 | 2.95 | 2.78 | **−5.6%** |
| vector p95 (ms) | 56.5 | 56.8 | 57.5 | +1.2% |
| hybrid RRF p95 (ms) | 75.1 | 62.0 | 60.0 | −3.2% |
| W2R metadata p95 (ms) | 2.44 | 3.46 | 3.48 | +0.5% |
| W2R BM25 p95 (ms) | 9.10 | 3.40 | 3.44 | +0.9% |
| W2R vector p95 (ms) | 56.5 | 57.3 | 57.8 | +0.9% |

## Verdict vs the outline's engineering gates (§十六)

- Correctness: **0 regression** (regression matrix T0–T15 green on both;
  server-mode table bootstrap verified live).
- Query p95: **all within ±10%**; BM25 and hybrid RRF improve.
- p99: BM25 +15.4% at n=200 (p99 = 2nd-deepest sample — tail noise);
  all other p99 within ±15%.  Watch item for the 35k-row full corpus run.
- Write throughput: +0.7% (no regression).
- Startup: 1.4 server connects slightly faster.

## Resource numbers (observed, not official-claim)

- 1.4 server process, warm after benchmark, default memory_limit=2G:
  RSS **705 MiB**, 86 threads, 38 fds.
- The official −73.6% idle-RSS claim is for a cold, tuned idle instance —
  our number is warm + default config; an idle/tuned measurement is
  scheduled for the soak (PR-4).
- 1.3 embedded inflates the HOST process (the engine runs in-process):
  benchmark process RSS 1.4 GB during runs.  1.4's out-of-process
  architecture makes the database's footprint separable and capped —
  that separation is itself the architectural win for rosclawd/agentd
  coexistence.

## Honest caveats

- 400-row corpus is the smoke scale; the 35k-row §十七 full workload
  (10k episodes / 5k failures / 20k memory nodes) runs in PR-3 with the
  multilingual golden corpus.
- Server numbers include loopback TCP; embedded's metadata filter is
  in-process (0.30 ms) — different deployment tradeoff, not a regression.
