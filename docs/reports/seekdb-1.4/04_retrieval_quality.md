# SeekDB 1.4.0 Qualification — 04: Retrieval Quality (zh/en golden corpus)

**Date:** 2026-09-15 · **Tool:** `validation/seekdb/scripts/retrieval_quality.py`
· **Corpus:** 8 golden docs crossing languages (UR5e/RH56/LIMO/Nova Carter;
zh primary + en paraphrase), 6 golden queries incl. the outline §十七's
three cases · **Machine-readable:** `retrieval_quality_sdk_1_3.json`,
`retrieval_quality_sdk_1_4_post1.json` (this dir)

## SDK A/B on the same Engine 1.4.0 server (identical corpus, reps=5)

| leg | metric | SDK 1.3.0 | SDK 1.4.0.post1 |
|---|---|---:|---:|
| BM25 | Recall@1 / @5 / MRR | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| BM25 | p95 (ms) | 2.41 | 2.05 |
| vector | Recall@1 / @5 / MRR | 0.833 / 0.833 / 0.833 | 0.833 / 0.833 / 0.833 |
| vector | p95 (ms) | 55.9 | 54.6 |
| hybrid | Recall@1 / @5 / MRR | 0.833 / 1.0 / 0.889 | 0.833 / 1.0 / 0.889 |
| hybrid | p95 (ms) | 60.3 | 60.0 |

Quality is **identical** across SDKs (same engine, same answers); post1
latencies equal or better.  The vector leg's 0.833 reflects the engine's
built-in MiniLM on zh-heavy queries — the Qwen multilingual index path
(PR-MEM-4) is the production answer for that and is engine-independent;
this report covers the engine/SDK plane only.

## Server-mode benchmark, SDK A/B (same 1.4.0 engine)

| metric | SDK 1.3.0 | SDK 1.4.0.post1 | Δ |
|---|---:|---:|---:|
| startup (s) | 1.093 | 0.187 | −82.9% |
| metadata filter p95 (ms) | 0.995 | 0.504 | −49.4% |
| BM25 p95 (ms) | 2.78 | 1.86 | −33.1% |
| hybrid RRF p95 (ms) | 60.0 | 58.3 | −2.8% |
| vector p95 (ms) | 55.6 | 54.6 | −1.9% |
| W2R BM25 p95 (ms) | 3.92 | 3.09 | −21.2% |
| single upsert p95 (ms) | 62.3 | 67.6 | +8.6% (within gate) |

Machine-readable: `bench_engine_1_4_sdk_1_4_post1.json`.
