# DF-20 — SeekDB Live Acceptance (implementation note)

Phase-II §27–§31.  This is where SeekDB itself gets tested: a live
acceptance harness at `validation/data_flywheel/` running the real Mode
A edge shape (SQLite Structured Store + SeekDB Embedded Retrieval).

## What landed

- **`scripts/run_live_acceptance.py`** — the runner.  Drives N mock
  practice episodes through the whole plane: fact ingest → memory
  distill (via the DF-19 reconciler) → projection (with lag/catch-up
  measurement) → retrieval sample → How lookup → insight → evolution →
  lineage trace.  Records the §30 metric set (write/query p50/95/99,
  projection lag/catch-up, memory decision + duplicate + untraceable
  rates, retrieval fallback/abstention, bad-evidence-write rate).
- **§31 Memory Hurt Gate** — five lanes (No Memory / Keyword / Vector /
  Hybrid / Hybrid+Body/Regime) over the regime fixture corpus
  (disclosed deterministic doubles).  Reference numbers on the bench
  corpus: keyword 67.6% hurt, vector 96.3%, hybrid 64.8%,
  **hybrid+regime 0.0% hurt / 0 unsafe** — the regime gate is
  demonstrably the safety feature, and the ungated lanes prove the gate
  isn't vacuous (pinned by `test_hurt_gate_ungated_lanes_actually_hurt`).
- **CI Gate E** in `data-flywheel-gate.yml`: the 50-episode loop + hurt
  gate + harness unit tests (§27.2: embedded only, no server in CI).
- **`scripts/soak.sh`** — 1000 / 10,000 / 7×24 tiers (§29), real
  machines only.
- Harness tests: hurt-gate P0 + meaningfulness + determinism, and a
  3-episode live smoke that asserts the full §28 completion set.

## Verified locally (Jetson, real embedded engine)

50-episode loop in ~30 s: 50 memories, duplicate_rate 0, untraceable
0, bad_evidence_write_rate 0, max projection lag 10 → final 0, all
live checks + hurt gate pass.

## Honest gaps (recorded, not hidden)

- The projection rebuild writes raw `memory_items` rows without a
  versioned ACTIVE collection pointer, so the facade's sample queries
  serve via the **declared sqlite-lexical fallback** (fallback_rate
  1.0 on the sample set).  Native semantic retrieval is exercised by
  the hurt-gate lanes; wiring the MEM-5 versioned projection into the
  rebuild path is follow-up work.
- Mock sessions emit a pre-existing fact-verify warning (1 error/1
  warning) — same as the DF-16B test sessions; the quality policy
  handles it (WARN × 0.7), and the metric is visible in the reports.
