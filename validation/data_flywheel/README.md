# Data Flywheel Live Acceptance (DF-20)

Phase-II §27–§31: the acceptance harness that runs the **real** SeekDB
data plane end to end — not mocked stores.

## Deployment modes (§27.1)

| Mode | Shape | Use |
|------|-------|-----|
| A — edge (CI-required) | SQLite Structured Store + SeekDB Embedded Retrieval | DGX Spark / Jetson / single robot |
| B — server | SeekDB SQL Structured Store + SeekDB Native Retrieval | fleet / lab / central experience pool |

CI runs **Mode A embedded only** (§27.2); Mode B belongs to
integration/deployment environments.

## Layout

```
validation/data_flywheel/
├── README.md                  ← this file
├── configs/mode_a_embedded.yaml
├── fixtures/                  ← reserved for future labeled corpora
├── scripts/
│   ├── run_live_acceptance.py ← the runner (loop + metrics + hurt gate)
│   └── soak.sh                ← 1000 / 10,000 / 7×24 soak tiers
├── reports/                   ← acceptance JSON lands here
└── tests/                     ← CI-safe smoke tests
```

## CI small loop (§28): 50 episodes

```bash
python validation/data_flywheel/scripts/run_live_acceptance.py \
  --episodes 50 --hurt-gate \
  --workdir /tmp/data_flywheel_live \
  --report validation/data_flywheel/reports/live_acceptance.json
```

Drives mock practice episodes through fact ingest → memory distill →
projection → retrieval → How lookup → insight → evolution → lineage,
and asserts the §28 completion set plus: zero duplicates on replay,
bad-evidence-write rate ≈ 0, proposal lineage reaching insight+memory,
final projection lag 0 (when the embedded engine is available).

## Memory Hurt Gate (§31)

`--hurt-gate` runs five lanes over the regime fixture corpus
(`benchmarks/memory/regime`, disclosed deterministic doubles):

| Lane | Policy |
|------|--------|
| No Memory | always abstain (baseline) |
| Keyword | lexical top-1, always applied |
| Vector | embedding-cosine top-1, always applied (fake 8-dim provider) |
| Hybrid | production facade top-1, always applied |
| Hybrid + Body/Regime | selective intervention pipeline (APPLY/ABSTAIN) |

P0 gate: **Memory Hurt Rate ≤ 5%** for Hybrid+Regime, unsafe
interventions = 0, and success no worse than the No Memory baseline.
(Reference run: keyword 67.6% hurt, vector 96.3%, hybrid 64.8%,
hybrid+regime 0.0% — the regime gate IS the safety feature.)

## §30 metrics recorded

`structured_write` p50/95/99, `retrieval.query` p50/95/99,
`projection.max_projection_lag` + catchup times + final lag,
`memory` candidate/store/merge/ignore/quarantine/duplicate/untraceable
rates, `retrieval` fallback/abstention rates + cross-body stats via the
hurt gate lanes, `data_quality.bad_evidence_write_rate`.

Known honest gap: the retrieval-projection rebuild writes raw
`memory_items` rows without a versioned ACTIVE collection pointer, so
the facade currently serves the sample queries via its **declared
sqlite-lexical fallback** (`fallback_rate` 1.0 on the sample set) —
measured and reported, not hidden. Native semantic retrieval is
exercised by the hurt-gate lanes (bench fake stack) and by the MEM-5
versioned collections in production deployments.

## Soak (§29) — real machines, not CI

```bash
validation/data_flywheel/scripts/soak.sh 1000      # 1000 episodes
validation/data_flywheel/scripts/soak.sh 10000     # 10,000 episodes
validation/data_flywheel/scripts/soak.sh 7x24      # time-bounded loop
```
