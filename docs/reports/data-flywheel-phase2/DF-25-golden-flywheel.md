# DF-25: Golden Flywheel — final Phase II acceptance (§42-§44)

**Branch:** `pr-df-25-golden-flywheel` · **Status:** implemented, all tests green locally

## What this is

The §42-§44 final acceptance: one real MuJoCo gripper-lift task drives the
entire data flywheel with real components — no mocks anywhere in the chain:

```
Round 1   closure 0.010m -> fingers never reach the box (grasp slip)
          -> ExecutionReceipt R1 (kernel contracts, SIMULATION mode)
          -> Practice session P1 (real Recorder path, mock=False)
          -> critic judgment FAILED -> auto-distilled Failure Memory M1
Recovery  How rule "close fingers fully" -> retry at 0.030m -> SUCCESS
          -> Receipt R2 -> Intervention Memory M2
Round 2   fresh slip (new seed) -> retrieval hits M2 -> historical
          recovery applied -> SUCCESS -> Receipt R3
Insight   repeated grasp_slip -> MemoryInsightService auto-emits
          repeated_failure + similar_failure_with_patch
Evolution AutoSubscriber auto-creates the memory-guided Proposal
          -> Patch -> Experiment
Darwin    independent 6-seed x 2-arm A/B on real physics; every episode
          carries a replay-verifiable grasp receipt
Promotion Champion (level sim) through the real PromotionGate
Lineage   `rosclaw data lineage champion:<id>` renders the §43 tree
```

Final tree (real run):

```
Champion champ_…
└─ promoted_from Evaluation eval_…
   ├─ evaluated_from Experiment exp_…
   │  └─ derived_from Patch patch_…
   │     └─ patched_from Proposal prop_…
   │        └─ proposed_from Memory_Insight ins_…
   │           ├─ derived_from Heuristic_Rule rule_golden_grip
   │           ├─ derived_from Memory mem_… (failure)
   │           │  └─ derived_from Episode ep_…
   │           │     ├─ derived_from Receipt act_r1_…
   │           │     └─ observed_in Practice prac_…
   │           └─ derived_from Memory mem_… (intervention)
   │              └─ derived_from Episode ep_…
   │                 ├─ derived_from Receipt act_r2_…
   │                 └─ observed_in Practice prac_…
   └─ supported_by Darwin_Benchmark grasp_receipt_{baseline,candidate}_{0..5}
```

§44's twelve criteria are asserted individually in
`validation/golden_flywheel/tests/test_golden_flywheel.py`, plus the §43
tree shape (graph edges + rendered text + real CLI round-trip), physics
truthfulness (slip really happened), determinism (two runs -> identical
Darwin metrics), and the gate's negative path (unauthorized promotion
raises `PROMOTION_EVALUATION_AUTHORIZATION_REQUIRED`).

## Product bugs found by the demo and fixed in this PR

The golden demo did its job: it caught four real defects that unit tests
had masked.

1. **`insights.py::_find_proven_fix` queried columns that only exist in the
   InMemory fake** (`failure_type` / `parameter_patch`).  The real
   `heuristic_rules` schema is `failure_signature` / `action_template`
   (DF-20 had noted the mismatch as latent).  Against any real store the
   lookup raised, was swallowed, and `similar_failure_with_patch` could
   never fire — the DF-11 insight→proposal loop was dead in production.
   Fixed to the real columns; the intervention-memory fallback now also
   matches outcome case-insensitively and requires the failure type in the
   document.  (`tests/memory/v2/test_insights*.py` seeds updated to the
   real columns.)

2. **AutoSubscriber's auto-created proposal carried no MemoryInsight
   provenance.**  `create_proposal` was called without `source_refs`, so
   lineage fell back to the (episode-id-bearing) failure argument and the
   §9.3 `Proposal --proposed_from--> MemoryInsight` edge never formed.
   The subscriber now passes the insight id as a typed source ref.

3. **Coordinator close-time fact-verify deterministically failed every
   session** (the load-dependent flake seen in CI as
   `test_session_close_runs_fact_verify` was this, deterministic at real
   cadence): (a) the verifier ran before the manifest existed — but the
   manifest embeds the verify report, so presence can't be required at
   close; (b) the coordinator's catalog batches event writes and the
   verifier counted 0 committed events.  Combined with the DF-19 quality
   policy, this quarantined **every** distilled memory on real sessions.
   Fix: `PracticeVerifier.verify(expect_manifest=False)` for the
   close-time pass (post-hoc CLI/reconciler keep the strict default) +
   `catalog.flush()` before verifying.

4. **`MemoryInsightService` failure-vocabulary mismatch**: critic payloads
   say `grasp_slip`, distilled documents say "grasp slip".  Insight
   memory_refs now match both, and the emitted insight derives from the
   failure memories AND the successful intervention memory (not just the
   How rule) — that's what puts both memory branches into the §43 tree.

## Design decisions (disclosed)

- **Closure-waypoint as the skill parameter.**  The PromotionGate pairs
  baseline/candidate by receipt contract: identical model, identical
  request modulo `trajectory`.  A force cap lives in the MJCF (would break
  `model_hash` equality); the finger-closure waypoint IS the trajectory.
  So the demo's parameter is `grip_closure_m` (0.010m slips — fingers
  never contact; 0.030m holds — capped 2.0N squeeze), not a force setpoint.
  Physics tuned: transition band found by sweep; margins chosen far from
  it; ramped lift (a step command jerks even a good grip loose).
- **Grasp receipt verifier at the gate's designed extension point.**
  `PromotionGate(receipt_verifier=...)` exists for non-trajectory-backend
  tasks.  `verify_grasp_receipt` enforces the same contract shape
  (variant/pair/seed/randomization/hashes) plus deterministic replay:
  re-run the same (seed, closure) and require the recorded outcome to
  reproduce exactly (MuJoCo is deterministic per build).  All gate logic
  (pairing, metric provenance, robustness, regression suite, sim ceiling)
  runs unmodified.
- **darwin.runner's deterministic fallback is untouched** — the demo's
  Darwin stage runs real rollouts; the fallback stays for unit tests.
- **Task safety limit vs servo cap.**  The lift transient legitimately
  spikes measured contact force to ~2.04N against the 2.0N servo cap.
  `is_safe` compares against the task crush-safety limit (4.0N), not servo
  bookkeeping.
- **Embedded SeekDB one-path limit**: pylibseekdb allows one embedded
  target per process; a second in-process demo run (determinism test)
  honestly degrades to the sqlite lexical lane, disclosed in the result.

## Files

- NEW `validation/golden_flywheel/` — `scripts/grasp_task.py` (task +
  receipt + verifier), `scripts/run_golden_flywheel.py` (runner + CLI
  entry), `tests/test_golden_flywheel.py` (6 tests)
- `src/rosclaw/memory/insights.py` — real columns + full memory lineage
- `src/rosclaw/evolution/orchestrator/events/subscribers.py` — insight
  source_refs
- `src/rosclaw/practice/verifier.py` — `expect_manifest` flag
- `src/rosclaw/practice/coordinator.py` — flush before close-time verify
- `tests/memory/v2/test_insights.py`, `test_insights_2.py` — real-column
  seeds

## Validation

- `validation/golden_flywheel/tests`: 6/6 pass (module-scoped demo run +
  determinism re-run + gate negative path)
- Related regression: tests/{memory,unit/auto,evolution,practice,how,
  storage} + validation/data_flywheel — green
- `ruff check` on all touched files clean
