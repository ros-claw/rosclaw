# Golden Flywheel (DF-25, phase-II §42-§44)

The final Phase II acceptance: one real MuJoCo gripper-lift task drives the
whole data flywheel — Practice → Memory → How → Insight → Evolution →
Darwin → Promotion → Lineage — with real components and no mocks.

## Run

```bash
python validation/golden_flywheel/scripts/run_golden_flywheel.py --workdir /tmp/golden_flywheel
pytest validation/golden_flywheel/tests -q
```

The runner prints the §43 lineage tree and the twelve §44 criteria with
PASS/FAIL each, exits non-zero if any criterion fails, and writes the full
result to `<workdir>/golden_flywheel_result.json`.

## What the tests prove

- `test_real_physics_drove_the_story` — round 1 really slips, recovery and
  round 2 really succeed, Darwin candidate arm really beats baseline.
- `test_twelve_acceptance_criteria` — §44 verbatim: real ExecutionReceipt,
  real Recorder data, auto-distilled memory, memory evidence, retrieval
  hit, verifier-passed recovery, auto-generated insight, proposal
  provenance, experiment lineage, independent Darwin, promotion gate,
  champion→receipt trace.
- `test_lineage_tree_matches_spec_section43` — graph edges, rendered tree,
  and the real `rosclaw data lineage champion:<id>` CLI output.
- `test_cli_lineage_command` — the §43 command verbatim.
- `test_determinism` — two runs, identical Darwin metrics (the second run
  honestly degrades to the sqlite lexical retrieval lane: pylibseekdb
  allows one embedded target per process).
- `test_promotion_gate_blocks_unauthorized` — the gate's negative path.

## Design notes

See `docs/reports/data-flywheel-phase2/DF-25-golden-flywheel.md` — including
the four product bugs this demo caught (dead insight→proposal column
mismatch, missing proposal provenance, deterministic close-time
fact-verify failure quarantining all memories, failure-vocabulary
mismatch) and the closure-vs-force design rationale for the promotion
gate's receipt contract.
