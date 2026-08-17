# DF-17 — Full Evolution Lineage (implementation note)

Phase-II P0 per `seekdb优化v2.md` §8–§14. Closes gap P0-2: lineage had a
table and a repository, but not a full typed graph.

## What landed

### §8 Typed vocabulary — `src/rosclaw/storage/lineage_types.py` (new)

- `LineageEntityType` StrEnum: action / receipt / practice / episode /
  memory / memory_insight / failure / diagnosis / proposal / patch /
  experiment / darwin_benchmark / evaluation / champion / dead_end / skill.
- `LineageRelation` StrEnum: derived_from / generated_from / observed_in /
  supported_by / diagnosed_from / proposed_from / patched_from / tested_by /
  evaluated_from / promoted_from / rejected_from / recovered_by / supersedes.
- Canonical orientation (§8.5): **child/derived → parent/source**. The
  ReceiptProjector's existing receipt → action edge already matches this.

### §8.3–8.4, §11 LineageRepository — `src/rosclaw/storage/lineage.py`

- `parents()`/`children()` now filter by `entity_type` AND `entity_id`;
  before, the type parameter was decorative.
- Idempotency key extended to the 5-field tuple
  (from_type, from_id, relation, to_type, to_id).
- `ancestors()`/`descendants()`/`trace()` walk (type, id) pairs, not bare
  ids, so same-id-different-type entities cannot cross-contaminate a walk.
- New `trace_graph(type, id, max_depth=16, max_nodes=500)`: full ancestry
  DAG (nodes + edges + `truncated` flag), cycle-safe, capped so a
  pathological graph cannot blow up the CLI.

### §9 AutoEngine auto-linking — `src/rosclaw/auto/engine/auto_engine.py`

- New constructor kwarg `lineage_repository` (default None → standalone
  behavior unchanged), threaded through `AutoPlugin`.
- Every edge write is best-effort (`contextlib.suppress`) — lineage can
  never break the evolution flow:
  - Failure --generated_from--> action (praxis event); plus
    --derived_from--> memory_insight when `evidence["insight_id"]` present.
  - Diagnosis --diagnosed_from--> failure.
  - Proposal --proposed_from--> typed `source_refs` (new kwarg;
    falls back to the `failure_case_id` argument).
  - Patch --patched_from--> proposal.
  - Experiment --derived_from--> patch (graph kept minimal; proposal is
    reachable through the patch).
  - Evaluation --evaluated_from--> experiment, plus --supported_by--> each
    simulation receipt (`darwin_benchmark` type when the receipt carries a
    benchmark marker).
  - Champion --promoted_from--> evaluation.
  - DeadEnd --rejected_from--> typed source via new `source_type`/`source_id`
    kwargs.

### §10 MemoryInsight → Memory — `src/rosclaw/memory/insights.py`

- `MemoryInsightService` accepts `lineage_repository`; every emitted
  insight links --derived_from--> each ref in `memory_refs` (all of them,
  not just the JSON attribute). `heuristic_rules:<id>` refs link with type
  `heuristic_rule`; bare ids link as `memory`.

### §12 CLI — `rosclaw data lineage <type>:<id>`

- New `data` command group (`src/rosclaw/storage/cli.py`,
  `add_data_subparser`), wired in `src/rosclaw/cli.py`.
- Text output renders the DAG as an indented tree with relation labels;
  nodes revisited through a second path render "(see above)" so cycles
  stay finite. `--json` emits the raw trace_graph payload for the
  Dashboard. `--max-depth`/`--max-nodes` override the caps.

### Runtime wiring — `src/rosclaw/core/runtime.py`

- The data plane's `LineageRepository` (DF-14) is now injected into both
  `AutoPlugin` and `MemoryInsightService`. Ordering is safe:
  `_create_data_plane` (which builds the repository) runs at the top of
  `_do_initialize`, before module wiring.

## Tests (§13 + §14) — `tests/storage/test_full_lineage.py`

Named set: `test_lineage_type_sensitive`, `test_lineage_idempotent`,
`test_lineage_multi_parent`, `test_lineage_cycle_safe`,
`test_trace_graph_branches`, `test_proposal_failure_link`,
`test_patch_proposal_link`, `test_experiment_patch_link`,
`test_evaluation_experiment_link`, `test_champion_evaluation_link`,
`test_deadend_source_link` — plus `source_refs` proposal linking, the
no-repo no-op path, insight→memory multi-link, CLI tree rendering, and
CLI bad-entity rejection.

Golden E2E (`test_golden_lineage_champion_to_receipt`): Receipt → Episode
→ Memory → Insight → Proposal → Patch → Experiment → Evaluation →
Champion, then `trace_graph("champion", id)` contains the receipt, memory,
proposal, evaluation, insight, and episode ids. 六问之六 passes.

## Deliberate deviations / notes

- Experiment links only to its patch (not also the proposal), per §9.5's
  "keep the graph minimal" option — the proposal is one hop away.
- `source_refs` accepts dicts (`{"type", "id"}`) or objects with
  `type`/`id` attributes, so a future typed EntityRef drops in without a
  signature change.
- Baseline champions created without an `evaluation_id` simply get no
  promoted_from edge (empty-id guard), which matches their provenance-free
  nature.
