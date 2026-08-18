# DF-22 — Memory Insight 2.0 (implementation note)

Phase-II P1 per `seekdb优化v2.md` §33.

## What landed

### Detectors (MemoryInsightService)

- **`known_dead_end_revisited`** (the endorsed one): subscribes to
  `rosclaw.auto.proposal.created`, matches the hypothesis against
  registered DeadEnds (token Jaccard over direction+rejection_reason,
  CJK bigrams included, +0.2 skill-name boost, ≥0.5 to fire).  Carries
  `recommended_action`: **skip** (≥0.8) / **narrow_search** (≥0.5),
  `dead_end_refs`, `similarity`, plus a `derived_from` lineage edge to
  the dead_end entity.
- **`skill_regression` / `skill_improvement`**: rolling 20-outcome
  window per skill; regression when the first half ≥0.6 and the recent
  half ≤0.3 (improvement mirrors).
- **`harmful_recovery_pattern`**: a proven fix exists yet failures
  reach 2× the threshold — the fix is not working; emitted as an
  escalation AFTER the patch hint (dedup is per type; the test pins
  ordering).

### Auto side — Evolution stops re-hitting the wall

- `AutoEngine.apply_dead_end_guard(proposal_id, insight)`: skip →
  status `rejected`; narrow_search → `dead_end_narrow_review` gate;
  otherwise `dead_end_stronger_evidence` gate.  The guard (insight id,
  refs, similarity) is recorded on the proposal so the evaluation path
  can see WHY.
- `AutoSubscriber._on_memory_insight` dispatches the new type to the
  guard instead of minting yet another proposal.

### Bug found by the full-loop test

`AutoPublisher`'s events lost their subclass fields on the bus:
`EventEnvelope.to_dict()` only serializes envelope fields, so
`proposal_id` / `hypothesis_statement` / `metrics` / `direction`
arrived as an empty `payload`.  All three publishers now fill the
payload dict explicitly (additive, compat-safe).  The insight detector
reads envelope fields + nested payload.

## Tests — `tests/memory/v2/test_insights_2.py` (9)

skip/narrow/no-emit tiers, lineage link, regression+improvement,
harmful escalation ordering, guard actions (skip/narrow/stronger/
missing), and the full bus loop proposal→insight→guard on the SAME
proposal.  DF-11 + unit/auto suites: 147 passed, ruff/mypy clean.
