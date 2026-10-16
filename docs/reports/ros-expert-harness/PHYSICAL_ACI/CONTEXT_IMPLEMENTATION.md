# P3a: evidence-bound current decision state

The old G5 protocol reconstructs full transcript each turn and mandates an
explicit model Memory query in C. This opt-in implementation preserves that
runner, TaskTools policy and the original 54 verdicts. A future preregistered
runner can use `EngineeringEpisodeContext` to send the current task card, phase,
revision, proposal IDs, verification refs and latest outcome instead. Full
requests and results remain in `PhysicalACI.trace`; the host must persist them.
Token figures are estimates, never provider usage or claims of model benefit.

## Existing agentd integration

`SourceBundle.extra["ros_engineering_case"]`, with its validated
`ros_system_model`, adds a compact case to the existing protected L2 dynamic Self
layer. The existing Body, Mission, capability and consent checks still apply.
Case hash, ROS snapshot and mission Body must match. Historical SEALED_REPLAY is
rejected. A live OBSERVATION case must be compiled at exactly the context review
clock: checking only snapshot age could preserve a scan that has since crossed
its freshness limit. The case hash contributes to Self and bundle identity;
case change/removal triggers recompilation. Protected-layer budget exhaustion
fails rather than cutting facts into misleading readiness. No capability or
authority is granted. Callers without this extra source keep existing behavior.

## Benchmark frames and Memory

`EngineeringEpisodeContext(aci)` performs no model calls and starts no World.
Frame construction reads current sealed state without adding hidden tool calls.
Each explicit call advances a revision and records original result bytes,
operation counts and Memory citations. `finish` still only reports. Verification
is explicitly scoped to a different World and never updates original-case L1.
State overflow requires operator review rather than silently trimming evidence.

A/B do not retrieve. C performs one host-side query through the existing
`TaskTools.memory_query` callback before any model decision, even when admission
is empty. Its signature comes from violated invariants, not evaluation labels.
The required callback is the existing provenance-filtered training interface:
it validates frozen records and original episode bytes, Body compatibility,
evaluation-run isolation and same-case exclusion. This adapter additionally
checks the returned admission receipt and projects at most two short historical
priors. Original unbounded training prose stays outside the frame. References
not actually projected into the episode cannot be cited. Citations record
intent; they do not prove causal benefit or automatically select a remedy.

This is read-only retrieval and state plumbing. Executable mechanism/remedy/
contraindication KNOW/HOW distillation remains a separate P3 task; the legacy
records currently provide diagnosis summaries. This code is not yet wired to a
new model runner, a canonical MCP tool, or a product default. Provider schema
compatibility, fresh read-only probes and the independent pilot remain open.

## Development verification, 2026-10-16

- 54 original snapshots replayed with scripted proposals, zero new models and
  zero new Worlds; old results retained. C used 18 actual retrievals from a copy
  of the frozen SQLite training database, with the original provenance checks.
  Three episodes projected two priors each; the other fifteen projected none.
- Original training database, manifest and reopened records stayed byte-identical.
- Initial frames remained about 2.5k characters and final frames about 2.8k;
  these are development context measurements, not model performance.
- Integration tests cover historic/mismatched/tampered sources, clock drift,
  deterministic hashes, protected budgets, stable frame size across 100 reads,
  empty Memory admission, rejected references and unchanged legacy C policy.

Local replay artifacts are in `recovery_2026-10-16/P3-development-context-54-v1`
outside this Git repository. They are not claimed to be uploaded public data.
No new model pilot is authorized by this development replay: a separate,
reviewable preregistration must define arms, budgets, baseline and stop rules.
