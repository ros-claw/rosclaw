# DF-24.2 — Evolution physical cleanup (implementation note)

Phase-II DF-16.2 per `seekdb优化v2.md` §十: `rosclaw.auto` →
`rosclaw.evolution.orchestrator`.  CLI `rosclaw auto` unchanged.

## What landed

- **Physical move**: `src/rosclaw/auto/` (48 modules) →
  `src/rosclaw/evolution/orchestrator/` via git mv.  Intra-package
  absolute imports rewritten to the canonical path.
- **Deprecation shim** `src/rosclaw/auto/`: per-module shim files plus
  package `__init__` shims that EAGERLY register the canonical modules
  into `sys.modules` AND bind them as attributes — shim and canonical
  paths share ONE module object, so class identity (`is`) holds across
  both paths.  (The naive `from X import *` + late sys.modules swap was
  caught by the identity test: parent-path resolution re-executed the
  module under the shim name, forking class identity.)
- **In-repo migration**: all src/ + tests/ importers rewritten; a
  source-discipline test pins zero shim imports inside src (event-topic
  strings like `"rosclaw.auto.proposal.created"` are not imports and
  stay — they are the wire vocabulary).
- `rosclaw auto` CLI verified unchanged (same subcommand tree).

## Tests

`tests/evolution/test_orchestrator_move_shim.py` (5): canonical
engine/plugin, single-module-object identity, package attribute access,
deep paths (storage/promotion), src discipline.  138 auto+shim tests
pass; full local regression running.
