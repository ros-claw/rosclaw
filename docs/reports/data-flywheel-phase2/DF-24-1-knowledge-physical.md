# DF-24.1 — Knowledge physical cleanup (implementation note)

Phase-II last-mile per `seekdb优化v2.md` §十 DF-16.1:
`rosclaw.know` → `rosclaw.knowledge.legacy`, import shim retained.

## What landed

- **Physical move**: `src/rosclaw/know/` → `src/rosclaw/knowledge/legacy/`
  (git mv, history preserved).  The DF-09 alias module
  (`knowledge/legacy.py`) is dissolved INTO the moved package's
  `__init__.py`: the DF-09 re-exports (KnowledgeInterface,
  LegacyKnowledgeRuntime, TaskCard, EmbodimentCard, VerifierCard) merged
  with the package's existing exports (AssetsLoader,
  KnowledgeBatchEngine, task_pack_for, lifecycle names).
- **Intra-package imports rewritten** to the canonical absolute path;
  logger names kept (`rosclaw.know.*`) deliberately — log-grep compat.
- **Deprecation shim** `src/rosclaw/know/`: package `__init__.py`
  re-exports the same objects plus one shim module per moved module
  (12 shims that alias `sys.modules` so `rosclaw.know.interface is
  rosclaw.knowledge.legacy.interface` — identity, not copies).
- **In-repo migration**: all src/ + tests/ importers rewritten to
  `rosclaw.knowledge.legacy.*`; a source-discipline test
  (`test_no_new_dependencies_on_shim_inside_src`) pins zero shim imports
  inside src.
- `tests/knowledge/test_consolidation.py`'s existing
  `import rosclaw.know` stays intentionally — it is the shim-compat
  check.

## Tests

`tests/knowledge/test_physical_move_shim.py` (4): canonical package
runtime, shim identity (same objects, same modules), src discipline.
Affected suites: 391 passed; full local regression running.

## Next in the DF-24 series

- DF-16.2: `rosclaw.auto` → `rosclaw.evolution.orchestrator` (CLI
  `rosclaw auto` stays).
- DF-16.3: `rosclaw.memory.v2.*` → `rosclaw.memory.*` — the DATA schema
  name stays `memory.v2`; source layout and protocol version are
  different things.

Deprecation countdown starts only after all three land.
