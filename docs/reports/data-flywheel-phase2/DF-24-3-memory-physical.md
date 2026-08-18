# DF-24.3 — Memory physical cleanup (implementation note)

Phase-II DF-16.3 per `seekdb优化v2.md` §十: `rosclaw.memory.v2.*` →
`rosclaw.memory.*`.  **The DATA schema name stays `memory.v2`**
(`SCHEMA_VERSION = "memory.v2"` unchanged — pinned by test): source
layout version ≠ protocol version.

## What landed

- **Physical move**: 36 modules from `memory/v2/` up into `memory/`
  (no name collisions with the flat v1 files; subpackages
  adapters/regime/runtime_retrieval move as units).  v2's public API
  merged into `memory/__init__.py`.
- **Shim** `rosclaw.memory.v2/`: package `__init__` files eagerly
  register canonical modules into `sys.modules` + bind attributes
  (same identity pattern as DF-24.2) and re-export the v2 public API.
- **Wire vocabulary untouched**: `SCHEMA_VERSION = "memory.v2"` and the
  `rosclaw.memory.v2.*` logger names are RESTORED after the bulk import
  rewrite — log-grep and data-protocol compat stay.
- **Path-depth fix**: `cmd_memory_v2_benchmark`'s repo-root resolution
  gains the new-depth candidate (the move is one level shallower).
- In-repo importers (55 files) migrated; src-discipline test pins
  import statements only.

## Tests

`tests/memory/test_v2_move_shim.py` (5): protocol name unchanged,
module/class identity, public API, deep subpackages, src discipline.
Memory suite: 192 passed.
