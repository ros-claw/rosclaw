# Declared Artifact Schema (Native ADR)

## Context

`rosclaw_deliver` accepts an opt-in `schema_path` declaring a deliberately
bounded local JSON schema subset. The parent implementation silently accepted
malformed *supported-keyword* values, read oversized inputs fully before
rejecting them, and let 5000-level nesting escape as a raw interpreter
`RecursionError`. This change hardens the real keyword-shape / read / depth
paths in the private Python bridge while preserving the original 22-test
legacy behavior exactly.

## Decision

1. **Keyword-shape admission before instance validation**
   (`_check_declared_schema` in `src/rosclaw/agentd/pi_bridge/tool_dispatch.py`):
   every permitted keyword's value is validated up front; invalid values are
   typed rejections, never silently ignored.
   - `type`: known type name or a nonempty list of unique known names.
   - `required`: array of unique strings (empty allowed).
   - `enum`: nonempty array of semantic-unique JSON values — numeric `1 == 1.0`
     collides, `true` is distinct from `1`, object key order is irrelevant,
     ordered arrays stay distinct.
   - `properties` / `items`: object child schemas; `additionalProperties` is a
     boolean or an object child schema (any other value rejected).
   - `minItems`/`maxItems`/`minLength`/`maxLength`: nonnegative integers,
     booleans and fractions rejected.
   - `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`: finite
     int/float, booleans rejected.
   - `title`/`description`: strings only.
   - All `$ref` (including local/cyclic), remote/file resolution,
     branching/composition/conditionals, regex/pattern/format and unknown
     validation keywords remain typed rejections before admission.
2. **Bounded reads** (`_read_declared_bounded`): each schema/artifact is read
   at most `cap + 1` bytes in a single bounded read; oversize is rejected
   without materializing the whole file. No new network or file-resolution
   authority; the permitted workspace roots are unchanged.
3. **Depth pre-scan** (`_declared_text_depth`): structural depth is scanned
   iteratively over decoded text (skipping string literals and escapes)
   *before* the recursive JSON decoder runs. Valid JSON nested beyond 64
   levels — including 5000 levels under the byte caps — returns the bounded
   typed `DECLARED_SCHEMA_BUDGET_EXCEEDED` error instead of a generic
   `RecursionError`.

## Invariants preserved

- Any rejection happens before registration/admission: zero NEW
  Task/Artifact/Op/revision/session-binding rows; existing rows and input
  files unchanged.
- Diagnostics stay bounded (8 errors, 160-char structural paths) and never
  echo artifact/schema values or property names.
- Absent `schema_path` keeps legacy registration/roles/current-task behavior
  byte-for-byte; no nonce aliases or normalization; no general JSON Schema
  compliance claim; no physical/server security claims.

## Evidence

See `reports/declared-artifact-schema-evidence.json`. Current-byte checker
runs: `--mode format`, `--mode check`, `--mode regression` — all
`PASS_NATIVE_PROVIDER_PROGRESS_SOFTWARE_CHECKS` (50/50 factory controls GREEN,
including the 4 keyword-shape, 2 bounded-read and 2 typed-depth counters;
459 compiled Node tests: 456 pass / 3 skipped / 0 fail; 44 private Python
bridge tests pass; Ruff check/format clean on both mutable Python files).
