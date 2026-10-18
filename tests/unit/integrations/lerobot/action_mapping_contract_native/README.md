# Named action mapping regression contracts

These synthetic contracts were authored through ROSClaw Native against the
unchanged action-mapping API. The original implementation accepted duplicate
policy names and a declared shape inconsistent with the named dimensions. A
nonfinite value in a later action chunk raised `IndexError` instead of producing
a blocked validation report. The production fix rejects these inputs while
retaining valid named permutations, unit conversion, omitted shapes, and the
existing explicit partial-mapping option.

All 14 new test methods and their fixture values remain unchanged in meaning.
Together with the existing 14 methods, normal repository pytest configuration
passes 28 methods and 23 subtests. Production imports were verified against the
publication checkout, with its real parent package initializers.

Public copies received Ruff formatting and import sorting only. One comment-only
B905 exemption preserves a zip loop whose equal lengths are already asserted.
Every other AST node and the import multiset match the original Native output.
`provenance.json` records the original and public source hashes. The empty
`native/__init__.py` is test packaging only.

The original test module's author-time `NOT_RUN` notes are retained as history;
the validation above describes subsequent operator verification. These tests
establish synthetic vector adaptation and rejection behavior, not learned VLA
inference, robot control, or physical safety.

Run from the repository root:

```sh
python -B -m pytest -q tests/unit/integrations/lerobot/test_body_action_mapping.py tests/unit/integrations/lerobot/action_mapping_contract_native
```
