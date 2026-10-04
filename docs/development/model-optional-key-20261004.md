# Optional model key schema repair (2026-10-04)

Parent: `44403f2f2d83650fec0f28d4c92d8d5ebdf1e3b1` (same source content as
root `e569459a`), based on PI 1.0.2 baseline `16c0d36b`.

The setup contract allows `local` credentials to be omitted. The prior writer
nevertheless emitted `apiKey: ""`. Installed PI's public schema declares the field
optional, but a supplied string must have at least one character
(`dist/core/model-config.js`, `ProviderConfigSchema`). Its bundled official
`docs/models.md` explains that an explicit dummy key makes an Ollama model
available to PI; that is an explicit configuration choice, not something this
writer should fabricate.

Actual private RED evidence is recorded in
`/tmp/rosclaw-empty-apikey-audit-smybadl7/results.json` (file and directory fsync).
`ModelRuntime.create()` did **not** throw for the empty-string case. It returned
a runtime with `getError()` reporting schema rejection and the custom model
absent. This is configuration rejection, not a reproduced SDK startup crash.

The minimal writer change omits a new absent key, removes an existing empty
string when no new reference is supplied, retains existing nonempty references
under the previous behavior, and replaces them when a new `env:NAME` reference
is explicitly supplied. It adds no dummy credential, account migration, auth
fallback or availability claim. Other invalid existing credential values are
outside this change.

The public PI 1.0.2 SDK fixture resolves only private temporary configuration,
uses absent auth files, and disables model-network refresh. It makes no model
requests and contacts no endpoint. Observed distinctions:

| Configuration | Schema | Model parsed | Available | Auth resolution |
| --- | --- | --- | --- | --- |
| Empty string (old generated form) | Rejected | No | Empty | Absent |
| Omitted key (fixed form) | Valid | Yes | Empty | Absent |
| Environment reference, variable missing | Valid | Yes | Empty | Explicit resolution failure |
| Environment reference, private test value present | Valid | Yes | Listed | Resolved metadata |

The last row is local credential/catalog bookkeeping, not proof of an endpoint
accepting a request. The writer repair does not establish usable unauthenticated
local inference; the existing PI auth requirement remains visible.

Set `ROSCLAW_PI_SDK_MODULE` to the absolute installed SDK `dist/index.js` path
and run the following from the worktree with the repository's Python environment:

```sh
PYTHONPATH=$PWD/src python -m pytest \
  tests/agentd/test_pi_model_optional_key.py \
  tests/agentd/test_pi_model_input_preservation.py \
  tests/agentd/test_p1a1_pi_single_source.py \
  tests/agentd/test_pr1_credential_unified.py \
  tests/agentd/test_r07_setup_readiness.py \
  -q --tb=short -W error::pytest.PytestUnraisableExceptionWarning
```

Result with actual installed PI 1.0.2: **55 passed** (six new fixtures, existing
19 input-preservation fixtures retained). Without an explicitly supplied SDK
module, both SDK fixtures skip visibly. Python schema tests still run.
