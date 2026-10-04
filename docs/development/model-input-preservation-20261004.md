# Model input configuration audit (2026-10-04)

Baseline: `16c0d36ba70aa588742d78276067cdea22192712`, installed PI 1.0.2.
All checks used private temporary homes and offline public SDK resolution.
No actual model request, physical execution, or live configuration change occurred.

The default `configure_model(home, "kimi-code")` writes only the builtin selector
`kimi-coding/kimi-for-coding`. Its resolved input is `["text", "image"]`.
New custom definitions without `input` resolve to `["text"]`, including custom
models named `k3`. This is PI's documented schema default, not a ROSClaw builtin
capability regression. ROSClaw's session adapter resolves selection through the
same session ModelRuntime; PI's selector uses that runtime's available snapshot.
No independent ROSClaw UI claim of vision support was found in the reviewed code.

One configuration regression was reproduced: rerunning `write_pi_model_config`
on an existing custom route erased the user's explicit `input` field. The public
SDK resolved `["text", "image"]` before setup, then `["text"]` afterwards.
The initial Python fixture run produced **3 failures / 15 passes**.

The fix preserves only a nonempty, unique array of supported input values
(`text`, `image`) from exactly one matching prior model. Provider, model ID,
provider API and base URL must match exactly. Any model-level API/base URL
override must also describe that same route. Changed routes, duplicate model IDs,
and invalid declarations do not transfer input capabilities. Nothing infers a
custom endpoint's capabilities from its model name or a builtin catalog.

The focused suite covers changed provider/ID/API/endpoint, model-level route
overrides, invalid declarations, ambiguity and text-only custom defaults. The
opt-in public SDK test additionally resolves before/after models and the builtin
default using `allowModelNetwork: false, refreshOnCreate: false`, absent auth files,
and fake environment-reference names. It makes no completion or stream request.

Run from the worktree (set `ROSCLAW_PI_SDK_MODULE` to an installed SDK's absolute
`dist/index.js` path; otherwise the SDK test explicitly skips):

```sh
PYTHONPATH=$PWD/src python -m pytest \
  tests/agentd/test_pi_model_input_preservation.py \
  tests/agentd/test_p1a1_pi_single_source.py \
  tests/agentd/test_pr1_credential_unified.py \
  tests/agentd/test_r07_setup_readiness.py \
  -q --tb=short -W error::pytest.PytestUnraisableExceptionWarning
```

With the installed PI 1.0.2 module explicitly supplied: **49 passed**.
Ruff and mypy pass for the changed source. These tests establish configuration
preservation and resolved model metadata, not successful endpoint image handling.

Scope limits: the existing setup writer also reconstructs other model fields
(such as explicit reasoning/compatibility settings); this patch deliberately
does not broaden preservation to those fields. The separate `apiKey: ""`
custom-provider configuration is rejected by PI 1.0.2's schema and remains a
recorded, unresolved issue. Neither limitation is concealed by the input fix.
