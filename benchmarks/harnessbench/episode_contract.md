# Prepared native episode bindings

Run this offline preflight before signing a unique GO:

```bash
python -m benchmarks.harnessbench.episode_contract \
  --spec protocol/spec.json \
  --schema protocol/ROOT_GO_REQUIRED_SCHEMA.json \
  --prompt protocol/public_native_prompt.txt
```

Add `--go PATH` to check an existing GO's identifiers. The command never creates
or authorizes a GO. It checks the spec's source, case ID and nonce against the
schema's required constants, the phase constant, and every explicit `case_id`
and `runtime_nonce` assignment in the prompt. Assignments may be plain text or
JSON; identifiers use ASCII letters, digits, underscores and hyphens. Conflicting
assignments, missing labels and stale prefixes are refused with exit code 2.

This guard addresses a real preparation failure: replacing a nonce left the
prompt's `FILEUX_<nonce>` case ID different from the spec's `FILEUX_V4_<nonce>`.
Hash checks passed, but the native agent would have been told to deliver the
wrong case. Check both metadata consistency and hashes.

This is one preparation gate. Canonical entrypoint and dependency qualification,
actual SDK startup, provider routing, durable budgets, process cleanup and the
task's independent oracle still require separate validation. Passing this guard
does not verify an agent's answer or a physical capability.
