# Registered delivery digest

Native SOURCE testing exposed an extra-call requirement: delivery registered the actual SHA256 internally but returned only artifact ID and size. Retrieving the digest required `rosclaw_artifact_resolve`, consuming a call and additional context.

Successful `rosclaw_deliver` results now include the kernel-registered `sha256` in the model-visible summary and the registered ID in the existing `artifact_refs` field, including supporting artifacts appended after task completion. The digest comes from registration, not a model-supplied value. No lifecycle or wire-schema change is required.

Validation: actual private SQLite registration tests cover UTF-8 bytes, repeated registration, and post-terminal append without task revival. Focused bridge/lifecycle and Practice suites: 218 passed, 9 skipped. Compilation, the new test file Ruff checks, and focused core mypy (50 files) passed. Existing repository-wide lint issues are outside this change. No paid native run has yet qualified this new revision; previous F49 budget and tool-allowlist failures remain recorded.
