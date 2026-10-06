# Registered delivery digest

Native SOURCE testing exposed an extra-call requirement: delivery registered the actual SHA256 internally but returned only artifact ID and size. Retrieving the digest required `rosclaw_artifact_resolve`, consuming a call and additional context.

Successful `rosclaw_deliver` results now include the kernel-registered `sha256` in the model-visible summary and the registered ID in the existing `artifact_refs` field, including supporting artifacts appended after task completion. The digest comes from registration, not a model-supplied value. No lifecycle or wire-schema change is required.

Validation: actual private SQLite registration tests cover UTF-8 bytes, repeated registration, and post-terminal append without task revival. Focused bridge/lifecycle and Practice suites: 218 passed, 9 skipped. Compilation, the new test file Ruff checks, and focused core mypy (50 files) passed. Existing repository-wide lint issues are outside this change. Previous F49 budget and tool-allowlist failures remain recorded.

An isolated native Kimi K3 run on code revision `c177d6095aa34e483daf6933afa32658518a02e1` and PI 1.0.4 subsequently consumed the returned digest and artifact ID, wrote matching metadata, registered its progress report, and stopped naturally. Independent review of frozen SQLite and session entries verified both artifacts belonged to the current task and native session, and that the source registration result preceded the metadata write. All six requests closed; the run used 47,007 cache-inclusive input tokens and 1,022 output tokens. No artifact-resolve call, process operation, or ROS execution was needed.

This qualifies the registered-digest workflow component. The original experiment runner still exited with code 1 because it compared against another scenario's success constant; an additional operator gate also incorrectly required an absolute source path that the public task had not required. Those original results are retained. The additive public-aligned review passed without repeating the model run. It does not certify broader algorithms or physical behavior.
