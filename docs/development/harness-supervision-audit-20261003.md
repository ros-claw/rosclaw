# Native chat supervision audit (2026-10-03)

This follow-up reviewed the entire original M20 PI transcript, supervised a new
M21 simulation task through ROSClaw, and exercised the native Kimi provider.
Scenario implementation belongs to ROSClaw; reviewer code only inspects evidence
and repairs the framework. No hardware execution was performed.

## Reproducible transcript inspection

Run `python scripts/audit_pi_session.py SESSION.jsonl --out audit.json`.
The script reads every entry, records the snapshot SHA-256, checks parent/tool
relationships, and lists error, abort and completion-claim line numbers. It does
not emit message contents, tool arguments or credentials. A structural PASS is
explicitly **not** a semantic or physical-success verdict. An active transcript
changes after the snapshot; rerun the audit for a later state.

The original history contained repeated provider timeouts, large context, and
almost exclusively shell-based implementation. Earlier self-reported physical
PASS claims required independent M20 telemetry/replay checks and corrections.
Native compaction later succeeded with OpenAI at 251,530 input tokens, preserving
the raw history and recent messages. Its summary omitted recent M21 progress;
the retained recent context and an explicit evidence-based progress update were
needed. A successful compaction event alone does not prove summary correctness.

## Framework repairs

- Align the independently packaged `rosclaw-tui` PI dependency with 1.0.0.
  Updating the main agent's four packages had left this package on 0.85.1.
- Run the `bash` tool with Bash and `pipefail`, including inside bubblewrap.
  Brace expansion and failures within a direct pipeline now follow the tool's
  advertised shell contract. An explicitly nested `bash -c` still owns its own
  shell options.
- Reject an overlong `ROSCLAW_HOME` Unix socket path before kernel startup, with
  an actionable error. This does not add support for arbitrarily long homes.
- Add bounded optional provider wait overrides, in milliseconds:
  `ROSCLAW_PROVIDER_FIRST_TOKEN_TIMEOUT_MS` and
  `ROSCLAW_PROVIDER_STREAM_IDLE_TIMEOUT_MS`. Accepted range is 1,000–3,600,000;
  invalid values preserve the existing 30s/45s defaults. Progress notices remain
  active, and tool execution does not count as provider inactivity.
- Show supported patch and controller payload examples in simulation CLI help,
  including unsupported online policy callbacks.
- Add a native `kimi-coding` HarnessBench profile using `anthropic-messages`.
  Both comparison legs use the same profile settings and an environment key
  reference; credentials are not written to generated model configuration.
- Fix the PTY progress test to wait for an actual numbered progress line rather
  than matching the command echo containing `progress-step-$i`.
- Prefer the reported repair candidate before running the existing independent
  lineage, audit and strict replay gates. The oracle previously chose the first
  passing model in store order, falsely rejecting a later verified iteration.
  Unverified, unrelated and broken-original claims remain rejected.

The formal chat needed a restart with the operator's proxy environment: its old
Node process had no proxy variables to inherit. Provider proxy support cannot
invent a missing parent environment. Access-only OpenAI OAuth was used for the
comparison and compaction; no Codex refresh token was rotated or copied.

## Evaluation boundaries

An eight-task native Kimi run at a 180s settle budget independently verified U01
and D01; R01, E01, H01, V01, I01 and S01 did not settle within budget. Their tool
histories show work, so these are not all provider connection failures. This is
2/8 verification under those conditions, not a general model-quality ranking.
The formal chat and benchmark ran concurrently, and the benchmark used medium
thinking rather than the formal chat's high setting. Longer-budget follow-ups
must be reported separately. The 48-question/144-repetition matrix is a plan;
questions without executed runs and dedicated oracles are NOT_RUN.

At the extended 600s budget, R01 completed in 526.7s with two repair iterations.
The original oracle incorrectly reported `claimed_ref_mismatch` against the
first iteration. Reverification of the unchanged answer and artifacts with the
candidate-selection fix verified the reported second model. The original result
and separate correction record are retained; this is not a new model run.

The first benchmark also exposed the Unix socket path limit. Keep benchmark
runtime homes short and copy completed file evidence to durable storage.
Bubblewrap was unavailable on this machine; tool-layer checks are not proof of
OS isolation or protection against shell-accessible credentials/control files.

M21 uses a separate Python MuJoCo loop with official policy assets. It does not
prove native multi-robot rollout/receipt integration. A simplified 12-DOF G1
with a torso-mounted racket must be identified as such. Stage A locomotion is
not ball interception, real racket contact, a legal tennis rally, or strategic
agent competition. Initial state setup must be distinguished from prohibited
mid-episode state/ball-velocity manipulation.

Default chat task attachment also reopened the finished M20 task when the new
mission was first sent. Explicit `/newtask` subsequently bound M21 correctly;
the earlier state transition remains in the audit history. Use `/newtask` for
an intentionally separate mission rather than assuming prose changes identity.

A native SeekDB pagination test inserting 2,300 records individually exceeded
the diagnostic time budget inside its database upsert call. Its result remains
unresolved, not PASS; targeted integration/performance investigation is needed.
Broad non-live regression excludes native SeekDB modules only after retaining
this separate failure/performance evidence. Local reports contain exact run
counts and artifacts; repository-wide pre-existing lint debt is not green.

Validation completed: broad non-live regression 1,848 passed / 7 skipped /
24 deselected; daemon/kernel/MCP boundary checks 370 passed / 1 skipped; main
TypeScript package 247 passed / 3 skipped; independent TUI package 27 passed;
oracle repair/adversarial regression 59 passed. Some suites overlap; these are
separate results, not an additive count of distinct tests. Configured mypy scope
passed 121 files. Modified Python lint and `git diff --check` passed.
