# PI 1.4.0 compatibility validation

Upgrade the four exact PI dependencies and overrides together from 1.3.0 to
1.4.0, including the terminal package. The reviewed upstream `_startRun` and
forced prompt anchors still match; patch replacement behavior is unchanged.
The lock preserves every unrelated dependency entry. The only new dependency
is `marked-cjk-friendly@0.1.2`, required by PI TUI. Official npm integrities bind
the release; an upstream source commit has not been independently verified.

Offline validation on the final minimal dependency tree: Agent build and
790 tests passed, with 3 existing bwrap-dependent skips; TUI build and 29 tests
passed; 19 Python dependency/runtime boundary tests passed. Agent test fixtures
used their existing owned-tool contract with real hash-checked rg and fd. An
initial run failed to locate rg before two test files ran; test assertions were
unchanged. The final suite ran after removing unrelated npm dependency drift.

PI 1.4.0 raises summary output allowances and changes retry, tool recovery, and
resume behavior. ROSClaw's opt-in model request admission rejects summary
dispatch and disables cache warming/retries. Without that policy, summaries
can become larger. Neither local admission nor the observer proves a provider
billing or remote output limit. Offline tests do not establish live ChatGPT
quota handling, summary behavior, or robot capability. A real ROSClaw Python CLI smoke on code commit `fc19a84d6f9bc6455b64cf0d0dfdae76fcb8ab4c`
subsequently passed with ChatGPT `gpt-6.1-sol` at low effort: one actual file
read, normal quit, same-session UUID resume, correct recall with zero resumed
tools, and normal quit again. All captured processes exited without signals.
Three real requests used 5,543 inclusive input tokens and 107 output tokens;
the entire worker took 47.61 seconds. These are observed usage, not a billing
bound. Compaction and truncated-response recovery remain untested live.
The isolated read/resume smoke itself did not activate a default; the ordinary launcher was subsequently switched to this candidate after verification.

Primary release sources:
- https://registry.npmjs.org/@earendil-works/pi-coding-agent/1.4.0
- https://github.com/earendil-works/pi/blob/v1.4.0/packages/coding-agent/CHANGELOG.md

Local reproducibility evidence is retained outside the checkout at
`/tmp/ROSCLAW_PI140_UPGRADE_VERIFICATION_20261023_v2_MINIMAL_LOCK/`: exact commands,
complete logs, dependency trees, semantic lock changes, and SHA-256 manifest.
Exact byte copies also live under the persistent tennis audit directory in
`ROOT_PI140_OFFLINE_VALIDATION_WHOLE_EVIDENCE_DURABLE_ARCHIVE_NOT_LIVE_ACTUAL_20261023_v1`.
The live receipt and independent Root transcript checks are in
`ROOT_PI140_TRUE_CLI_READ_SAME_UUID_RESUME_ACTUAL_20261023_v1`.
Private credentials and raw session reasoning are excluded from publication.

## Ordinary launcher nonzero-result observation

After Root restored the ordinary ROSClaw launcher, a separate finite CLI probe
used `gpt-6.1-sol` at high effort and admitted exactly one bash command, `exit 7`.
The closed session persisted `toolResult.isError: true`, executor `exitCode: 7`,
no signal, timeout or abort, and the correct final answer `PI140_BASH_EXIT_7`.
The product quit normally and all explicitly captured processes were gone
without cleanup signals. Two real requests used 3,597 inclusive input tokens
and 31 output tokens; these are observed usage, not a provider billing bound.

The original probe and its supervisor remain **CLOSED1**, not PASS. Its first
assertion incorrectly required the exit header at text position zero: the
installed product correctly prepended its source-declared SIM sandbox-degraded
notice. Its postcheck also incorrectly required an empty workspace despite
product-owned telemetry. A separate closed-evidence review checked the original
persisted result and these two harness mismatches; it did not rerun the model or
rewrite the failed receipt. The SIM shell was explicitly not OS sandboxed.

This is one current-runtime nonzero-exit observation, not proof that every
historical error, cancellation or timeout path is fixed. It does not establish
robot, SDK, physics, compaction or recovery capability. Safe selected metadata
is retained outside the checkout in
`ROOT_PI140_CLOSED_ORDINARY_EXIT7_INDEPENDENT_VISIBLE_FACTS_NOT_PROBE_PASS_DURABLE_20261023_v1/INDEPENDENT_CLOSED_REAL_BASH_EXIT7_VISIBLE_FACTS_NOT_REGRADE.json`;
private credentials, raw session reasoning and runtime authority are excluded.

## HarnessBench readonly OAuth integration on the complete 1.4 baseline

The isolated integration starts from `5063c76463858567b125fad3488b60b7dc1f1789`,
retains its model request admission and UI behavior, and applies the reviewed
HarnessBench commits `8eae45d8725d79278f0d93b3ad5d9bf49485df1b` and
`c9635597d4388cd66f7de00bd12f55bb9e3390dc`. The only merge conflict adds the
readonly opt-in import and branch while preserving the original runtime file
byte for byte after removing those additions. Package locks and PI versions
remain unchanged. This is the complete current 1.4 baseline, not the separate
1.0.4-wrapper dependency compatibility experiment.

Both builds passed. The complete current Agent suite passed 810 tests with
3 existing bwrap skips (813 total); TUI passed 29 tests. The selected Python
inventory passed 93 tests: the 74 current feature/runtime checks plus 19
dependency and entry-runtime checks. These are distinct inventories, not an
addition of historical test counts. The real ROSClaw runtime startup fixture
completed with zero prompts or fetches; asynchronous readonly mutations retain
the public Promise contract. Tests use synthetic credentials only.

The first complete Agent run retained two failing older idle-gate assertions
that still expected a task-completion claim without an acceptance scope. Their
fixtures and presentation source were unchanged from 5063. The two original
methods now require the exact unknown-scope delivery notice and activity
instruction and reject goal-success claims; idle gating, exact single delivery
and terminal-event filtering remain asserted. The existing configured-scope
positive presentation tests also pass. Production presentation code was not
changed. The initial two failures and complete logs remain archived.

This integration qualification made no model, credential-source, robot or SDK
calls. It does not activate or publish the isolated checkout and is not evidence
that a real A/B matrix completed. Complete commands, the preserved conflict,
source inverses and both test runs are retained in
`/tmp/ROSCLAW_DEFAULT5063_PI140_FEATURE_V6_MIGRATION_OFFLINE_20261024_v1`.
