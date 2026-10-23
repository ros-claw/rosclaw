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
quota handling, summary behavior, or robot capability. Live read/resume smoke
validation is pending. This candidate does not activate an existing runtime.

Primary release sources:
- https://registry.npmjs.org/@earendil-works/pi-coding-agent/1.4.0
- https://github.com/earendil-works/pi/blob/v1.4.0/packages/coding-agent/CHANGELOG.md

Local reproducibility evidence is retained outside the checkout at
`/tmp/ROSCLAW_PI140_UPGRADE_VERIFICATION_20261023_v2_MINIMAL_LOCK/`: exact commands,
complete logs, dependency trees, semantic lock changes, and SHA-256 manifest.
