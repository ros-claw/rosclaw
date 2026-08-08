# Native Agent routing

Status: PASS.

Core adds a deterministic intent contract for Body, Memory, Know, How, Skill,
and Action. Thirty fixed natural-language boundary cases passed with:

- Know/Memory wrong routes: 0;
- How/Action wrong routes: 0.

Examples covered current-device versus official specification, previous-run
failure versus community approaches, current remediation versus physical
execution, unknown API/version questions, and explicit motion commands.

Automatic Know triggers are limited to explicit research, similar projects,
papers/upstream sources, time-sensitive APIs, unknown errors, stalled
implementation, and comparison of external approaches. Routine chat does not
automatically research.

Budgets are fixed as:

| Depth | Max sources | Max tokens |
|---|---:|---:|
| shallow (Native default) | 8 | 20,000 |
| standard | 20 | 60,000 |
| deep (explicit request) | 50 | 150,000 |

The ActiveReferenceWorkspace stores IDs and state only: pack ID, active
project IDs, opened evidence IDs, warnings, and stale flag. Wiki prose is not
copied into session state.

Validation: 64/64 Core knowledge tests and 32/32 rosclaw-agent Node tests.
