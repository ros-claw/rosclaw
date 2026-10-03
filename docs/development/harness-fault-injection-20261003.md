# Native harness fault injection, 2026-10-03

This review uses an isolated worktree based on `19368f13`, temporary shell
workspaces and actual pinned PI 1.0.0 libraries. It does not call a model or
execute a robot controller. Live Kimi capability evaluation is a separate cohort.

## Reproduced defects

1. Workspace bash printed failed exit codes, explicit timeouts and cancellation
   in text but omitted `isError`. PI 1.0.0's actual `runToolCall` classified
   nonzero exit, SIGKILL and timeout as successful tool calls. Five initial
   regression cases failed. Shell results now retain `exitCode`, `signal`,
   `timedOut`, `aborted`, optional `spawnError`, and an explicit boolean
   `isError`. Successful SIM degraded execution remains successful; the
   no-OS-sandbox notice alone is not an execution error. An additional native
   `runAgentLoop` test checks both execution-end events and the persisted
   `SessionManager` JSONL: failed exit 7 remains `isError: true`, successful
   exit 0 remains `isError: false`.
2. Closing a user modal during an ongoing tool rearmed stream-idle warnings.
   Abort was guarded, but the user still saw a false Provider-stall notice.
   The reproduced test failed before adding the tool-pause guard to user
   resume. Closing a modal now resumes only when no tools remain active.
3. Nine malformed JSONL field fixtures crashed the session auditor or accepted
   negative token counts. Invalid identifiers, parents, entry types, roles,
   content containers, usage containers/counts and context targets now produce
   redacted `REVIEW_REQUIRED` issues rather than raising or silently passing.
   This remains a structural audit; physical and semantic success are never
   inferred from transcript text.

## Fault coverage and checks

Concrete injected conditions include nonzero exit, process signal, explicit
timeout, cancellation with a TERM trap, throwing heartbeat observer, concurrent
isolated shell outputs, modal/tool interleaving, nested tool/user waits, throwing
notification/abort callbacks, independent concurrent watchdogs, late resume
after end, and the nine malformed field classes. Existing process-tree tests
also verify pipeline descendants are terminated on timeout/cancel and heartbeat
timers stop after completion. The actual PI loop yields exactly one end event
per failed/successful call, and session persistence retains each result.

Validation: 27 targeted native tests and 14 transcript-auditor tests pass;
TypeScript builds; Ruff checks and formatting checks pass for the Python
changes; `git diff --check` passes. Counts include existing relevant regression
tests and should not be added to earlier overlapping suite counts.

Raw red and green logs are retained locally under
`experiments/harness_audit_20261003/framework_fault_injection` in the tennis
workspace. Model rate limiting, physical control quality and arbitrary detached
daemon descendants are outside these fixtures' demonstrated scope.

## Explicit workspace isolation

Concurrent native Kimi evaluation exposed the distinction between launching
inside a git subtree and explicitly choosing it as the task workspace. Default
startup intentionally selects the enclosing git root; isolated evaluations must
pass `chat --workspace <fixture-root>` and validate the resolved root before
scoring. Samples that ran in the enclosing tennis repository remain invalid for
isolated capability scoring.

A separate reproduced defect affected explicit subdirectories: the frozen
task context kept the requested directory, but startup persistence widened it to
the enclosing repository. This caused `/workspace show` and later restored
bindings to disagree with the tools' actual root. Exact startup bindings now
skip git normalization; automatic git-root startup and ordinary legacy binding
retain their existing behavior. The new exact-directory persistence regression
was red before the fix; all 15 workspace/context/scratch-root tests pass after
the fix. This does not make SIM tool-layer-only mode an OS security sandbox.
