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

The interactive `/workspace use` command also saves the exact directory now.
It does not perform a live context migration: tools and the current task remain
bound to their frozen task root. Its message explicitly says the selection is
for the next chat startup, subject to explicit path/current git-project priority,
and states the current directory. `/workspace show` reports the actual task
directory together with any different saved selection. The current header is
not changed to imply a live switch. A new command regression checks the on-disk
selection, frozen task root, header and both messages; 19 related tests pass.
Atomic live workspace/task/session migration remains unimplemented.

## Model patch mass provenance

An actual Kimi repair attempt exposed MuJoCo 3.13 `MjSpec.to_xml()` eliding
explicit mass/density when its physical value equals the compiler default.
For a box of half-size 0.06 m, explicit mass 1.728 kg or density 1000 kg/m³
was omitted after even an unrelated RGBA patch; `A02_explicit_mass` then
rejected the descendant. Another fixture showed density patches were ineffective
when a previous explicit mass remained, because MuJoCo gives mass precedence.

Structured patch serialization now preserves finite authored mass and source-
or patch-declared density, while leaving originally implicit default-density
geometry implicit. Geometry identity/order must match the compiled spec; no
guessing or globally adding density is used to silence A02. Setting density
clears the old mass with MuJoCo's unset sentinel. Patch lineage and parent bytes
remain unchanged. Six new fixtures independently recompile saved child XML and
check body mass, declarations, lineage, unrelated-patch behavior and mass/density
precedence. Five were red before the fix; all six now pass. The relevant patch,
geometry and determinism cohort has 42 passing tests; Ruff and two-module mypy
checks pass. No robot controller or physical rollout was authored here.

## PI 1.0.1 isolated compatibility check

The official [v1.0.1 release](https://github.com/earendil-works/pi/releases/tag/v1.0.1), published 2026-10-03, and npm latest were independently checked. Native agent dependencies and overrides, and the standalone TUI dependency, are pinned to 1.0.1. The upstream release removes its npm shrinkwrap, so ROSClaw keeps the regenerated application lockfile as the reproducibility boundary.

Both actual upstream patch anchors apply without changing AgentLoop or session format. Full native build/tests passed: 272 passed, 3 skipped; standalone TUI: 27 passed. The initial run rejected the old hard-coded 1.0.0 pin assertion; the expected exact pin was updated and the complete suite rerun. Private raw logs are `pi101_native_all_tests.log`, `pi101_native_all_tests_green.log`, and `pi101_standalone_tui.log` in the supervision evidence directory. These are compatibility tests, not model or physical-success claims. Existing model trials remain frozen at their recorded 1.0.0 version. The formal ongoing session must reach a normal idle boundary before its runtime is changed.
## Actual execution warnings and reset validity

Frozen real Kimi R04 evidence exposed a false valid rollout: the requested
500 steps at 0.002 s ended at 0.996 s, included two consecutive sampled
timestamps at 0.004 s, reached recorded velocity 6,577,310.683550647 and
silently changed the target control from 1 to 0. Nevertheless its receipt
claimed both `simulation_valid` and `physical_audit_pass`. Those historical
records remain unchanged and invalid; their warning counters were not recorded.

Per-data guards now reject MuJoCo warnings, actual per-step time discontinuity
and non-finite state, acceleration, control, force and sensor arrays. Warnings
are inspected on each `MjData`; no global warning callback is installed, so
independent parallel sessions do not share a warning hook. The shared guard
applies to serial rollouts, interaction stepping, audit sweeps, servo holding,
solver/timestep probes and the public Python simulation API. Native batch
output is checked in full before downsampling, including warnings and time
progression. Unstable/reset execution raises `SIM_DIVERGED` and cannot create
a valid trace or success receipt. Invalid sensitivity probes remain explicit
diagnostic warnings, with `simulation_invalid`, rather than a fabricated
deviation computed from reset states.

Trace audits also inspect executed motion: A15 rejects non-increasing sampled
time, and A16 applies the configured velocity limit to supplied trace samples
as well as the fresh-state hold probe. A safe zero-command hold therefore
cannot hide an explosive executed tracking trace. Numerical validity and model
audit remain distinct from task completion or robot/hardware safety.

The original reset/finite-dynamics cohort had seven failing fixtures and one
stable control; native batch and historical trace audit supplied two additional
red cases. All 13 final targeted cases pass, covering actual solver reset,
serial receipt rejection, native batch, interaction, audit sweep, public API,
injected warning/time/acceleration/control/force faults, and stable nonzero-time
initialization. An 80-test relevant cohort passed, followed by 460 broader
simulation/API checks passing with one skip and one deselection. The final
source, including the public-API regression and all finite force/sensor fields,
then passed 461 checks with one skip and one deselection (163.42 s). Final
shared-guard regression results are retained in the local audit directory. Ruff and
eight-source-module mypy checks pass. A read-only independent frozen baseline
review is saved as `framework_fault_injection/r04_frozen_baseline_validity_independent.json`.

## PI compaction event compatibility

The installed 1.0.0 and 1.0.1 SDKs emit `compaction_start`/`compaction_end`; the ROSClaw adapter only recognized the older `auto_compaction_*` names. It also mapped aborted/error endings to completed. The adapter now accepts both names and emits separate failed/cancelled endings; an ending without a result cannot claim a saved summary. Two failing lifecycle cases were reproduced before repair and both pass afterward, alongside the two existing adapter contracts. Full native tests against actual isolated PI 1.0.1: 274 passed, 3 skipped. These event-contract checks do not claim that an ongoing compaction is deadlocked or authorize interrupting it.

The formal eighth compaction completed normally. Its 256,553-token prefix preserved the corrected M22 objective and the public eight-case limit. Some prefix status was older (static analysis independent acceptance pending and task revision 101); the retained suffix starts at line 2755 and contains the independent static receipt at line 2769, while the authoritative task is revision 102. The full private summary and the independent boundary review are retained as `compaction_eighth_summary.txt` and `compaction_verification_eighth.json`. No session entries were rewritten.
