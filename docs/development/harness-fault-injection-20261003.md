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
## Explicit keyframe initialization

R03 live Kimi evaluation exposed a missing initializer: a keyframe could be
repaired through structured model patches, but snapshot/rollout could not
select that keyframe. Altering body placement was not an equivalent execution
of the repaired home state; the old evaluation remains unchanged.

`sim snapshot MODEL --keyframe NAME` and `sim rollout MODEL --keyframe NAME`
now select the exact compiled name through actual `mj_resetDataKeyframe`,
followed by forward computation. Snapshot returns a content-addressed
`simkey_…` initializer reference. `--keyframe-ref REF` reuses that exact
model-bound initializer; missing names, modified/foreign initializer records,
cross-model references and competing state/keyframe selectors fail closed.
Neither initialization nor proof creation patches body placement or steps
physics. The default qpos0 initialization is unchanged.

The initializer includes the compiled keyframe time, qpos, qvel, act, ctrl and
mocap values plus model ref/digest. Full-integration state metadata retains its
source; serial trace records include `initial_state_ref` and initialization
lineage, and receipts point to that actual initial snapshot. Restoring the
snapshot retains provenance and permits exact replay. Keyframe/snapshot clocks
can start above zero: trace times remain absolute, while rollout/receipt metric
duration is now the actual elapsed interval rather than the absolute end clock.
The Python runtime, MCP client/tools and CLI expose the same selection rules.

Seven initial regression cases were red before the implementation. Eight final
cases cover actual MuJoCo reset equality, immutable reference reuse, default
initialization, direct/resumed rollout lineage, exact replay, unknown/ambiguous
selection, foreign model references, real CLI success/failure and native MCP
client delegation. The relevant state/patch/CLI/MCP cohort has 78 passing tests;
Ruff, formatting, eight-module mypy and diff checks pass. The tests use generic
fixtures and do not implement a robot controller.

## Actual execution validation and rejected partial evidence

Serial rollout traces and experiment receipts now share `runtime_validation`
(`rosclaw.sim.runtime_validation.v1`). Its `serial_each_step` method records
actual warning counters, checked step count, initial/final clocks, expected and
actual elapsed time, and the exact finite dynamics fields checked. Both time
continuity and warning checks run at every step before metrics/sample collection.
Historical objects remain unchanged; absent counters cannot imply zero warnings.
Native batch does not claim this serial validation scope.

Rejected serial execution keeps content-addressed `failed_simulation_trace`
and `simulation_failure` objects. The error returns their `trace_ref` and
`failure_ref`, including through the real CLI. They bind model/digest, actual
controller/action digest, initial snapshot, seed, requested steps and timestep.
Diagnostics retain failed step, expected time, actual failure arrays and warning
counters, the sampled valid prefix, and the exact last valid integration state
projected to time/qpos/qvel/ctrl. Failure arrays exceeding 4096 entries have
explicit truncation metadata. Nonfinite numbers are JSON-safe strings with
separate finite-field flags; they never become valid zeros or JSON NaN tokens.
The record always declares FAILED and `simulation_valid=false`, without a
successful simulation receipt. Audit and strict replay reject these failure kinds.

The real unstable fixture fails at step 3 with BADQACC warning count 1,
last valid time .004 and velocity above six million; the finite reset state is
still rejected. Five new tests cover successful measured validation, both
failed serial entry paths, immutable retries, rejection by audit/replay,
injected nonfinite acceleration and native CLI error references. The combined
rollout/receipt/keyframe/contract/replay cohort passes; source mypy and Ruff pass.
Inspect local `framework_fault_injection/runtime_failure_evidence` for immutable
sample objects, the native error and the targeted regression transcript.

## Compaction observability without speculative cancellation

Installed PI 1.0 compaction uses its shared provider stream but only awaits
the summary result. Session subscriptions do not receive summary token deltas;
the existing turn watchdog is disarmed before automatic compaction. A long
summary therefore cannot be diagnosed as a stalled stream merely from missing
session entries. PI exposes distinct public start/success/failure hooks and
the compaction AbortSignal. Its separate `abortCompaction()` API exists, but
automatic cancellation requires actual stream progress evidence.

ROSClaw now observes those public extension hooks. After 30 seconds it shows
an elapsed waiting notice, then at most once per minute. The notice explicitly
states that progress cannot currently determine a stopped request. The observer
never cancels or changes the provider signal, summary, user goal or prepared
session branch. Bounded lifecycle records are written to local
`logs/compaction-lifecycle.log`; there is no `appendEntry` during prepared
compaction. Success, failure, cancellation, session shutdown or replacement
detach timers and AbortSignal listeners. Notification/log failures are isolated.

A prior red extension test reproduced missing start/failure observation.
Fifteen targeted native tests pass, including the installed PI's public
`generateSummaryWithUsage` with a delayed deterministic provider stream,
which completes without cancellation despite missing session token events.
Other cases cover independent observers, terminal cleanup, cancellation,
shutdown, failed UI/log callbacks, native hook wiring and existing compact
task/artifact anchors. This improves observability; it does not claim to
resolve or identify a genuinely stalled summarization request.

Each lifecycle record now has a unique compaction ID and captures the owning
session/process at start. Parallel sessions and subsequent session replacement
cannot relabel an earlier compaction's waiting or terminal events. The type-only
public ExtensionAPI import carries an explicit HP2-COMPAT boundary rationale;
it creates no PI session and accesses no private runtime. The complete native
cohort passes 277 tests with three skips, including the HP2 structure gate.

## Native batch sample alignment

The native batch trajectory contains post-step rows. Downsampling with
`full_traj[::stride]` previously recorded steps 1, 3, ..., 499 for a 500-step
run with stride 2. Serial sampling records steps 2, 4, ..., 500, and the
batch final snapshot already contained step 500. Batch trace samples now
use the same step indices as serial execution and always retain the true
terminal row. Each sampled control row uses that actual step index. Trace
duration subtracts the actual initial clock, including restored keyframes.

Six red cases used changing open-loop controls with 5, 6 and 500 steps and
initial clocks 0 and 2 seconds. All six now match serial timestamps,
positions, velocities and controls; the final sampled state equals the
actual final snapshot. The related batch/state/initializer/validity cohort
passes 43 checks. No serial per-step warning telemetry is attributed to
native batch execution.

Further restored-control testing found that batch `hold` silently replaced
caller control with zero, and a two-row `ctrl_series` with five requested steps
only executed two steps while previously reporting five. Batch now preserves
each branch's actual held control. Explicit steps truncate a longer series or
continue its last row for a shorter series, matching serial behavior. Initial
sample controls describe the restored state before the first series command;
final snapshots retain the actual last executed control. No command semantics
are inferred from a zero-filled temporary data object.

The two original control/request cases were red. Additional cases verify a
longer series and two independent branches holding different commands.
The resulting ten sampling/control cases and related cohort pass 47 checks,
with source mypy/Ruff and diff checks passing.

Direct batch callers now receive the same elapsed-duration preflight budget
as serial callers, and branch-count rejection precedes model compilation.
Two red fixtures proved over-budget requests reached native batch execution;
both now fail closed before that noninterruptible call. The relevant cohort
passes 23 checks. Native C++ batch execution still does not expose per-step
wall-clock cancellation; these preflight guards do not claim otherwise.


## Public summary stream observation after host restart

PI summary generation consumes `agent.streamFunction(...).result()` and emits
no session token updates. A public stream tee now records the actual provider
request start, first content, bounded progress notices, waiting phases, and
terminal usage/reason. Records bind the original session ID and a unique
request ID; they contain no prompt, token text, auth, or headers. Waiting notices
never classify JSONL silence as a stall and never automatically abort a summary
or the main agent. Ordinary turns return the original stream unchanged.

The tee retains provider event order and object identity, the original request
model/context/options/receiver, and the original cancellation signal. Public
`end(result)` is a legitimate completion even without a terminal event. A closed
stream without either terminal event or result, or a thrown/rejected provider,
returns an explicit protocol error rather than hanging on `.result()`. Provider
error/abort results and their measured usage remain unchanged; synthetic
protocol failures label usage `NOT_RECORDED`. Logging failure is observational
and cannot interrupt a request.

Fifteen focused stream fixtures and four existing lifecycle fixtures pass on
actual installed PI 1.0.1. Four use real public `createAgentSession` and
`session.compact()` with isolated in-memory history and fixture auth, checking
canonical history, request route/auth/header, legitimate result-only completion,
provider error, malformed closure, and public `abortCompaction()`. The SDK writes
the real compaction entry only on success; abort/failure never creates a fake
summary. No loop fork, private method, branch rewrite, or summarization replacement
is introduced. This worktree's historical manifest still pins 1.0.0; the isolated
fixtures explicitly run via its existing dependency symlink to the main checked
PI 1.0.1 installation. Integration must validate the current main manifest.

The host reboot erased any unpersisted streaming preview. A previewed toolcall
is not proof that a tool was executed. Recovery must compare the canonical
assistant/tool-result journal, operation ledger, and target artifact; an absent
assistant/tool result and absent target file are interrupted/unexecuted work,
not a successful write. This stream observer does not invent tool results or
reconstruct vanished previews.


## SimStore durable acknowledgement after reboot evidence loss

The reboot left content-addressed simulation JSON files empty although their
refs had already been returned. The preexisting writer used a temporary file
and `os.replace`, which ensured namespace atomicity but never synced file bytes
or directory links. That is insufficient for an acknowledged object to survive
a host power loss.

`SimStore.put` now flushes and fsyncs the temporary JSON/binary file, atomically
replaces its destination, then fsyncs the partition directory before returning
its ref. Parent directory links, including newly created task/store/partition
paths, are synced; retries also complete failed directory-creation barriers.
A partition-directory advisory flock serializes cooperating writers across
processes, including collision checks. It does not claim protection against an
uncooperating process deliberately modifying the store. Existing same-content
objects are never rewritten, but file and directory barriers are repeated
before an idempotent acknowledgement. Same-ref alternate JSON/binary payload
kinds fail closed rather than creating an ambiguous resolution.

Empty/truncated/different existing bytes remain untouched and cannot be
reconstructed by retry. `exists` requires a matching content digest; `resolve`
checks path containment and digest before returning an artifact path. Legitimate
empty binary payloads retain their actual digest semantics. A post-rename
directory-sync failure raises without returning a ref and leaves its complete
immutable object for a subsequent verified retry.

The initial durability cohort was red with nine failures and two passes.
After repair, 86 store/ref/model-patch/runtime-failure/state-restore checks pass.
Fixtures cover both payload types in all six partitions, file/replace/directory
failure injection, directory-creation retry, stale empty/truncated refs,
idempotent barriers, concurrent writers and forced digest-prefix collision,
and preservation of original bytes. Ruff, mypy and diff checks pass. These are
filesystem fault-injection checks, not a claim to have repeated a physical
power-cut test or to restore historical damaged evidence.


A follow-up review found that a failed first ancestor sync after creating a
multi-level task root was not retried fully: all paths were visible and the
next missing-directory scan forgot the earlier ancestor. Both same-instance
and fresh-instance injected fixtures were red. A temporary in-memory barrier
map fixed only the former and was superseded before integration.

SimStore now durably registers an immutable namespace directory intent in its
first already existing ancestor **before creating any missing directories**.
The exact namespace and anchor are canonical bytes in a namespace-hash-named
file; file and ancestor-directory fsync must succeed first. A failed intent
barrier therefore cannot leave a newly created unregistered directory chain.
New instances/processes find and byte-validate the exact registered intent
through read-only ancestor inspection, then repeat the whole parent-link sync
chain from that original anchor to the object partition. No directory above
the registered anchor is fsynced. Invalid or conflicting intents fail closed
and their bytes are preserved. Registration establishes the namespace owner's
existing ancestor as its trusted boundary; it does not infer historic losses
or repair preexisting corrupt receipts.

The 92-check store/ref/model-patch/runtime-failure/state-restore cohort passes,
including two separate subprocesses for a failed deep-directory sync followed
by fresh-process retry, intent file/directory failure before mkdir, and corrupt
intent preservation. Ruff, mypy and diff checks pass. These verify actual
filesystem operations and injected I/O failures; they do not replace a physical
power-cut test.


## Practice artifacts and persistent plans use the same durable boundary

Further filesystem-only review reproduced six failures: Practice returned a
registered checksum without reading a damaged artifact, rebuilt empty/broken
manifests, and saved its manifest using in-place truncation; PersistentPlanStore
created/consumed records with unsynced bytes and could overwrite an unreadable
UUID collision or silently skip a damaged consume.

The verified immutable namespace registration is now shared in
`rosclaw.storage.durable.DurableNamespace`. SimStore retains its exact existing
intent format and barriers. Practice and Plan owners establish their own
registered existing-ancestor boundary before creating their roots. Atomic
mutable writes use unique temporary files, flush/file fsync, replace, and
directory fsync; paths must remain in their owning namespace. Same-content
Practice retries validate actual bytes and repeat file/manifest barriers.
Registered checksum mismatches, unreadable manifests, and unregistered differing
payloads fail closed and preserve existing bytes. Valid intentional Practice
updates remain supported. Artifact and manifest are individually durable writes;
this does not claim a cross-file transactional commit.

Plans preserve corrupt bytes and reject damaged reads/consumption. If directory
fsync fails after a CONSUMED record replaces its predecessor, the method raises
without acknowledgement and preserves CONSUMED; a fresh owner never recreates
PLANNED. A later valid retry may synchronize the existing state. Plan UUIDs
remain random and existing envelope/raw-record compatibility is retained.

The new six RED cases pass, with five additional failure/restart/boundary
fixtures. Full Practice plus evidence, SimStore durability and typed-plan cohorts
pass 250 tests with nine skips and one existing dependency warning. Ruff, mypy
(four source files), and diff checks pass. No historical receipts were rewritten.
Legacy `sim/api.py` writes and other stores are outside this change's durability
claim and still require separate review.


A forced first-writer interleaving reproduced a registration race: an ancestor
intent scan was stale after another writer registered its ancestor and created
the namespace, so the delayed writer chose a second deeper anchor under a
different lock. Both puts returned but future instances rejected the two
anchors. The shared helper now re-scans under the selected directory flock and
releases/reselects the original registered anchor if necessary. No second
intent is published from a stale scan. The actual threaded fixture is RED to
GREEN; already conflicting intents remain fail closed without deletion.
The broadened Practice/store/ref/patch/failure/state/typed-plan cohort passes
294 tests with nine skips, plus source mypy/Ruff/diff checks.


## Cooperative transactions are separate from durable writes

Two real multiprocessing fixtures forced the first writer to pause immediately
before mutation. Both Plan processes read a PLANNED snapshot, then both
`consume` calls acknowledged; for Practice both artifact writes acknowledged
but the last manifest replaced the other writer's entry. Atomic replacement
and fsync alone do not enforce single use or protect read/modify/write updates.

A stable regular-file `.rosclaw-transaction.lock` now serializes cooperating
transactions. It is separate from namespace registration directory flocks,
never replaced, and rejects symlink destinations. Practice holds the same
session/episode manifest transaction through checksum validation, payload
write, and manifest update. Plan consume re-reads status and TTL under its
transaction lock before durably recording CONSUMED; put/capacity and clear
share the same lock, and clear synchronizes deletions. `get_for_execute` remains
a read-only snapshot, not an execution claim. The existing executor invokes
its action only after consume succeeds; exactly one cooperating consume may
acknowledge. Native raw-plan compatibility now also preserves CONSUMED.

After a post-rename sync failure, consumption raises and leaves its conservative
CONSUMED record: no execution is authorized by the failed acknowledgement and
a later reader cannot reset it to PLANNED. This provides at-most-once guarded
consumption, not exactly-once physical execution or a cross-file transaction
across unrelated stores. External writers ignoring the cooperative lock are
outside this transaction guarantee.

Both actual process RED fixtures pass; additional tests cover native raw-plan
single use, missing consume, durable clear, symlink locks, and valid Practice
updates. The full affected Practice/evidence/store/ref/patch/failure/state and
typed-plan cohort passes 301 checks with nine skips and one existing dependency
warning. Ruff, four-file source mypy, and diff checks pass.


## Stop feedback preserves actual terminal operation state

A live audit observed `process_stop` claiming cancellation after the operation
had already SUCCEEDED; its durable ledger correctly remained SUCCEEDED. Two
private fixtures ran real exit-0 and exit-17 processes to terminal states and
reproduced the false CANCELLED tool result. The bridge now returns the existing
terminal state with an already-completed/failed/lost/cancelled explanation and
performs no new cancellation or ledger event for a terminal operation. After
an active cancel request it reads the actual ledger again: CANCELING means a
request awaiting confirmation, and a completion that won the race retains its
actual terminal state. Unconfirmed active state is a retryable error, not a
successful cancellation claim.

The two real process RED cases and five state/race fixtures pass. The affected
operation/cancel/tool-bridge cohort passes 46 checks with two preexisting
subprocess event-loop cleanup warnings; Ruff, source mypy and diff checks pass.
The original live operation and its ledger were never modified by this repair.


## Legacy model-ref staging preserves XML bytes and original provenance

Three filesystem/load RED cases reproduced legacy `sim/api.py` failures:
different input directories sharing `robot.xml` changed the staged body behind
an earlier `model_<digest>` ref, a zero-byte registry was silently rewritten,
and imported XML/registry files were acknowledged without fsync barriers.
New imports now stage XML under a digest-named file, validate existing registry
and staged XML digest, and preserve the first exact registry bytes/provenance
when an identical XML comes from another source. Source bytes changing during
inspection reject the import before a ref is acknowledged. Existing damaged
registry/XML/state bytes fail closed and remain unchanged.

Legacy model, observation and operation JSON writes use the same registered
namespace and cooperative transaction helper with file/replace/directory
barriers. Operation IDs historically bind states rather than wall time: only
a differing `wall_ms` may reuse the original frozen metadata, and every other
field must match. Original wall time is never replaced or claimed as a fresh
measurement. Legacy XML hashes still do not bind the full external asset
bundle; these refs are not promoted to canonical strict experiment receipts.
Legacy rendering/export durability is also outside this change's scope.

Eleven load/filesystem fixtures pass, including corrupt input preservation,
original-body retention, repeated-source provenance, and post-rename sync
failure without acknowledgement. The broadened relevant evidence/store and
W02/W03/step/eval cohort passes 95 checks with six skips and one deselection.
The deselected preexisting W09 hinge-ball case commands motor controls 0.2 and
triggers actual QACC warning index 5 at step 14. An independent unchanged
HEAD copy of the old API reproduces the identical warning/step, so the new
storage change does not cause it; the guard remains strict. Ruff, source mypy
and diff checks pass. No original tennis source, historical receipts, or live
operation was modified by these fixtures.

### Typed model external-file identity boundary (fresh-process proof)

Actual private filesystem reproduction: `load_model_xml` accepted a valid MJCF
whose mesh file was an absolute host path, with `assets={}`. Replacing the STL
from scale 1 to scale 2 changed body inertia in a fresh Python process from
`[.10, .0625, .0625]` to `[.40, .25, .25]`, while the model digest was identical
and the old state restored successfully. Same-process compilation hid this
fault because MuJoCo caches file assets. This proof involved compilation and
state initialization only; no physical controller or live task was changed.

The existing manifest schema remains unchanged. Before parsing/compiling,
reading a model manifest, or calculating identity, every MJCF file dependency
must now match captured asset bytes. Absolute/traversing paths, unsupported
file-bearing elements, missing assets, and includes fail closed with
`MODEL_ASSET_UNBOUND`. Includes require a producer to flatten their transitive
closure before import; merely supplying an include XML blob is not a closure.
Unsafe old manifests are rejected without rewriting or reconstructing them.
The file importer also captures hfield and skin files alongside meshes/textures.

Six new tests first failed against the old implementation (four preflight
error-category cases, include acceptance, and a fresh-process old-manifest
acceptance case). Positive tests use the worktree source explicitly in child
processes, bypassing the shared editable installation. Captured mesh identity
and inertia remain identical after source mutation; a newly imported mutated
mesh has a different digest and actual fresh-process inertia, and rejects the
old state. Meshdir/assetdir, texturedir and hfield imports compile in fresh
processes after source files are deleted. A task-local include is explicitly
rejected. This does not claim the old structural snapshot signature contained
mesh geometry, nor does it prove arbitrary plugin resource closure.

### Public sampled-recording metadata (observational output)

`SimulationTrace`, `SimulationReceipt`, and native batch branch results now
expose `recording`: `total_steps`, `sample_stride`, `saved_samples`,
`full_step_recording`, `max_record_gap_steps`, `max_record_gap_s`, and
`includes_initial_state`. Counts include the initial sample. Gaps and stride
are derived from actual saved timestamps relative to the real initial time
and integrator timestep; a final shorter interval is allowed. Irregular grids
have no single stride. Duplicated, nonfinite or off-grid timestamps yield
unknown density rather than an invented full-step claim. An old contract
without metadata defaults to `{}`, which also means unknown.

This describes retained trace evidence. It does not change integration,
per-step validation/metric collection, budgets, sampling, or replay. Existing
raw trace `recording.max_record_points` configuration remains byte-for-byte
the same; no derived metadata is written into the immutable trace or receipt
payload, and their content-addressed refs remain unchanged. Public contract
digests include the new output field as usual.

Four fixtures first failed with missing public metadata. Full/stride-3/stride-5
cases exercise serial, experiment receipt and native batch results at a
nonzero initial time, including the shorter final gap and immutable raw-ref
checks. Six missing/irregular/invalid timestamp fixtures prevent overclaims.
Recording + sampling alignment + rollout + three replay suites: **46 PASS**.
Ruff, mypy (three changed source files), and whitespace checks pass.

### PI provenance reconciled to the already installed 1.0.1 runtime

All four `@earendil-works/pi-*` dependencies, overrides, package-lock entries,
and actual installed package manifests are 1.0.1. Read-only `npm view` of each
exact 1.0.1 package reports gitHead
`a7229ddc21810d6245105978033b7df645ecc2f7`. The authoritative upstream metadata
now records that version/source and actual package-lock SHA256
`d52da2ca3aa9efd9fcfaf457d612194ce2b35ec82492b490e766be0e907271d7`.
This corrects stale provenance; it performs no installation or live upgrade.
The previous metadata recorded 1.0.0 / `a13d35a742c6ef8462812a28fbe1d8c8b7431c32`
and lock SHA `d86c930f7689a090dfb0af954b0e074c74ee9009b266151ac5b397d4469d5785`.
The characterization suite remains version 1; earlier real public PI compaction
fixtures were already executed against installed 1.0.1. Existing W08 pin and
lock-integrity assertions initially failed twice and now pass: **4 PASS**.
This is not a new release-artifact qualification claim.

### Consumed operation reminders no longer queue redundant model turns

Read-only analysis of the formal session's entries 3188/3196/3202/3208/3212/3218
found six distinct `rosclaw.operation.result` messages, not repeated delivery
of a single operation after reboot. These operations were started in entries
3062–3139. Several terminal results had already been consumed in that same
run: FAILED status/output at 3096/3098, terminal status/output at 3129/3133 and
3141/3145, and SUCCEEDED status at 3176. Nevertheless each completion was
already queued using PI `deliverAs: followUp` while the agent remained busy.
The queue drained only after the long agent turn ended, producing six separate
reminder-driven turns and repeated NOT_DONE narratives. No historical entry,
queued live message, operation or task state was changed by this audit.

OperationWatcher now retains busy terminal reminders in its own pending map.
Successful finalized `process_status`, `process_output` or terminal
`process_stop` toolResult messages acknowledge only the matching operation,
owning task and revision, removing the now-redundant reminder. ACK occurs on
public `message_end`, after PI agent-core has retained the finalized toolResult
in agent state, rather than early on `tool_execution_end`. Agentd supplies
structured operation identity/status; the bridge preserves all result metadata
and distinguishes inner result failure from a successful transport envelope.
Process-start registration prefers the structured operation ID and keeps the
old textual fallback for compatibility.

Unconsumed reminders publish only at idle, with task/revision checked again.
Running output, failed reads, unrelated messages and mismatched identity do
not ACK. Interrupted publication or temporary ownership lookup failure retains
the reminder for retry. Stop/start of the watcher leaves pending reminders
intact. The watcher does not alter user queues or create fake tool results;
its pending/delivery state remains in memory, with no new cross-process reboot
or persistent delivery guarantee claimed.

Eight initial private watcher fixtures failed before repair. The final eighteen
fixtures include public PI 1.0.1 Agent runs aborted during both successful and
failed terminal reads, with bounded local streams that honor abort signals,
plus early-consumption and await-race cases. Real OperationManager fixtures
verify backend terminal status/output identity and no ledger/event mutation.
Terminal/toolbridge/cancel propagation Python cohort: **34 PASS** under
unraisable-warning-as-error. Ruff and mypy for both changed Python sources pass.
Full package build + Node suite against installed PI 1.0.1: **312 PASS,
3 SKIP (315 total)**. No live process/session restart was required.
