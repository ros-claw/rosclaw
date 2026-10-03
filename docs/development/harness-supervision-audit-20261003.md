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
- Recognize PI's built-in OpenAI and OpenAI Codex providers in Python's chat
  configuration gate without requiring custom `models.json` entries. An actual
  access-only OAuth comparison originally stalled at the Kimi onboarding prompt
  before any model ran; after repair, native OpenAI U01 independently verified.
  Align the Kimi diagnostic view with PI 1.0's actual `anthropic-messages` API
  and endpoint. Authentication/reachability remain separate probe decisions.
- Run the `bash` tool with Bash and `pipefail`, including inside bubblewrap.
  Brace expansion and failures within a direct pipeline now follow the tool's
  advertised shell contract. Enabled shell options are exported so an ordinary
  nested `bash -c` also reports a failed `timeout | tail` pipeline; a shell can
  still explicitly disable options. The M21 recording attempt reproduced this
  masking bug: its 900s timeout was originally displayed as exit 0.
- Reject an overlong `ROSCLAW_HOME` Unix socket path before kernel startup, with
  an actionable error. This does not add support for arbitrarily long homes.
- Add bounded optional provider wait overrides, in milliseconds:
  `ROSCLAW_PROVIDER_FIRST_TOKEN_TIMEOUT_MS` and
  `ROSCLAW_PROVIDER_STREAM_IDLE_TIMEOUT_MS`. Accepted range is 1,000–3,600,000;
  invalid values preserve the existing 30s/45s defaults. Progress notices remain
  active, and tool execution does not count as provider inactivity.
- Show supported patch and controller payload examples in simulation CLI help,
  including unsupported online policy callbacks.
- Show observation channels, JSON examples and the existing quaternion/raw
  velocity representations in `sim observe --help`. H01's extended run ended
  during unbounded recursive searches for channel names across the parent
  workspace. The Bash tool now recommends scoped ripgrep and explicit timeouts;
  this guidance does not guarantee compliance or change the no-default-kill policy.
- Add a native `kimi-coding` HarnessBench profile using `anthropic-messages`.
  Both comparison legs use the same profile settings and an environment key
  reference; credentials are not written to generated model configuration.
- Fix the PTY progress test to wait for an actual numbered progress line rather
  than matching the command echo containing `progress-step-$i`.
- Recognize the actual PI cancellation wording, `This operation was aborted`,
  as a recoverable cancellation rather than an unknown provider failure.
  Cancellation clears provider-pause UI state without suggesting a model switch;
  both PI abort stop reasons and SDK cancellation error messages invoke the
  existing user-interruption cascade (watchdog aborts retain their exception).
- Background-operation guidance permits independent remaining steps that do not
  modify the operation's inputs. Dependent work still waits for the automatic
  result notification; repeated polling and sleeping remain discouraged.
- Rebuild the task/registered-artifact anchor after each distinct PI compaction
  entry. Deduplicating solely by task/revision incorrectly skipped later
  compactions, even though their summaries could omit an earlier anchor and
  artifacts could have been registered without advancing the revision.
  Duplicate delivery of the same entry still emits only one anchor.
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

Remaining 600s native Kimi trials independently verified V01 in 496.6s;
I01 and S01 did not settle (602.7s and 602.6s, with 26 and 19 tool calls).
These are separate reruns after guidance/configuration repairs, not replacements
for the original short-budget failures or a controlled model ranking.

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
unresolved as an insertion/embedding performance result. A separate batch
insertion diagnostic also spent its time in the default ONNX embedding function
(`onnxruntime.InferenceSession.run`), so it does not establish a database
pagination deadlock. The pagination test now seeds the real collection with
explicit, dimension-matched fixture vectors, avoiding unrelated inference. Its
2,300-row count/delete checks passed in 15.71s, including a filtered count and
deletion of 1,150 rows across the 1,000-row page boundary. This verifies native
pagination under these conditions; it does not resolve embedding throughput.
Broad non-live regression excludes native SeekDB modules only after retaining
this separate failure/performance evidence. Local reports contain exact run
counts and artifacts; repository-wide pre-existing lint debt is not green.

Validation completed: broad non-live regression 1,848 passed / 7 skipped /
24 deselected; daemon/kernel/MCP boundary checks 370 passed / 1 skipped; main
TypeScript package 250 passed / 3 skipped; independent TUI package 27 passed;
oracle repair/adversarial regression 59 passed. Some suites overlap; these are
separate results, not an additive count of distinct tests. Configured mypy scope
passed 121 files. Modified Python lint and `git diff --check` passed.

## Native-provider comparison and physical review follow-up

The Python onboarding gate previously recognized only Kimi, rejecting PI's
built-in OpenAI providers before any model call. It now recognizes the actual
`openai`/`openai-responses` and `openai-codex`/`openai-codex-responses` provider
contracts. PI remains responsible for credentials and model availability.
Kimi's diagnostic view matches its installed native Anthropic-compatible
provider. Focused gate/engine tests passed 19 cases; separate credential and
onboarding checks passed 22 cases (overlapping suites, not additive).

With this repair, three native `openai-codex/gpt-6.1-sol` trials at a 600s
budget independently verified U01 (57.1s/4 tools), E01 (117.1s/14 tools), and
H01 (107.2s/12 tools). E01 reduced replay RMSE from 0.2642573 to 0.0653108.
A native Kimi H01 follow-up using the updated tool guidance also verified in
236.3s/22 tools. Earlier timeouts remain recorded. This is a small comparison
under differing execution times and framework revisions, not a controlled
model ranking. I01 is interaction/grasp/lift; S01 is system identification.
Neither settled in its executed Kimi 600s follow-up. The expanded 48-task
matrix remains planned, not fully executed. Temporary comparison credentials
were removed; the formal chat keeps an access-only OAuth entry without a
Codex refresh token and continues implementing the physical scene on Kimi.

A second native compaction reduced the formal session from 267,270 input
tokens. Its summary was semantically checked against the latest physical
results, and a new compaction-entry-specific M21 task anchor was observed.
Original history remains intact. This does not guarantee every field in a
summary is current; the durable task and raw experiment evidence remain
necessary sources of truth.

Independent physical gates continue to reject unsupported successes. Stage A
has three M20 locomotion passes, while G1 fails the strict stopping peak-speed
gate. B v10 has 12 contacts/20 attempts and zero valid returns. B2 v19's claimed
one return crosses the net plane at ball-center height 0.10747m, below its
required 0.9475m: it is not a valid tennis return. Net collision was enabled;
this observation alone does not prove why the low crossing occurred.

C is an explicitly reduced 0.35m-net, close-range experiment, not standard
tennis. The v3 full-state NDJSON segments passed hash, dimension, finite-value
and time-continuity checks, but revealed invalid initialization: the M20 starts
at x=5.179m in trial 0 and x=3.668m in trial 9 instead of the declared +1.7m.
Its 17 paddle-contact episodes include repeated contacts with the same racket;
the longest alternating-owner chain is one. These failures are frozen and
have been sent back to ROSClaw for implementation repairs. No verified G1–M20
rally or strategic multi-agent competition has been delivered at this point.

Further OpenAI trials verified I01 in 167.1s/23 tools. The original S01 trial
finished in 117.1s/18 tools and was scored FAIL for recovering 0.01 instead of
0.3. Inspection exposed an invalid benchmark fixture: S01 supplied only the
0.01 calibration model and no independent measured observations. Fitting data
recorded from that same model correctly recovered 0.01 with NO_IMPROVEMENT.
The original S01 results, including earlier Kimi timeouts, remain raw evidence
but cannot be scored as identification-quality evidence.

S01/S02/S03 staging now provides independent synthetic observations generated
outside the agent workspace. The producer model/parameter values are withheld;
both legs receive identical raw trajectories, and B receives native immutable
dataset/trace/initial-state objects. S02 has actual zero-motion observations.
The S01 oracle resolves the explicitly reported receipt and checks that it used
the independently expected observation dataset, rejecting self-generated
calibration data. The A-leg oracle for these tasks has not been qualified; raw
observation parity does not establish a complete cross-leg comparison.

Fixture/SysID/runner regression passed 49 tests; additional oracle/adversarial/
model-profile checks passed 37 (overlapping scopes). A new OpenAI S01 trial
using the repaired fixture verified in 85.1s/13 tools, recovering
0.3000000000000001. This is a new run with corrected inputs, not a rescore of
the earlier run. Modified Python lint and whitespace checks passed.

The corrected native Kimi S01 follow-up also verified in 330.3s/15 tools,
recovering 0.2999999999999996. These single trials establish that both provider
paths can use the repaired observation contract; they are not latency rankings.

The formal UI retained `调用 bash/read` after the corresponding tool had already
ended, misleadingly showing a command still running during the next model wait.
Tool activity now tracks actual call IDs, clears completed/error calls, retains
overlapping active calls, and resets on agent end. A real extension-event test
first reproduced the stale label, then passed with the repair; the full main
TypeScript suite passed 250 tests with 3 skips. A running Node process loads this
change only after a safe restart; rebuilding files does not update its modules.

A third native compaction completed from 256,982 input tokens, with a distinct
task anchor at revision 44. Its main summary preserves the recent failures,
but the split-turn appendix contains older progress. Durable evidence and the
new task anchor must take precedence; compaction is not a semantic guarantee.

During supervisor maintenance, Esc canceled the two active video operations.
The kernel correctly recorded CANCELLED with `user_interrupt`; this was a
supervisor interruption mistake, not a silent worker loss. Their partial output
is retained and must be rerendered under unique output names. Esc is intentionally
a task interruption that includes its running background operations. Bash
heartbeat wording and embodied guidance now state that scope explicitly. Three
process-tree/heartbeat checks passed; cancellation semantics were not weakened.

The restarted formal chat now visibly clears completed tool labels. The latest
full session snapshot contains 1,298 structurally valid entries; this remains a
transcript check, not physical validation. ROSClaw's unique v3 diagnostic replay
for C v10 trial 3 independently decodes all 513 frames at 60fps, with 8.55s
playback of 6.125s source time. Its frame map records actual sampled indices and
times within 5ms of playback targets. It honestly shows a failed reduced-net
attempt; long event labels and ground aliasing still need presentation work.

Further adversarial evaluation found two scoring holes: S02 accepted a numeric
parameter concealed behind `identifiable=false`, and S03 accepted an unrelated
shadow observation as evidence. Both reproduced before repair. S02 now requires
no parameter claim and a reason; S03 requires a divergent answer and a report
bound to the independently supplied observation traces. It searches for matching
evidence rather than letting an unrelated report veto a later matching report.
The observation-fixture and v2 oracle suites passed 36 tests. This scope overlaps
earlier checks and must not be added to their totals.

The first live OpenAI S03 trial exposed a further evidence integration defect:
native `shadow_compare` returned a report without persisting it, while the oracle
required an experiments-store receipt. The agent exported a correct DIVERGED
report and damping-only MATCH controls. A separate immutable-workspace review
recomputed its residual, 2.317650954935967, exactly against the independently
supplied trace. The raw FALSE_SUCCESS is preserved alongside this independent
verification, not silently replaced. Native shadow reports now persist as
content-addressed experiment objects and return `report_ref`, including the
NOT_COMPARABLE path. A test using the actual native API without manually
inserting the report reproduced the missing receipt before repair. Shadow,
clock, observation and oracle regression passed 53 tests (overlapping scopes).

Fresh native S03 trials after the receipt repair verified for OpenAI in
87.1s/11 tools and Kimi in 107.0s/15 tools. S02 also verified for both providers
(107.2s/14 tools and 167.2s/27 tools respectively). The pre-repair Kimi S03
diagnosis was independently reproduced against all three supplied traces and
its damping-only repair; its original automated FALSE_SUCCESS remains intact.
These are single trials at different revisions and times, not model rankings.
The default mypy invocation passed 1,318 source files after the shadow repair.

Both C v3 diagnostic videos completed and their six video/timing/frame-map
artifacts were registered under the active M21 task. Trial 4 independently
decoded all 801 frames at 60fps/13.35s; its checksums and <=5ms source sampling
map matched. Neither clip establishes a successful rally. Artifact registration
also exposed a UX defect: the public `role` parameter was silently dropped by
the dispatcher. Registration now retains the role in metadata, alongside any
post-terminal flags. The delivery lifecycle/coordinator suites passed 12 tests;
the failing-before-repair test demonstrates the lost diagnostic role.

Native ROSClaw engineering continuation switched to OpenAI within the same
session and harness after a saved Kimi checkpoint. It repaired the collision
exit-velocity measurement and produced four selected-seed samples with real
velocity reversals, but zero net-clear/opponent-box returns. This is a supervised
continuation with access to earlier evidence, not a blind provider comparison.

A stage report explicitly labelled `diagnostic_progress_report_NOT_DONE` then
incorrectly closed the unfinished M21 task, blocking subsequent process admission.
The still-running old dispatcher lost the role, and Coordinator treated any
artifact as a completion signal. Explicit `progress`/`progress_*` and
`diagnostic`/`diagnostic_*` roles now bypass terminal verification and cannot
satisfy final deliverables. They remain in the evidence ledger; a later final
artifact can complete the same revision. Unlabelled legacy final-delivery behavior
is preserved: this is not semantic validation of arbitrary Markdown goals.
The observed mistaken r61 terminal transition is retained as historical evidence,
not rewritten into a successful physical result.

Coordinator also wrote outcome details into a nonexistent PiToolResultV1 field,
then swallowed the exception after mutating task state. Final outcome dimensions
now appear in the valid summary field so the model can see the transition.
Regression covers role persistence across the actual wire path, subsequent
admission, no premature verifier/outcome rows, frozen-spec progress, final
completion and rejection of diagnostic media as a required deliverable.

The repaired live session registered a new Chinese `progress_report` artifact
without completing M21; the subsequent eight-attempt contrast completed
normally. Independent recorded-force/state/exit checks passed. Two G1 returns
cleared the reduced net without contact and first landed on the opponent side;
M20 had no net-clear return, and neither side reached the opponent's racket box.
This is still NOT_DONE for the requested alternating rally.

A separate reopening defect left current `accepted_at` and `terminal_reason`
showing stale success after an unaccepted SUCCEEDED task was revised to RUNNING.
Reopening now clears those current-status fields while retaining SUPERSEDED
verification rows and the original revision outcome. A failing-before-repair
regression demonstrates the stale timestamp; related lifecycle suites passed
35 tests after repair (overlapping earlier scopes).

Before expanding understanding trials, four adversarial answers exposed weak
oracles: correct PID roles at swapped indices, correct sensor names with wrong
types, names without required types, and replacing the original model with an
empty one all passed. The oracle now checks compiled control addresses/counts
and PID input-signature bits independently of the product inspection helper,
requires both sensor names and types, and rejects changed understanding inputs.
MuJoCo 3.13.0 installed headers were cross-checked against official mjmodel.h /
mjtype.h. All 33 v2 synthetic oracle tests passed after these four red cases;
existing physical receipt/shadow cases in this scope remain passing.
The separate reopening/admission/terminal-authority suites passed 24 tests.

The expanded live Kimi U02 trial independently measured correct control indices
and physical setpoint semantics, but wrote "position setpoint" / "velocity
setpoint" rather than undocumented oracle tokens pos/vel. Its original FAIL is
retained; independent read-only reverification passes with explicit, finite
synonym normalization. The prompt now documents canonical role values; swapped
indices, missing types and modified models still fail. The updated synthetic
oracle suite passed 34 tests. This interface repair does not count as a new
successful model trial. The stream-idle notice also now says "waiting, not yet
cancelled", avoiding the misleading claim of an interrupted stream before any
cancellation occurred.

Compiled PID truth uses control addresses/counts and signature bits described
in the [official MuJoCo model header](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmodel.h)
and [input enum header](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjtype.h),
checked against the installed 3.13.0 headers.

PID inspection also exposed real product metadata defects on MuJoCo 3.13.0:
proportional gain was reported as zero for a PID actuator, and the actuator after
a multi-input PID read its range using the actuator ordinal instead of its
compiled control address. Inspection now reports the correct PID gain, control
address/count/per-channel ranges, and distinguishes actuator count from control
channel count in the Chinese summary. A mixed PID/position test failed before
repair; inspection/control/model-patch/scenario regression passed 38 tests.

The new declared shallow/rising INIT protocol produced bilateral net-clear shots
and actual receiver contacts; a center-plane projection can miss a real collision
because the ball reverses before reaching that plane. Subsequent native ROSClaw
two-sided station smoke produced two clean alternating contacts in each case.
Independent review checked all-ms ball continuity and momentum, actual-owner
contact/exit episodes, directional net/bounce links, hashes and full-state frames.
Neither case establishes three contacts. One missed third swing has a recorded
rearm-state defect; it is not evidence of a physical upper bound. Scene code and
experiments remain authored by ROSClaw, not the supervisor.

The fourth native automatic compaction completed from 256,111 tokens. Its summary
covers the older prefix through the progress-v5 period; the newer contrast,
shallow INIT and v17 implementation records remain in the retained suffix.
This boundary was independently checked against firstKeptEntryId, rather than
mistaking an older-prefix summary for loss of the newer results.

The station rearm repair was implemented by native ROSClaw: the old instantaneous
outgoing-velocity check could miss departure while waiting for mechanical recovery.
The repaired controller latches a real contact episode's departure, then checks
actual hinge angle/velocity, separation and recovery time. Frozen independent
review confirms three strict valid alternating returns for the G1-start seed11
case, with a fourth contact whose return fails. The M20-start case remains at one
strict return. All-ms ball continuity, contact-frame/sign transforms, momentum,
state/mesh/source hashes and flight health pass. This is a selected, reduced-net,
perfect-state, experimental-racket simulation result, not robust bilateral tennis.
The corresponding independent report is
`phaseC_v17_station_rearm_latch_cbcde1cc7ef6_independent.json` in the local audit directory.

A live Kimi R03 trial was incorrectly recorded as an infra timeout: final assistant
stop persisted 596.932s after session creation, within its 600s turn budget, but
`_wait_settled` required another 20s of PTY quiet. The frozen answer and native
patch/rollout/reset receipts independently verify successfully; original ERROR
records remain unchanged with a separate reverification. HarnessBench now uses
persisted final assistant stop (envelope persistence time, not generation-start
time) for both A/B legs and waits for any native background operations. Old history,
tool-use/results, partial follow-up writes, active operations and operations finished
after the original deadline cannot close a trial. Retries reset the turn's lower
bound without extending its deadline. Quiet fallback remains for legacy callers
without session directories; an answer file or a completion claim cannot close
new trials. Synthetic regression includes the near-deadline counterexample.
A real native OpenAI U02 trial passed in 23.0s with four tool calls and no infra
retry; this is a new smoke trial, not a model-ranking comparison to older timings.

Recorded replay of that selected case now passes independent full decode,
source-SHA/frame-map/time checks: 413 G1-start frames and 244 M20-start frames,
60fps, actual 1x and 0.25x segments. Robots/ball and bounded event sidebars are
clearer. The net is still edge-on; these are diagnostic clips, not final acceptance.
A second independent pass ties stored full-state ball/root coordinates and
velocities to the ms logs, derives the initial gate directly from qpos/qvel, and
checks compiled ball mass/gravity against the force integration. Both cases pass.

The predeclared fresh replication (seeds 7/23/42/123, both starting directions)
completed all eight attempts, including four INIT failures. The original net-only
contract reports three returns in G1 seed123, but its last shot exits the reduced
court without reception or a bounce. An additive completion contract therefore
requires actual opponent reception or the first opponent in-court bounce, while
preserving the old fields and raw verdicts. Under that declared reduced-court
contract the selected G1 seed11 result remains three; none of the eight fresh
attempts reaches three. This is a real generalization/measurement limitation,
not a reason to omit the INIT failures from the denominator or relabel a net
crossing as a completed rally. M21 remains NOT_DONE overall.

A local comparison preflight found that the dedicated clean A Python still had
MuJoCo 3.1.6 while B used 3.13.0. New A environments now pin mujoco/numpy/imageio
to the running B versions; existing divergent or contaminated A environments fail
before prompting the model. The isolated local A environment was aligned and
verified to have no ROSClaw import. Two pre-repair red cases and 18 runner/profile
regressions validate the guard and creation pins. A real native PI/OpenAI U02
smoke then verified in 36.5s/six calls, with 26 inline Python lines; B's earlier
same-task smoke used 23.0s/four calls. These are individual, differently timed
smokes with equal library versions, not a statistical harness/model ranking.

The live slow provider also repeatedly crossed the idle-notice threshold between
real content chunks, flooding the chat history. Duplicate stream-idle notices are
now limited to one per minute per turn. Actual first-token and stream-idle abort
deadlines, tool/user pauses, and live-content renewal are unchanged. The new
counterexample failed with three repeated notices; all 12 watchdog tests pass
after repair, including unchanged cancellation. Formal work is not interrupted
merely to reload this presentation repair.

All 132 persisted assistant natural-language claim entries through session line
1847 (23,577 characters) were reread alongside the full structural chronology.
Historical all-PASS/self-score/policy-only and passive-upper-bound language is
not adopted as independent acceptance; the corrected source/raw evidence and
explicit selected-protocol limitations control the findings.

A fresh, physically executed controller comparison now has distinct measured
behavior: geometry intercept A versus fixed-station B. At selected development
seed11, A completes three reduced-court returns from each starting direction;
B completes three from G1-start and one from M20-start. B's stored full-state
qpos/qvel/ctrl/qfrc_actuator is exactly identical to the corresponding frozen
station baselines. Independently integrated robot position/quaternion changes
are continuous; A is actual motion, not merely a renamed strategy. A combines
a forward receiving anchor, ballistic geometry prediction and a reference-speed
cap, so this comparison cannot isolate predictor causality. The M20-start third
return completes at an opponent in-court landing, not a third continuing volley
reception. These are not two competing LLM decision-makers.

The unchanged controller then executed all 16 predeclared development attempts:
seeds7/23/42/123, both starting directions, both strategies. Eight INIT failures
remain in the denominator. Independent all-ms contact/momentum, source/state
hash and full-state checks pass; an independent additive completion audit uses
the earliest OUT/force-positive landing/actual strong reception and requires
connected actual opponent receptions between completed returns. A reaches
three in 1/8 attempts; B in 0/8. Overall 1/16 is not a robustness PASS or a blind
holdout result. Recorded per-axis command RMS has separate m/s and rad/s units,
with flight duration; the legacy mixed-unit scalar is not energy. Original
net-only verdicts remain unchanged alongside these stricter additive reports.
ROSClaw authors and runs the physical controller; supervision only reconstructs
recorded evidence. Clearer frozen video and an isolated experimental source
review package remain in progress; no final physical acceptance is claimed.

A further real A/B U03 trial exposed a launch-directory false positive: the
clean A interpreter sees a PEP420 `rosclaw` namespace when its dependency probe
inherits a parent containing the source checkout. A therefore failed before
model startup despite having no installed ROSClaw. The probe now runs from the
dedicated A venv directory; genuine import contamination and version divergence
still fail closed. The namespace counterexample fails before repair, and 20
runner/profile regressions, Ruff and module mypy pass after repair. Original
startup ERROR is retained and excluded from model-quality scoring. The real B
U03 side independently verifies in 25.9s/four calls; repaired native PI A
verifies in 26.4s/two calls (14 inline Python lines). Both persisted model
identities are openai-codex/gpt-6.1-sol. Temporary OAuth files were removed
from trial directories and excluded from durable evidence copies.

A PI compatibility audit also found that `/effort` called a nonexistent
`ExtensionCommandContext.setThinkingLevel`; the old unit fixture incorrectly
invented that method. PI exposes it on ExtensionAPI, and `auto` is not a PI
ThinkingLevel. A real-shaped context counterexample fails with TypeError.
ROSClaw now injects the typed host API, restores per-model/global configured
default (medium fallback) for auto, and reports the actual capability-clamped
level. It changes the current session, not global settings. Missing host support
reports no change. All 23 related command/UI regressions and TS build pass; a
fresh real PI-backed ROSClaw TUI executes high → medium → auto with actual API
readback and no model turn/network call. PI lazily creates no session JSONL for
this command-only smoke, so no persistence claim is inferred. The formal active
physical session remains on its previously loaded code until safe maintenance.

The same PI audit found `/switch` using nonexistent instance
`ctx.sessionManager.listAll`, plus ignoring `{cancelled:true}` from host
switchSession. A real-shaped context fails with TypeError before repair.
The command now uses the existing static PI session-list adapter and shared
exact-ID/unique-prefix/title resolver, does not resolve an ambiguous prefix
via an unrelated title, and respects cancelled switches. It reports lookup
or host errors instead of claiming success. All 26 related command/UI checks,
seven PI dependency-boundary tests and TS build pass. A separate real TUI
command-only smoke correctly resolves a missing session without TypeError;
formal session history and physical jobs are untouched.

The formal native session later recorded a real `WebSocket closed 1012` at
09:24:05 UTC. The protocol registry assigns 1012 to Service Restart
([IANA WebSocket registry](https://www.iana.org/assignments/websocket)); the
close code alone does not establish which service or network hop restarted.
ROSClaw incorrectly rendered MODEL_UNKNOWN, and generic action BLOCKED
caused pure-SIM footer Operator Offline despite no operator reason. Both
counterexamples fail before repair. WebSocket disconnects now classify as
recoverable PROVIDER_UNAVAILABLE with a fresh-message recovery action,
without asserting an automatic retry. Pure-SIM unrelated blockage no longer
implies operator involvement; REAL, READY operator and explicit OPERATOR
reason behavior remain. Related model-error/UI regressions and TS build pass.
The same native task continued from its queued source-review message,
with all files and prior physical evidence preserved; no physical operation
was active or cancelled. Formal code reload is deferred until idle.


By the source-package review boundary, the complete original JSONL contains
2,193 entries and passes structural audit. All 152 assistant text claims have
been read against available evidence, including ten new claims after entry
2,017. Five compactions were reviewed; no additional compaction occurred in
this suffix. Structural integrity is not an oracle for physics or every hidden
reasoning statement.

Native ROSClaw authored the isolated experimental source package and executed
four fresh selected-seed closed loops. The supervisor independently verified
all millisecond force/momentum/state and actual receiving/court completion,
root integration, and direct old/new raw JSON equality without importing the
producer comparison. A completion [3,3] and B [3,1] are selected-case results;
the sixteen known development attempts remain A 1/8, B 0/8, with eight INIT
failures. New package qpos/qvel/ctrl/actuator force and millisecond ball/contact/
self-contact/root/actuation records are bit exact to their frozen counterparts.
Forty-three runtime function/class AST nodes match their frozen sources.

The supervisor repeated twelve package boundary tests, verified every SHA and
size in the 44-file public source manifest, checked all 83 external asset SHAs,
and verified the four independent receipt references. Only these source files,
the manifest, small evidence summaries and license notices were committed.
Large states, videos, meshes, weights, sessions and credentials stay outside
Git. The reviewed source was pushed to ros-claw/rosclaw-tennis branch
`audit/m21-embodied-harness-20261003` at
`a6513e1c303dd0a1a734ecbd0cb7f2ea9e29c553`. Clean installation remains NOT_RUN;
the measured existing Python environment is not claimed as a portable lock.
The original tennis working branch and user changes were not staged.

Both new oblique diagnostic films and their maps/timings (six artifacts) have
registered SHA values independently checked. Every frame fully decodes; the
projection oracle independently covers the compiled court and ball with at
least 17.544 pixel margin. Videos show complete 1x and quarter-speed traces,
explicit OUT endings, and the difference between third-shot landing and a
fourth actual reception. They remain simplified 12DOF/perfect-state/low-net
experimental paddle diagnostics, not standard tennis or two LLM strategists.
The task remains RUNNING/NOT_DONE. The next native protocol targets initial
stability with unchanged thresholds and retained failures, separate from the
frozen baseline.


Subsequent native INIT-only tests reject two proposed stability fixes. Four
fixed-five-second baselines retain their original single-frame result (2/4),
while four extended-wait cases (at most twelve seconds) achieve zero genuinely
continuous 0.200-second windows. A sample count of 200 spaced one millisecond
covers 199 milliseconds; acceptance uses elapsed timestamps and at least 201
samples. An independently verified first-five-second raw prefix is byte exact.
A separate four-case G1 outer-loop candidate retains full velocity damping and
smooths the near-anchor position multiplier, but also yields 0/4 sustained
readiness and higher lateral RMS (~0.0936 versus ~0.0868 m/s). It is rejected
for duel use. Its complete 1 kHz numeric-array finite checks, all-qpos integration
and actual command-formula reconstruction pass; valid records do not make a
failed controller effective.

A further native paired seed-23 measurement captures the actual neural-network
input and return tensors through the original call, without extra inference or
argument/result replacement. The initial draft reconstructed an observation
and mislabeled it actual; supervision caught that before execution. Each final
condition contains 330 actual calls, independently checked against state-derived
47D observations and sequential replay of the pinned checkpoint (zero errors).
The checkpoint is recurrent: forward updates hidden_state and cell_state buffers
of shape [1,1,64]. The supervisor's first oracle incorrectly reused one network
across two fresh cases, producing a false mismatch; fresh loading per case and
ordered replay repairs the oracle. Zero three-axis motion references still
produce changing 12D actions/targets and alternating recorded foot loads in the
measured 1.6-second interval. This does not establish phase-only causality or a
successful stand mode. The six complete session compactions have been reviewed;
current failures are present in the retained suffix after the sixth prefix
summary, and queued measurement corrections survive compaction.

The supervisor initially misinterpreted actuator-only limit flags: all twelve
G1 leg motors have ctrl_limited=false and force_limited=false, but joint-level
actuator-force limits ARE enabled. Complete compilation/introspection and
independent comparison to committed upstream URDF bytes correct that inference.
The twelve joint actuatorfrcrange bounds are +/-[88,139,88,139,50,50] Nm on each
side, matching unitree_rl_gym@276801e46c5d433564f24658bac64f254b7d2d4b
resources/robots/g1_description/g1_12dof.urdf. No world edit or extra guessed
clip is required. M20's twelve leg motors have control ranges +/-76.4 and four
wheels +/-21.6; both experimental paddle servos have control and force limits.
A disabled [0,0] actuator range is neither a zero-torque bound nor proof that
joint-level bounds are absent. Requested control, individual actuator force
and final joint actuator-force values must be distinguished when the joint
layer clamps their aggregate. These are source/model effort bounds, not a
thermal or continuous-duty hardware certification. The proposed stand-reference
candidate remains unexecuted pending source/execution review; original model
and prior evidence stay frozen. The erroneous actuator-only inference is
preserved in audit history and explicitly corrected here.
