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
