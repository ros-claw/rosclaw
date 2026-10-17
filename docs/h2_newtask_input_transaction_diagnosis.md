# H2 new-task input transaction diagnosis

## Status and evidence boundary

SOURCE-ONLY; all authored tests are **UNEXECUTED**. All nine published inputs were read, including both paginated production files. Production files are unchanged. This is a test-fixture synchronization correction, not a demonstrated product race fix.

Reported historical evidence: PR721 a8cf CI39071017223 FAILED (9794 passed, 116 skipped, one failed, 1332.73s). H2 expected two tasks after `/newtask`, observed one at line 119; earlier restart16 did not fail. The original intermediate PTY/ledger artifact is unavailable: exact historical cause remains UNKNOWN.

Root's separate original-H2 short-directory baseline56723 CLOSED0/1PASS in 8.790870s, source pre/post exact, known children gone with zero signals, using a real loopback FakeServer and no paid model. Its classification is BASELINE_PASS_NOT_HISTORICAL_FAILURE_REPRODUCTION, not evidence of a flake or fix. Baselinev1's too-long home path failed before GUI; the monitor's self-including cleanup exited241 without a success receipt. Those preparation failures are not product fixes. That baseline used isolatedPR721606/oldUX, not this e47d5557 seed. The accepted/frozen e47 UX is untouched. The old CI FAILURE is not erased.

## Source chain and authority

1. `extension/index.ts` registers public `newtask`: synchronously sets `inputController.forceNewNext = true`, then notifies `下一条消息将开始新任务`. The notification acknowledges arming the next input, not creation of a task. It performs no admission/RPC itself.
2. The natural-language input hook records a turn (best effort), then awaits `InputController.persist(text)`. A null result prevents model delivery. Otherwise the current bridge returns PI_CONVERSATION/suppress=false and the hook continues; no root is created just by input.
3. `InputController.persist` allocates a message UUID, sends `pi.input.persist` with the flag and current mission/session, awaits its response, clears the flag, and fire-and-forgets `pi.input.dispatched`. A transport exception notifies and returns null. The supplied RPC excerpt calls `TaskKernel.persist_input`, appends INPUT_PERSISTED, and returns the input/disposition. This diagnosis does not infer transaction commit mechanics from the selected server excerpts.
4. `persist_input` is message-ID idempotent and inserts `user_inputs.delivery_state = FORCE_NEW` when requested. Its immediate return dictionary says PERSISTED even for FORCE_NEW; the stored row, not that return label, drives admission. No task is created here.
5. At first effectful admission, `ensure_task_for_effect` selects the latest input for the mission/session (created_at then rowid). Missing motivation rejects; an already-bound row returns its task without another revision. An unbound row passes its message ID/text and stored FORCE_NEW to `bind_message`, then receives the resulting task_id. The supplied ensure-effect RPC checks schema, envelope mission/session, peer PID, session binding and writer ownership before mutation. The supplied wire-dispatch helper also uses the kernel; neither fixture markers nor model prose grant admission authority.
6. `bind_message` first honors message-ID replay. FORCE_NEW deactivates the old primary binding before terminal/SUCCEEDED-reopen logic. Otherwise a live root gains a revision; unaccepted SUCCEEDED can reopen the same root. A new root gets revision1 and a primary binding. The new fixture checks input/message/revision/task/session linkage read-only after the real tool path; it does not call this kernel directly or seed rows.

The latest-input rule deserves separate scrutiny if overlapping inputs/effects are shown, but the supplied sources alone do not establish such an interleaving in the historical failure. No speculative production change was made.

## Defensible invalid fixture premise

The old driver waited for a constant final phrase after `len(session.clean)`. `PtySession.send` only writes bytes; it is not an input-handler acknowledgement. Its background drain appends stripped chunks to `clean`; `expect(after=...)` searches that append-only transcript, not logical screen state. A redraw can append the previous turn's identical final phrase after the marker. Therefore an offset excludes already-drained bytes, not previous logical output. This is a demonstrably insufficient prerequisite for proceeding to the next input or querying its resulting task count. It is not proof a redraw caused the historical failure.

H1's actual provider uses constant call_write/call_bash tool-call IDs to choose write → bash → final. Its final prose is identical for every input. `_FakeResponseIdentity.apply` already assigns one process-unique HTTP response ID across that response's SSE chunks; it does not rewrite tool-call IDs or make the visible final text input-specific. Changing response IDs again would not fix the PTY predicate. Reusing tool-call IDs is not itself diagnosed here as a production failure.

The old two-second sleep after `/newtask` likewise proved no handler acknowledgement. Missing acknowledgement is an independently missing prerequisite, not an assumed historical cause. A command acknowledgement alone would still leave the repeated-final/redraw ambiguity.

## Narrow correction and checks

Only the H2 test and this document change:

- Wrap the existing H1 fake answer method, retaining its write/bash frames and tool-call routing. Only final presentation gains a deterministic, distinct marker for each of the three real input texts, selected from the latest matching user message, never assistant history. The original final phrase remains. The real HTTP handler still applies response identity rewriting.
- Wait for the input-specific final within the existing 180s bound. An old final redraw cannot satisfy the next marker. This is fixture orchestration evidence, **not task success authority**.
- Immediately read the real ledger join from user_inputs.message_id through task_revisions to tasks and the session's primary binding. Require exactly one row, input.task_id agreement, active revision agreement, active binding, and FORCE_NEW for the third input (PERSISTED for the ordinary inputs). This also discriminates a visible answer without real admission. No polling, sleeps, retries, admission stubs or task creation are added.
- Replace the two-second sleep with a fresh-offset wait for the actual command acknowledgement (60s failure bound). Only then send the third input.
- Preserve every original assertion, including input echo/presence, first root/revision1, followup one root/revision2/two revisions, final two roots, and zero work_orders/task_executions. These remain on the real PTY/RPC/tool path.
- On the existing finally path, stop the owned PTY and wait for its process after the existing terminate/kill logic; close the owned fake server even if waiting fails. Existing fixture startup and daemon lifecycle behavior are not broadened or claimed fully audited from selected excerpts.

An appended finite regression exercises the actual H1 producer and actual response-ID rewriter across three user/write/bash/final sequences. It verifies tool routing IDs, final stop, nine distinct response IDs and per-input markers. Its negative control appends an old final as a redraw: the old phrase-after-offset predicate accepts it while the next-input marker rejects it. Synthetic tool results in this isolated producer test test presentation only; they are not the integration gate's admission evidence. The original integration still executes real write/bash admission and checks the ledger.

Tempting wrong fixes rejected: increase sleep/timeouts; retry until count2; weaken count2 to count1; precreate a root; infer force_new from different goal wording; swap only response IDs; treat a fake answer as root-task authority. In particular, changing only the command wait or byte offset does not pass the redraw negative control.

## Bounded Root verification plan (not executed here)

1. Review AST preservation/scope, Python parsing, fixture wrapper binding, current provider message shape and read-only ledger schema/state semantics. Independently compare all three production files byte-for-byte with the e47 seed. No build or execution occurred in this author purpose.
2. Under separate explicit permission, run the appended producer regression once, then the original H2 node once in an isolated short home/workspace using pinned built entry and loopback fake only. No paid model, hardware, credentials/global configuration mutation, or broad test suite is needed for this check. Keep the existing finite per-step bounds; no automatic reruns.
3. Retain PTY raw output and input/binding/revision ledger snapshots. Verify three distinct finals, actual newtask acknowledgement before third input, FORCE_NEW on that input, its new revision1 root and deactivated old binding, followup revision2, and every original assert. A marker mismatch must fail, not fall back to generic prose.
4. Use an outer bounded owner-aware cleanup monitor for fixture startup failures and descendants, excluding the monitor itself. Confirm process/server ownership and exit receipts; inspect failures rather than expanding waits. The selected source excerpts do not establish all descendant cleanup details.
5. If it fails, classify the earliest broken prerequisite: marker/provider shape, command handler acknowledgement, persisted FORCE_NEW, input-to-revision linkage, or final task count. A durable FORCE_NEW row admitted to the old root would justify further kernel investigation; no such observation is claimed now. Passing this revised gate still would not reproduce or explain the original CI failure, and source-only review is not a passing verifier.
