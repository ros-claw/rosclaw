# Operation restart recovery diagnosis

## Evidence and limits

CI39067648647 at eb3d956f58e10a78dc527373f8ad6c1f87182dbb remains **FAIL**:
1 failed, 9790 passed, 116 skipped, 877.44 seconds. The dead/no-exit test
expected LOST and observed RUNNING. Root's separate fresh-home run of the
three original recovery cases passed. Neither fact supersedes the other.
No failing-session snapshot, writer-lock observation or checkpoint was saved;
the exact historical interleaving is UNKNOWN. Nothing here was executed or
reproduced by the author: only the seven supplied inputs were read and source
edits authored. All tests below are UNEXECUTED, pending independent Root review.

The earlier provider-interrupted purpose remains CLOSED1/failed, with no
accepted component. Its request7 usage remains UNKNOWN; its six known requests
are a lower bound, not total episode usage. This is a separate author purpose.

## Source-grounded diagnosis: wrapper death is not output EOF

* `_spawn` opens a regular spool and takes an exclusive flock before Popen.
  stdout and stderr inherit that locked open-file description through dup,
  fork and exec. Closing the parent's fd does not release descendants' copies.
  The shell runs the payload, then writes the real payload status via a
  temporary exit file and rename. Killing the shell can prevent that write.
* `_PreservedProcess.wait` polls/reaps only its retained Popen shell. It does
  not join grandchildren, inspect the owned session, or prove spool closure.
  `_reap_finished_children` likewise only reaps retained shell handles.
* `killpg` is signal delivery, not a join barrier. In the old fixture, wrapper
  and descendants receive SIGKILL, but the wrapper can finish and be reaped
  before another killed process has closed its inherited spool descriptors.
  A nested shell can also introduce extra process lifecycle boundaries.
* `_pid_alive` excludes a zombie leader, whereas `ProcessIdentity.matches`
  establishes the complete persisted eight-field live leader identity.
  Neither proves that every output writer has closed. Session snapshots can
  identify owned live descendants, but do not encompass writers outside the
  session; the output lock is a separate, conservative observation predicate.
* In the spool branch, `recover_on_boot` synchronously drains a dead producer
  only if `_writers_closed` also proves closure. With a dead leader but an
  outstanding writer it instead starts a spool observer and returns. That
  observer can leave/set RUNNING until the writer closes. The reported
  `reattached` count there denotes spool observation, not new stop authority.
* `_drive_spool` checkpoints bytes and pending UTF-8 together with output
  events, waits for writer closure and confirms EOF with another read before
  finalization. With no attached Popen after recovery, missing exit evidence
  means LOST; a genuine exit file means the genuine terminal result. A local
  Popen rc is not available after a restart and must not be invented.
  Terminal guards and CANCELING ownership prevent late terminal overwrite.

Thus the original dead/no-exit fixture had a concrete invalid implication:
`killpg + await wrapper.wait()` was treated as proof of all-writer closure.
A possible interleaving is: all group members signalled; wrapper reaped;
child still closing output; recovery sees locked spool and returns with an
observer; assertion sees RUNNING. This is a source-permitted explanation,
NOT a claim that this was the recorded CI cause. A permanently retained writer
is also meaningful evidence, not permission to force LOST.

## Change rationale

`operation_manager.py` is intentionally unchanged. Removing the writer gate,
forcing LOST on leader death, or using exit-file presence alone would truncate
remaining output or finalize partial UTF-8. The existing synchronous closed-
writer/dead-producer path already implements the asserted reconciliation.
No product reconciliation defect requiring an in-scope source edit has been
established. If Root observes a failure with independently proved closed spool
and dead leader, that is new evidence requiring investigation, not a reason
to bypass this gate now.

The original recovery assertions are preserved. The fixture now uses the
public preserve-only close to settle its original observer. The no-exit case
uses a ready-marked, no-fork Python payload; snapshots the verified owned
session; signals exact captured births; awaits the wrapper; and independently
acquires writer-closure evidence before the single recovery call. Missing exit
evidence is asserted, not manufactured by deleting files. The known-exit case
also proves dead/closed prerequisites and verifies the genuine exit value 3.
The live case independently checks matching identity, live PID and held lock.

The bounded prerequisite helper polls evidence, not expected recovery states;
it fails after five seconds if evidence is unavailable. It never retries
recovery to turn the original assertion green. Marker files and a stdin byte
provide process coordination; delays alone establish no premise or outcome.

## Discriminating coverage

`TestRestartRecoveryEvidence` adds:

1. Dead shell plus a separately owned, controlled spool writer, both without
   an exit file and with genuine exit 3. The original producer is held behind
   a release marker until its observer has handed off, preventing an accidental
   already-terminal setup. A retained Popen child independently holds the same
   kind of exclusive spool lock and writes an incomplete UTF-8 prefix before
   advertising readiness. While blocked on stdin it cannot finish. Recovery
   must observe, not mark LOST/FAILED or finalize output. This is a deliberate
   writer-lock counterexample, not a reproduction of the CI schedule or an
   assertion that the independent child belongs to the original session.
2. Preserve-only handoff while that writer is blocked, followed by writer
   release and a second manager. The exact output `head€-tail`, cursor, empty
   pending bytes, finalized checkpoint, genuine FAILED/LOST event and report
   are checked. A second reconciliation must not duplicate events, and late
   success must not overwrite the terminal ledger. Ignoring the writer gate,
   discarding pending UTF-8, or rewinding the cursor fails these checks.
3. A verified live producer with deliberately missing or changed-start-ticks
   persisted identity. Recovery must be unresolved DEGRADED, not adopted,
   driven, LOST or heartbeat-resumed; stored identity bytes stay unchanged and
   the original identity still matches the untouched live producer. Only the
   test-corrupted row is restored for cleanup; no identity is recaptured as
   replacement authority.

## Independent Root verification (not run here)

After reviewing the diff, Root may run a bounded local-only check in a fresh
private home, with no ROS/SDK/network/hardware dependencies:

```sh
home=$(mktemp -d)
ROSCLAW_HOME="$home" timeout --kill-after=10s 90s python -m pytest -q \
  tests/agentd/test_p1b1_operation_v2.py::TestRestartRecovery \
  tests/agentd/test_p1b1_operation_v2.py::TestRestartRecoveryEvidence
# After confirming owned child cleanup, remove only this temporary home.
```

Expected collection is three original plus four parametrized evidence cases;
acceptance requires actual test results, not this expectation. Root can then
run the whole file under a separate bounded local-process approval, reviewing
unrelated baseline fixture ownership before doing so. A timeout is a failure,
not proof of cleanup or success; do not loop reruns until green.

Cleanup uses retained Popen children and verified persisted session identities,
never system-wide process matching or unknown/reused PID adoption. Manager
close joins observers without requesting producer stop. Owned wrapper waits
and external-writer waits are bounded; temporary markers/spools live under
pytest's private tmp_path. The independent writer is killed through its own
retained Popen handle if needed. Session members receive birth-checked signals;
zombies are not live writers. If identity or session inspection is unavailable,
cleanup fails closed instead of signalling unknown processes. Root should
inspect any timeout/unresolved cleanup before deleting temporary state; an
outer timeout alone cannot guarantee descendant cleanup. No process or test
was launched during authoring, and no verification pass is claimed.
