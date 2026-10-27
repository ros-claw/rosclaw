# Process recovery and explicit cancellation

`OperationManager` records Linux process birth evidence in the existing
provider-private `operations.checkpoint_json` field at admission. This change
adds no migration and rewrites no previously applied migration checksum.
The checkpoint contains PID, kernel start ticks, process group, session, UID,
boot ID, working directory and a command digest. It contains no command or
environment plaintext. Existing admission events remain unchanged.

A live numeric PID alone cannot authorize recovery. Reattachment requires the
stored identity to match the current independent session leader. Missing,
malformed, reused or changed identities produce `LOST`, without adopting or
signalling that PID. Previously stored checkpoint bytes are retained. Systems
without the required Linux process evidence cannot prove reattachment.

Explicit cancellation of a recovered process snapshots the proved session,
including children in separate process groups. Signals use kernel pidfds and
birth checks rather than an unverified numeric PID or process group. After
SIGTERM, the existing five-second grace is followed by SIGKILL only for
previously proved, still-matching members. Cancellation is confirmed only
when the inspected session has no live members. Unavailable evidence, newly
unproved members, or an unconfirmed stop produce `LOST` with
`process_cancel_unconfirmed`; Native `process_stop` reports an unsuccessful
result requiring reconciliation. There is no automatic restart or replay.

Cancellation while spawn is pending owns the eventual admitted child. The
same cleanup task survives loss of its requesting coroutine, and explicit
supervisor shutdown joins that task. Closing a recovery watcher alone does
not signal recovered operations that were not explicitly cancelled. Liveness
checks still do not introduce a default wall-clock kill.

A persisted `CANCELING` row restores its original explicit cancellation owner,
not an ordinary PID watcher. Recovery either proves and finishes that stop or
records `LOST`; it cannot remain indefinitely `CANCELING` or use a late
exit-code file to overwrite cancellation intent with success. This resumes a
previous stop request, never the interrupted command.

These guarantees concern managed host processes. They do not grant robot
permissions or prove physical motion stopped. Canonical Session, Lease,
Permit, action cancellation, native terminal and independent stop evidence
remain separate requirements. Existing terminal rows are not retrospectively
promoted to verified stops. Output-spool recovery and compatibility migration
of databases from other development branches are outside this change.

Regression coverage in `tests/agentd/test_recovered_process_cancellation.py`
uses actual detached temporary host processes: abrupt supervisor exit,
birth/boot mismatch, unproved cancellation, pending spawn, a SIGTERM-resistant
child in another process group, requesting-coroutine loss, shutdown joining,
and Native start/stop status preservation. It does not connect to ROS, a model
provider, or a robot.
