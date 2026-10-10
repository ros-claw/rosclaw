# Owned UI cancellation provenance (Source7)

This is source-only software work continuing Native6, not provider, model-stream,
PTY bootstrap, ROS, physics, or hardware evidence. Previous aggregate-budget
failures remain failures; they earned no whole-task PASS.

## Receipt and ownership

The original admitted `process_start` constructs the ownership receipt from the
current production request, durable input/turn/task and operation. The production
extension's original ToolResult callback validates that receipt and updates
per-session owner and turn maps. No turn is synthesized. The only new optional
options hooks are `uiReceiptOwners?: Map<string, UIReceiptOwner>` and
`uiReceiptTurns?: Map<string, string>`; existing options remain intact.

Production main invokes the shared per-instance `attachOwnedUIAbort` adapter.
The SDK UI fixture observes editor Escape/Ctrl-C dispatch into the original
SDK restore and session abort, not arbitrary watchdog/programmatic aborts. The
actual Main bootstrap is separately exercised; the fixture's late prototype
replacement test covers that replacement shape only, not all installed SDK
hook mutation timings. Original SDK modal, Bash, clear-editor, retry and
compaction handlers retain their distinctions.

The declared UI RPC is **`pi.op.cancel_owned`**. The original authenticated UDS
server checks peer writer identity, session binding, mission, task revision,
current turn and the server-created receipt before effects. Client tuples do not
create authority. A matching SUCCEEDED operation with recorded end and no live PID
returns ALREADY_SUCCEEDED (quiescent, zero cancellations); FAILED, LOST, stale,
foreign and mismatched receipts never become successful cleanup. A second durable
input updates the current turn from the writer-only pi.turn.latest projection and
invalidates its cached prior-turn owner: Escape on that turn does not submit a
cancel for the prior worker or claim it stopped. Cancellation of a still-running
owned operation delegates to OperationManager with process-identity and
stop-confirmation safeguards.

`pi.op.cancel` remains the legacy private authenticated operator control route;
it does not provide receipt-bound UI ownership. Its original no-turn/no-memory
receipt unresolved-stop contract is retained, including CANCEL_STOP_UNCONFIRMED,
zero cancelled operations and CANCELING while stop cannot be confirmed. The UI
adapter never falls back to that route. Private control scope is not an assertion
that all legacy control endpoints implement strict UI ownership.

## Pending, retry and cleanup

Each adapter instance tracks pending requests and deduplicates an identical
in-flight or confirmed receipt. Failure does not mark a receipt confirmed, so a
later genuine input may retry. The latest outcome for that receipt supersedes a
failed attempt only when the retry actually resolves. Ownership is checked both
before sending and after resolving; a changed session/task/turn is not falsely
reported as confirmed cleanup. RPC rejection becomes a typed failed outcome,
not an unhandled rejection. `drain()` awaits all pending work and returns typed
outcomes. Main awaits it at its genuine interactive finally boundary and records
unconfirmed cancellation as teardown failure before writer release. Existing
confirmed close, stop, writer release and exit-code semantics are preserved.
Repeated input is not evidence of repeated Kernel cancellation effects.

Assistant `message_end` is an outcome, never a provenance signal for an editor
Escape/Ctrl-C. The old extension callback inferred user intent from `aborted` or
`MODEL_REQUEST_CANCELLED` and called `pi.session.interrupt` after the typed
adapter's owned RPC. That legacy server route can cancel active-task operations
and globally kill renders. The callback no longer invokes either the legacy or
owned cancellation route. Provider error cards, deduplication, provider health
and watchdog abort handling remain in production; this callback-only regression
has no initialized latest UI context and does not establish provider UI behavior.
Programmatic/watchdog/provider aborts cannot grant user-input authority. The
legacy explicit operator route remains intact. Escape evidence does not establish
Ctrl-C equivalence; actual fullMain event acceptance is separately required.

A synchronous `scene_render` has no authentic owned operation/PID receipt. Its
cancellation therefore fails closed: neither the editor adapter nor a model
message may claim the render was cancelled, and no global render kill is used
as a fallback. Owner-bound renderer cancellation is separate work.

## Evidence limits and integration

Public checks must run against current bytes after build. The authentic fixture
uses the installed SDK, original production input and updater, shared main
adapter, real private UDS and Kernel workers; maps start empty. Only unique CLI
autostart suppression is test-only. This is not full terminal/agent bootstrap or
provider streaming evidence. Protected Main6 exercises healthy shutdown and five
faults on the unchanged compiled CLI. Reported checks and counts must come from
actual checker receipts; absent or RED proof remains SOURCE_FAIL.

No G19 snapshot work is restarted or overwritten. Promotion requires merging
these narrowly admitted changes without replacing newer artifact_snapshot logic;
conflicts require explicit source repair and new verification.
