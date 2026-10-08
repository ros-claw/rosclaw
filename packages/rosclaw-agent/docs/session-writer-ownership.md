# Session Writer Ownership — residual concurrency repair

Scope: current Native source-only software controls, not historical session-loss
causality, a global concurrency proof, or an OS sandbox.

## Explicit stale fail-closed policy

Each `SessionWriterOwnership` instance has a random token. Same PID is not same
owner. Canonical real paths unify directory/file symlink and dot aliases within
this path model. A cooperating writer reserves `<canonical>.owner.json` before
SDK open/initial append/provider work. New claims use `openSync(..., "wx", 0600)`
(O_EXCL): only one creator can succeed. Same owner remains idempotent; distinct
sessions remain independent. Normal token-only release permits subsequent resume.

**Automatic stale reclaim is disabled.** Every foreign existing lock is refused
by both acquire and check, including a lock whose recorded birth is demonstrably
dead. No acquire/check path renames, unlinks, restores or overwrites an existing
lock. A dead birth is diagnostic, not permission to mutate authority. UNKNOWN
(unreadable/malformed lock, unavailable boot ID, EACCES reading proc) is likewise
refused unchanged. No PID-only/TTL steal and no signalling external processes.
Crash or failed/partial lock write may leave the session unavailable indefinitely.
This is an intentional availability tradeoff, not transparent crash recovery.

The prior marker precheck followed by open, and empty-slot check followed by
restore rename, did **not** establish exclusion. The supplied finite three-process
receipt observed A/C both acquired while B restored A over C. This repair removes
that movement/restore transition rather than claiming a marker mutex works.
There is no current reclaim critical section to synchronize. New acquisition's
sole authority transition is O_EXCL; a foreign lock never moves. Legacy `.reclaim`
markers are still refused (including unreadable markers), left unchanged, and
never created by this version. Their precheck is advisory compatibility refusal,
not a synchronization primitive. Concurrent old reclaiming versions are not
supported: upgrades must quiesce all writers first.

`lockOwnerAlive` retains Linux boot_id + `/proc/<pid>/stat` starttime diagnostics;
UNKNOWN conservatively returns true. Neither false nor lock age authorizes
recovery. On platforms lacking proc, existing foreign claims still fail closed.
`check()` is advisory and creates no reservation: public switch must acquire
before SDK open, not rely on check-to-open timing.

## Manual verified-dead recovery (operator only)

1. Disable admission and stop **all** writers/reclaimers for this canonical
   session, including old versions and external SDK users. Keep them stopped
   throughout inspection and recovery; a snapshot followed by removal while
   writers can start is unsafe.
2. Identify the canonical transcript and its owner record, legacy marker and
   any old `.stale-*` records. Preserve evidence privately without dumping
   transcript/auth/secrets. Establish recorded process death with boot-qualified
   birth identity and that no writer or reclaimer is active. PID absence alone,
   age, malformed data or access denial are not sufficient operational clearance.
3. If authority/liveness is UNKNOWN, do not remove or overwrite it. Investigate
   or use a different session. Never recover a live owner claim.
4. Only under verified quiescence and verified-dead authority may the operator
   archive the residual dead lock/marker out of the acquisition path. This is
   not a product auto-unlink shortcut. Do not merge/truncate JSONL histories.
5. Deploy one current version to every cooperating entry point before reopening
   admission. Normal acquire creates a fresh token with O_EXCL. If verification
   cannot be completed, keep the affected session refused.

## Preserved lifecycle seams (other five source files unchanged)

| Seam | Ownership behavior |
| --- | --- |
| main new chat/runtime creation | eventual file acquired before initial model/thinking append |
| CLI resume/resume-path/continue/browse and backend resume | `openPiSession` acquires before SDK `SessionManager.open`; denial before UI/provider/append |
| replacement factory | incoming file acquired before agent runtime initial writes |
| public `session_before_switch` | target acquired as reservation before SDK open; foreign live/UNKNOWN/stale veto leaves no claim |
| same-current-file branch/switch | existing own claim idempotent, not marked as a new reservation |
| target open/cwd failure before teardown | releases only new target reservation, retains old live exclusive claim |
| post-teardown replacement failure | releases target; does not assert torn-down SDK runtime remains alive |
| initial constructor failure | releases own claims only |
| adapter close | single-flight abort, strict confirmed idle, successful disposal and post-disposal idle; only then releases own claims |
| main bind failure, print/UI return or rejection | abort + strict idle + awaited session disposal + strict idle + host disposal + strict idle before ownership/lease release |
| crash | stale claim retained and refused; manual quiescent verified-dead recovery required |

Release checks the own token; foreign/UNKNOWN replacements remain untouched.
This assumes cooperating current writers and no concurrent administrative lock
replacement. External mutation during release is outside the supported protocol.

## Failed or unknown adapter close

Close shares one promise across concurrent and repeated calls. Abort rejection,
disposal rejection, idle false/unknown or a throwing idle getter keeps the claim
and rejection visible on repeat close; there is no automatic retry or recovery.
An unresolved abort/disposal retains exclusivity. Local event consumption stops
immediately on the first close request, independently of the remote outcome:
registered waiters are woken, timers cleared, subscriptions detached exactly once,
and already-pending reads finish with done=true. Newly opened iterators finish
without subscribing. Even an iterator paused at yield is unsubscribed locally.
This local closed flag is not confirmation that the remote agent stopped; it
never bypasses either strict idle check, releases ownership, or hides the sticky
close rejection. Healthy close releases once; subsequent close remains successful
and resume is allowed. The healthy contract fixture explicitly supplies SDK
isIdle=true; unknown/throw/nonidle negative controls remain unchanged.
Retained authority is an intentional availability tradeoff requiring operator
quiescence, not permission to remove a possibly live writer's lock.

Tests inject AgentSession faults against the product adapter and filesystem
ownership helper in temporary directories. These are not actual provider behavior
or evidence of a historical cause. The ownership algorithm and other lifecycle
seams are unchanged; the current repair changes only main exit wiring and its tests/docs.

## Direct main exit confirmation

Main retains its InteractiveMode handle and calls public stop in a finally, even
when run rejects before initialization. This closes local watchers/subscriptions;
it is not a remote-stop receipt. This narrowly scoped dynamic SDK construction
replaces the helper that discarded the consumer handle, without changing it.
Print and startup attach/resume failures share the outer confirmed teardown.
Main explicitly awaits session disposal because host dispose alone does not
await an asynchronous session dispose. Host cleanup follows confirmed session
teardown. Any rejection/unknown idle throws MAIN_EXIT_TEARDOWN_UNCONFIRMED and
retains writer claims and lease; there is no confirmation retry. A best-effort
local agent abort, idle drain and host disposal on failure stops local extension
consumers via genuine terminal events but never clears the first failure or
authorizes release, even if that local cleanup succeeds. No synthetic terminal
event or remote-stop claim is generated. The original UI error remains
visible when teardown succeeds. Exit uses exitCode, never forced process.exit,
so live resources are not hidden. Pending teardown keeps ownership indefinitely.
Existing protected adapter single-flight/sticky behavior is unchanged.

## Evidence and limits

Native tests cover same-PID exclusion, aliases, token-only release, stale dead/PID
reuse refusal with byte preservation, three stale contenders with zero destructive
calls, precheck-to-open contender insertion excluded by O_EXCL, malformed and
unreadable authority, residual marker refusal and normal recovery after fixture
quiescent cleanup, plus existing open/switch/runtime/close/failure seams. FS timing
injections are finite software schedules, not proof of all possible interleavings.
Public controls independently cover actual backend lifecycle13, atomic3, and
private two-CLI behavior. Current compiled checker receipts, not test intent,
determine SOURCE_PASS. Fake-provider/private owned software checks perform no
real provider requests, robot/physics work, retry or supervisor feedback.

Mission/action lease protects authorization, **not JSONL file exclusion**. This
mechanism is separate and changes no lease/auth/provider/tool policy. Direct SDK
writers bypassing ROSClaw are not blocked; no fcntl/flock or OS sandbox guarantee.
Cross-machine/NFS identity, hardlink aliases, adversarial directory replacement,
external lock edits and concurrent mixed-version upgrades are not qualified.
Interactive resume/new ultimately use public switch/factory seams; fork does not
add a separate file ownership hook. Existing transcript branches are not repaired.
No claim is made about a historical hang/session-loss cause or universal SDK
writer safety.
