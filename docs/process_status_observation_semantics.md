# Truthful read-only process status

SOURCE_ONLY_NOT_TESTED

## Two independent observations

`process_status` preserves the ledger's `operation_id`, `task_id`, `revision`,
`state` and `failure_code`. It does not resume, cancel, reconcile, adopt or
reclassify an operation. Tool success means the status read succeeded, not that
work succeeded. Terminal and CANCELING states retain their existing meaning.

The human/model-visible summary and structured `operation.observation` expose:

- `producer_freshness.heartbeat_at`: normalized recorded producer timestamp,
  or null for missing, malformed or timezone-unknown values.
- `producer_freshness.age_s`: elapsed seconds at the read, or null when unknown
  or in the future. No new stale threshold is imposed by status. This timestamp
  represents recorded producer freshness, not physics or work progress.
- `process_identity.status` and `reason`: an instantaneous, read-only identity
  comparison, independent of heartbeat freshness.
- `observed_at`: time the observation completed, not a lease or continuing
  guarantee. The ledger read, timestamp calculation and process capture are not
  an atomic snapshot.
- Bounded fixed warnings and limits. The original failure code remains in the
  owner payload and summary; observations never clear unresolved warnings.

DEGRADED is a ledger state, not a unique cause. The frozen manager can degrade
for stale producer freshness, unresolved process identity or unresolved output
observation; some non-sweep transitions refresh heartbeat. State alone therefore
does not establish stale freshness, and a fresh heartbeat does not clear an
unresolved failure code. Status reports state/failure code as recorded facts and
timestamp/identity evidence independently; it does not reconstruct event cause
from timestamps or assume a cause when the failure code is empty or unknown.
Neither DEGRADED nor heartbeat age establishes that a process was killed, is
dead or stuck, or is alive or making progress. A quiet locally owned process
can have stale recorded freshness while its exact identity is observed present.

## Identity boundary and uncertainty

Only process-provider rows with a valid stored `process_identity_json` and an
integer ledger PID equal to the stored identity PID are eligible. The stored UID
must also belong to the actual local boundary `{os.getuid(), os.geteuid()}`,
observed read-only at status time; otherwise status returns UNKNOWN without
probing. Matching stored/current foreign UIDs are not local ownership evidence.
Parsing uses the existing `ProcessIdentity.parse` boundary (bounded stored
string); capture uses `ProcessIdentity.capture` for that one eligible stored PID.
Exact dataclass equality is required: birth, group, session, owner and boot
identity plus captured cwd and command digest must all match. Thus a changed
current UID also remains UNKNOWN; equality cannot bypass the local UID check. No raw identity, cwd, argv, environment,
command digest, authentication/session data or capture exception text is added
to the response. Existing operation/task identifiers remain available.

`OBSERVED_IDENTITY_PRESENT` / `EXACT_IDENTITY_MATCH` means only that the exact
stored identity was captured at this observation. It is not a claim about work
progress, performance, service readiness, physics readiness, physical outcome
or safety. Even a same-birth process that changed cwd or exec identity remains
UNKNOWN under this deliberately strict comparison.

All absent/invalid stored identities, empty-command legacy markers, conflicting
stored PID, capture failures/None (including unreadability, disappearance,
zombies and races), foreign/reused/changed identities are UNKNOWN. Current
capture failure is not proven process absence; numeric PID presence alone is
not ownership. A mismatch is not adopted. No dead/alive boolean is fabricated.
No group scanning, kill(0), signals, spawning, output collection, heartbeat
refresh, SQL writes or event emission occur in the status body/helper.

Identity presence never resolves `PROCESS_IDENTITY_UNVERIFIED`,
`OUTPUT_OBSERVATION_UNRESOLVED` or `CANCEL_STOP_UNCONFIRMED`. It is not evidence
of whole-session cleanup or stop confirmation, including for terminal rows.
The existing cancellation/retry/deadline/terminal logic is untouched.

## Scope and validation still required

Only the dispatcher status body and one observation helper, the Node status
description literal, this document and the new isolated test module are
changed. Output handling, TUI, lifecycle producers, ProcessIdentity and existing
regression tests are frozen. The normal outer bridge authorization, auditing
and idempotency mechanisms are unchanged; the read-only guarantee here concerns
the operation observation seam, not removal of existing bridge bookkeeping.
Replayed bridge responses remain observations at their original time.

No tests, TypeScript checks, commands, processes, SDK or physics runs were
executed by the source author. This is source-only work, not a verified pass.
Root must later validate:

1. Python unit collection and execution of the new status module plus existing
   liveness, cancellation, recovery, output and bridge regressions.
2. Production schema/ProcessIdentity field compatibility, serialized nested
   `operation.observation` preservation, and Node summary/details projection.
3. TypeScript type/build checks and an exact diff/seam check confirming no
   unrelated dispatcher or Node definitions changed.
4. The owned quiet-process fixture: independent-session child, stale producer
   heartbeat, repeated exact-identity status reads with unchanged ledger,
   events/output checkpoint/progress, followed by owned-handle EOF cleanup and
   UNKNOWN capture after exit. Fixtures create nothing at module import;
   cleanup targets only their own Popen handle, never arbitrary PID groups.

The tests also cover malformed/legacy identities, foreign or reused identity,
permission/race failures, unknown times, terminal/CANCELING preservation,
failure-code warnings, summary/structured agreement and read immutability.
The isolated SQLite tests use the real manager `get()` against a deliberately
minimal table; they do not substitute for Root's production-schema regression
validation. No performance, safety or physical claim is made.
