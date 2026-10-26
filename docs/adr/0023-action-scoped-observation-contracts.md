# Action-scoped observation facts (local candidate, 2026-10-26)

Status: local candidate. No new runtime, Agent loop, safety authority or physical acceptance is introduced.

Compiled Body `provider_interfaces.ros_observation_contracts` binds a topic to its exact type, publisher, installed package version and, for conditional telemetry, the selected plugin parameter. Optionality is never inferred from a topic name or Boolean type. Native probe exposes installed package versions and existing current graph/parameters; this is installation evidence, not remote binary attestation.

Supported publication categories: periodic, command, static, event, conditional, optional. Required topics, explicit rates and critical sensor types cannot be downgraded. Missing/mismatched contracts retain conservative legacy behavior. Old auxiliary values remain UNKNOWN. Conditional contracts record execution phase from fresh Action status and lifecycle; lifecycle ACTIVE alone is not an executing goal. Independent controller failure diagnostics and native watchdogs remain active.

Full diagnostics retain all issues. Only Body-bound affected-capability scopes narrow readiness vetoes. Body, global hazards and the selected capability's observation requirements participate in its dependency fingerprint. Full audit context remains separately available.

Migration 040/041 adds optional per-capability hashes to existing context leases. Empty maps preserve legacy full-hash semantics. Authority-produced scopes are captured when issuing a lease; caller/model-provided scopes are not accepted. Identity, writer, mode, Body, revision, TTL, approval and exact grant checks remain in their existing authorities. Relevant facts are rechecked before dispatch; changes reject with NEEDS_REPLAN, without consuming a grant or replaying an action. The patch does not extend TTLs or add automatic action retries.

Qualification requires paired healthy/failure tests, clean installed-artifact checks, same-policy SDK/runtime physical controls, then complete H1 and held-out trials. Component regressions alone are not qualification. Critical controller/sensor stop and cancellation remain owned by existing daemon/native control, not the LLM.

A follow-up candidate separates transport/physical uncertainty: missing or lost terminal results remain RECEIPT_PENDING/UNKNOWN, exact owned transactions can be reconciled via pi.action.reconcile, and known terminal receipts cannot be overwritten by delayed nonterminal updates. The Native tool exposes approval_id as query-only reconciliation and marks UNKNOWN as an error through Pi's supported hook. No physical replay is triggered. Restart qualification so far includes the real SQLite journal/service with fixture action transport, not a physical daemon power-cycle.

Body `ros_action_observation_requirements` now also declares topics/types, TF pairs and lifecycle dependencies for motion, docking and navigation. This makes the capability-scoped lease path real for bound robot actions, rather than only for resolver-recognized navigation types. Unknown required publisher evidence stays UNKNOWN. Known dependency failures affect the declared capabilities; unbound hazards, TF cycles and global clock hazards remain conservative/global. The read-only probe accepts a bounded, operator-configured `--read-type` list so robot-specific status messages can be observed without hardcoding a robot vendor into the core allowlist. Missing measurements never become healthy merely because a type was enabled.

Canonical lease/owner-loss results include TIMED_OUT, ORPHANED, BLOCKED and DEGRADED. These are known non-success terminal outcomes, not permanently unknown results and never verified completion. Receipt indexing, ActionTxn transitions and Native error projection retain those distinctions. A call interrupted in DISPATCHING can reconcile a later exact terminal receipt without re-dispatch.

If the entire response was lost before any local receipt event, pi.action.reconcile may read the existing daemon action by its authority-persisted ID. It validates session/caller/transaction ownership first, checks the persisted ExactAction hashes and compares receipt action/Body/hash/mode/capability. No session, grant or action is created by this query. A running, unavailable, synthetic, unverified or mismatched result remains UNKNOWN. Legacy synchronous MCP without an authority lookup API remains UNKNOWN rather than inventing a query or replay.


Operator-owned probe transport may additionally require reliable delivery for
explicit readable message types with `--reliable-read-type`. This is independent
of topic names and never marks a signal healthy: actual publishers must offer
RELIABLE, otherwise observation stays unavailable with a contract diagnostic.
Default sensor subscriptions retain their original best-effort policy. The
opt-in keeps only the latest complete sample; received old frames still require
native header freshness checks at the motion authority. No simulator, sensor
resolution, safety timeout or action authorization is changed.


Explicit AgentService shutdown now closes its existing OperationManager before
closing the event store. Accepted spawns remain owned when an RPC requester
disconnects; shutdown waits for those spawns and uses the existing process-group
cancel/reap path, including a cancellation already in flight. New process
admission is rejected once shutdown starts. This revokes an explicitly closing
supervisor, not a wall-clock duration limit or a liveness sweep. Abrupt-crash
reattachment and the no-default-duration-kill policy remain unchanged. Recovered
PIDs without a locally owned subprocess handle are not signalled on close;
their local watchers are stopped before the journal closes. This is process
lifecycle evidence, not proof of physical ROS action cancellation or stopping.

### Candidate completeness and declared lifecycle (2026-10-26 follow-up)

All candidate paths, including typed cleaner bindings, merge the matching
compiled Body observation requirements before availability ranking. A different
interface candidate cannot omit a declared input. Missing facts remain UNKNOWN;
wrong types, stale required inputs and relevant blocking diagnostics block work.
The independent cleaner-off path retains its own minimal declared observations.

Lifecycle nodes in the existing action declaration are checked explicitly:
missing/UNKNOWN nodes do not become ACTIVE, and an ACTIVE observation must retain
the existing five-second evidence freshness bound. This closes a gap affecting
nonstandard Action bindings such as FollowPath; it adds no new health runtime.

The added regressions failed seven negative cases before the fix. They are
software contracts, not proof of a particular live context rejection or a
complete cleaning mission. That attribution still requires actual authority
snapshots and independent physical evidence.


### Driver-bound freshness

An action may declare `topic_freshness` for its typed dependencies, with a
positive finite `max_receive_age_ms` and optional `max_header_age_ms` (at most
60000 ms). The compiled Body must derive these from its actual executor
contract; this field grants no permission and does not change executor limits.
Malformed explicit policies block the action. Omitted policies preserve legacy
Signal limits. Explicit receive limits are strict (`age < limit`); native header
ages retain the executor's -100 ms future tolerance and inclusive upper bound.
Receive age advances in wall time; native header age is measured by Probe at
its ROS-clock snapshot and is never extrapolated from wall time. Missing header
evidence stays UNKNOWN. The executor rechecks current inputs before dispatch.
Generic diagnostics keep their original threshold and severity. Their action
scope uses the compiled dependency policy, which is displayed in Native context
and bound to the effective Body hash. A generic warning cannot silently override
a different, valid Driver contract or exempt a genuinely stale dependency.
