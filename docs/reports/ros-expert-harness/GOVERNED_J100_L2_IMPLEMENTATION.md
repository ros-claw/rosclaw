# Governed generic SIM navigation: implementation and remaining acceptance

P03 merged normally in PR #678 at
`f361423b040312777676f94ce81b92a21bc7ddf8`. Three source and four main original
workflows passed. Twelve fresh actual-main passive Worlds passed independent
raw recomputation and pinned read-only CLI inspection. P03 grants no motion.

The next implementation retains canonical MCP `request_action` → the independent
existing `rosclawd`, with its Session, Lease, Permit, executor and Receipt checks.
It adds daemon-configured supplementary supervision before admission, lease
operations, dispatch, execution polling and completion. Loss permanently latches
stopping; lease revocation and cancellation remain available. There is no REAL
executor, Agent ROS transport, Agent Runtime or MCP actuator primitive.

The SDK proof producer binds immutable Body/hash/workspace manifest, actual World
SHA, run, source epoch, daemon generation and action. Authenticated observations
expire within 250 ms. Monotonic sequence/cursor checks reject rollback, tampering,
replay and identity changes. Private fixture keys remain outside public evidence.
Proofs supplement the canonical governance chain; they are not permits.

A separate SDK ROS sink starts without a drive publisher. Explicit authenticated
operator admission and a current canonical lease are both required before it
can forward commands. Its final gate checks finite planar commands and unchanged
Body limits, fresh authenticated proof and independently observed ROS/ECM facts.
Lease expiry is checked before and after collection. Synchronizing an independently
observed source cursor waits at most 100 ms and never extends proof or lease expiry.
The original four-wheel J100 stamped controller retains its 200 ms watchdog.

Current facts come from original SHA-bound Native bytes and the pinned bounded
ECM stream. The collector checks immutable enrollment, Body workspace, URDF and
World bytes every cycle. Execution policy bounds snapshot/source age to 500 ms.
An independent raw review found that earlier pilots did not apply that bound
to every live stream: the candidate check used capture-time message age and
separate fixed thresholds. The current implementation additionally checks
odometry, lidar, localization and both costmap streams using original message
age plus elapsed time, with the execution policy and conservative margin.
Lifecycle and parameter observations use the same policy. Future observation
timestamps are checked against actual review time before projecting age.
Earlier accepted local pilots remain evidence for their original source and
policy implementation; they do not establish this corrected strict-age gate.
Fresh final-source and actual-main runs are required. The ROS consumer uses
an SDK-owned pure observation subprocess, keeping full model/Body review outside
ROS control callbacks. Its daemon observer independently checks actual socket
peer PID/UID, generation, watchdog, durable ledger, E-stop and SIM executor scope,
with an independent 250 ms deadline. No observer can renew a lease or dispatch an
action. Exit, expiry, failed observation and clock rollback latch permanently.

A conservative age margin of up to 100 ms is evaluated once on the original
facts. Actual review time is distinct from projected age-evaluation time; this
projects age only, not future lifecycle or physical state. Consumers check actual
monotonic deadlines. Atomic snapshot readers distinguish an unlinked immutable
old inode from in-place mutation; immutable assets retain strict ctime checking,
and mutable snapshots retain size/mtime, bounded regular-file and no-follow checks.
Native freshness clocks are sampled after actual pointer reads, because a new
original pointer can legitimately be published during collection.

Prepared source bringup uses pinned declarations for namespaces, frames, scanner,
controllers and routes. J100 remains an explicitly adapted official four-wheel
robot, not zero-shot onboarding. Staging changes are declared before World launch:
costmap update/publication 10 Hz, commands at 80% of Body bounds and Nav2 goal
checking at half the Body tolerance. An explicitly pinned SDK operator helper
calls AMCL `request_nomotion_update` at 10 Hz so actual laser/odometry filtering
continues while stopped. It publishes no pose, TF, velocity or goal, and does not
reinitialize localization. Original URDF geometry/inertias/joints and all eight
compiled Body limits remain unchanged.

The isolated ARM64 SDK image contains 33 exact official-SHA-checked dependency
wheels, installed offline without host package changes:
`sha256:d3a808fc091bec017657a2706cb9723799a5ea56f5e9f997663e4dc884cd78b5`.
The original P03 base layers remain intact. Canonical MCP tools import and execute
inside the SDK; there are no Runtime or MCP stubs in the physical proof path.

Three strict-age seed Worlds on source `18625857` completed all six canonical
position/yaw goals and passed independent raw recomputation. Earlier source
`50c36d15` also has a sealed three-seed functional matrix; it does not establish
the corrected cumulative stream-age checks. All eight Body bounds remain fixed.

All ten exact execution faults now have independent local original evidence:
`collision_monitor_disabled`, `lidar_stream_lost`, `out_of_bounds_goal`,
`entity_replaced`, `wrong_body_hash`, `wrong_run_id`, `daemon_sigkill`,
`lease_loss`, `emergency_stop`, `rosbridge_disconnect`. These developmental runs
span different source commits and do not replace the final uniform matrix.

Before-dispatch invalid Body hashes and out-of-bounds goals produce original
BLOCKED receipts with zero admission, lease updates and forwarded commands.
The wrong-run trial changes only the SDK-signed operator context run; the
canonical request stays valid and unchanged. Authentication succeeds, identity
validation rejects it before admission, and the original action remains FAILED.
No Agent reads the private signing key or signs an execution permit.

During-motion trials use actual independent motion to trigger SDK-owned fault
actors. Scan and rosbridge faults affect only declared owned components;
collision-monitor lifecycle deactivation uses the observed lifecycle services.
After independent stop, restored components cannot revive the old action or
lease. SIGKILL honestly lacks a terminal daemon receipt. Other original FAILED,
CANCELLED or TIMED_OUT receipts retain their actual verification fields; separate
physical stop evidence never promotes a failed action to COMPLETED.

Entity replacement removes the original entity and inserts the exact original
model XML. Original ECM must establish old ID absence and new ID presence.
The source guard stays permanently latched. The existing robot_state_publisher
sets its byte-identical original description to initialize the replacement's
controller; there is no new publisher or fabricated TF. Expired loading records
remain UNKNOWN. Independent review recomputes replacement stationarity directly
from ECM world poses indexed by the new entity ID, never the removed body's speed.
Source `325ac3ca` established 1.995 seconds of replacement stationarity and actual
old-action/lease rejection, with the original FAILED receipt preserved.

An independent contact-sensitivity World on source `040ca8b3` inserts one static
SDK-owned test collider at the observed original chassis surface, then removes
it. Raw records establish a zero/positive/zero sequence: 0, 35, 0 nonfloor pairs
in independently reviewed windows of at least 300 ms. Across the whole World,
57 actual nonfloor pairs are retained. Original robot ID remains unchanged,
with zero goals and zero forwarded commands. Inventory drift intentionally
latches source supervision; this instrumentation trial does not isolate a
collision-only stop cause. Cumulative contact episode counts alone proved
insufficient: the corrected fixture also requires fresh, complete original
contact messages without an active nonfloor pair. The earlier failed attempt
and its original claimed local outcome remain preserved.

Current software verification includes 1,143 ROS tests after contact clearance
repair, with 11 deselected and one existing dependency deprecation warning.
The prescribed mypy scope passes 121 files. Changed SDK contracts and formatting
pass. Whole-repository Ruff and formatting retain existing baseline failures.

Still required for G3: one final-source three-seed matrix and all ten exact
fault Worlds, contact sensitivity, sealed public evidence, normal source/main
CI and PR integration, and fresh actual-main physical revalidation. G4 requires
five real engineering repair/revalidation/rollback episodes. G5 requires 54
actual same-model A/B/C episodes with original usage and causal memory traces.
Those gates remain unaccepted; actual model experiment calls = 0 and
`v1_done=false`.

The first final-source matrix stopped before action dispatch because the initial
30-second ROS readiness window included controller subprocess bringup. The SDK
now records an owned, run-bound process-spawn milestone; capture waits for that
bounded preparation stage before starting its unchanged ACTIVE-lifecycle and
complete-physics window. The milestone grants no readiness or execution authority.
Execution freshness and every Body limit remain unchanged. The original timeout
and incomplete matrix remain retained.

A subsequent seed-202 uniform-matrix World failed safely before forwarding any
command: initial lease acknowledgement left 106 ms of its original proof-bound
lifetime, while the delayed first heartbeat plus guarded checks reached renewal
35 ms after expiry. Guarded navigation now performs its first heartbeat without
the initial sleep and requires an actual successful first renewal before goal
dispatch. Expiry, proof lifetime and old-lease rejection remain unchanged. A
failed first renewal dispatches no goal; both acknowledgement cases are tested.
The original failed World and incomplete matrix remain preserved.
