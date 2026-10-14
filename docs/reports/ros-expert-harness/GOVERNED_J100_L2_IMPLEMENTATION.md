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
World bytes every cycle. ROS/source age is at most 500 ms. The ROS consumer uses
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
costmap update/publication 5 Hz, commands at 80% of Body bounds and Nav2 goal
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

Two fresh pilot Worlds on source `43d8b0a0` completed both canonical position and
yaw goals and passed independent raw review:

| Seed | Initial x/y/yaw | Health rows | ECM samples | Original contact pairs | Unexpected contacts |
| --- | --- | ---: | ---: | ---: | ---: |
| 101 | 0 / 0 / 0 | 282 | 192 | 7472 | 0 |
| 202 | 0.12 / -0.12 / 0.2 | 275 | 209 | 8132 | 0 |

Independent review recomputes initial G1/L1, current ROS health, source continuity,
original contact classification, canonical goal errors, matching raw physics,
lease-closed stopped samples and exported public artifact SHAs. Worker terminal
rows in these pilots lack their own clock; original sink terminal timestamps
independently occur after explicit owner shutdown. That evidence limitation is
recorded, and subsequent worker rows include their own monotonic timestamps.

Earlier development failures remain intact. They exposed fixture output collisions,
bootstrap delay, premature proof expiry, synchronous callback overload, background
observation lag, stopped localization publication, concurrent pointer publication
and atomic inode replacement. A partial motion or a FAILED receipt never counts
as completed L2. Seed 303 exposed the atomic-read race and requires a fresh World
on the corrected source. All three seeds must be verified again on final source
and actual merged main before closing normal acceptance.

Current software verification: 1016 ROS tests passed, 10 deselected (one existing
deprecation warning); the prescribed mypy scope passed 121 files. Changed SDK
contracts and formatting pass. Whole-repository static checks retain existing
baseline failures and are not represented as green.

Still required for G3: the final three-seed matrix, all ten exact execution fault
Worlds with original actors and independent fail-stop/old-lease rejection,
actual contact-monitor sensitivity, sealed evidence and normal source/main PR
integration. G4 requires five real engineering repair/revalidation/rollback
fixtures. G5 requires 54 actual same-model A/B/C episodes with original usage
and causal memory traces. Those gates remain unaccepted; `v1_done=false`.
