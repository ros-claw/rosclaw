# Backend cache probe evidence

Status: source preparation and offline contracts. Actual physics, runtime
backend admission, whole-world integration and Native acceptance are **NOT_RUN**.

A fresh PostUpdate contact-cache packet and a resting support contact do not
prove that Physics continues updating the cache. The proposed fixture adds one
explicitly declared dynamic sphere instrument outside the work polygon plus
conservative robot clearance. It uses a separate model, ROS namespace and
`/rosclaw_sim/backend_probe_components` observation producer. Its source hash
binds an instrument artifact; it cannot become a robot Body fact or permission.
No third robot asset is selected or adapted.

`backend_probe_fixture.py` writes an exclusive instrument SDF, bridge and source
bindings. It requires an operator-approved SIM declaration, bounded geometry,
three distinct observation endpoints, and separation from the active robot.
The declared maximum robot radius still needs correspondence with admitted
actual Body collision geometry before world integration. SDF validation proves
schema validity, not actual support, gravity, contacts or placement safety.

`backend_probe_evidence.py` requires measured stable ground contact, a fresh
original lift acknowledgement, measured actual lift pose, a stable empty
initialized contact cache while airborne, then measured ground recontact.
The lift acknowledgement only arms the state machine. Contact at inconsistent
height, non-ground contact, source loss, changed inventory, repeated sequence,
paused packets or missed SIM/wall refresh deadlines latch rejection. Readiness
is withdrawn when measured ground contact disappears. No empty ROS message is
inferred from silence and no interpolation fills missing pose observations.

`probe_lift_evidence.py` derives acknowledgements from retained exact instrument
request bytes and a bounded successful Boolean reply. It refuses an active
robot target. Its service specification is unexecuted argv with explicit world
identity and bounded timeout; it grants no authority. A future fixture-owned
controller must admit whole-world source ownership, actual geometry clearance
and live measured readiness before any scene service. This module invokes no
service and provides no robot command path.

`closed_backend_probe.py` replays original official TFMessage CDR, original
component JSON, and retained request/reply bytes. It checks source bindings,
projection correspondence, source timestamps, genesis/sequence/hash chain,
writer closure and unchanged prepared source files. Components arriving before
pose remain bounded pending originals until an exact timestamp match; missing
pose expires, and source rejection cannot be skipped. No diagnostic projection
can substitute for original evidence. Source hashes establish correspondence,
not authenticated DDS publishers.

`backend_probe_cdr_contract.py` uses the installed official ROS serialization
library with explicitly synthetic native JSON frames derived from SDK fixtures.
Twelve cases exercise valid replay and corrupted CDR, frames, original hashes,
projections, robot-target requests, failed replies, UNIX receipt time, run
binding, SIM time, writer closure and incomplete final lines. It instantiates
no Node, starts no DDS/Gazebo server and executes no service. Actual plugin ELF
bytes are reopened as source-bound files but are not loaded into a world.

A successful replay reports `backend_health_admitted=false`,
`physical_acceptance=NOT_VERIFIED`, and `authorization=false`. Actual intervention
qualification must still be joined with the owned runtime, robot contact source,
brush actor, mission accounting and closed canonical Native acceptance. These
contracts cannot be counted as D1–D6 physical episodes or unseen-Body admission.
