# Additional SIM backend observation interlock

This is source preparation, not physical acceptance. No new Gazebo world, probe scene operation or Native episode has been verified here.
Isolated actual DDS transport has now been checked over explicitly synthetic
SDK-derived packets, separately from physical source admission.
The frozen N02 experiment source, image and launcher remain unchanged.

The new opt-in `require_backend_observation=true` actor mode requires the exact
owned `/evidence/backend_actor_constraint.json` run/Body/policy binding. Before
an independent qualified observation arrives, heartbeat, cleaner activation,
hold release and navigation forwarding remain blocked. The first retained
sequence must be zero; subsequent sequences must be contiguous, with strictly
advancing original monotonic time and age at most 150 ms. Ready observations
require zero actual robot contacts, at least one measured probe cache cycle
and aligned robot/probe/actor SIM clocks. Invalid bytes, repeated sequence,
missed sequence, stale source, Body/run/policy mismatch, contact or observation
loss after admission latch failure, revoke the lease, publish zero velocity and
publish cleaning OFF. Later good messages cannot erase failure. The daemon
lease remains separately necessary. Observations grant no robot permission.

The actor retains original observation UTF-8 bytes, SHA256, original receipt,
interlock samples, lease changes and source faults in a separate closed audit.
Dropped or failed actor observation audit data also closes the interlock.
This proves source correspondence only, not DDS publisher authentication,
actual velocity delivery or independently measured physical stopping.

`BackendObserverReplay` applies retained original robot TF CDR/contact JSON,
instrument TF CDR/contact JSON/lift request/reply bytes and sample events using
the same logical source trackers online and offline. A rejected source remains
latched. `backend_observer.py --prepare-only` prepares an exclusive actor
configuration without importing rclpy or creating a Node. Its observer entry
exposes four observation subscriptions and one observation publisher, with
parameter metadata publishing disabled and no external services. Original
source bytes and projections are retained; source files/plugin hashes are
reopened periodically. It waits for an actor subscription before issuing the
first envelope, preserving sequence zero.

The instrument controller and its original lift transaction IPC are still
unjoined. The observer does not synthesize an ACK or execute a scene service;
therefore it cannot qualify an actual probe cycle through this entry yet.
Actual whole-world/Body geometry admission, a source-bound owned instrument
controller, independent all-step DDS delivery, closed joined replay and Native
canonical/Practice/independent-stop acceptance remain mandatory work before
this becomes the required P1 execution path. Logical readiness and a recorded
zero command do not certify the backend or physical stopping.

Current contracts use endpoint recorders and explicit pose decoder doubles;
these do not start ROS/DDS/physics. Actual installed ROS CDR source contracts
for the underlying trackers are separate and retain their synthetic labels.

`closed_backend_observation.py` reopens the robot/probe prepared policies and
plugin bytes before and after replay, requires a closed lossless hash-chain
writer, and recomputes every original source projection with the online engine.
It rejects substituted Body/policy/source identities, modified byte hashes,
missing/reordered events, clock or SIM mismatch, fabricated projections,
partial rows, duplicate/nonfinite JSON, unknown events and incomplete closure.
A fresh qualified logical final constraint is required. This result continues
to declare actual world/Body admission, physical stopping and task acceptance
unverified. Its 13 contracts use explicitly synthetic tapes and decoder/source
loader doubles; they are not additional physical episodes.

`backend_observer_dds_contract.py` runs actual rclpy Nodes and an independent
observer process in an isolated `--network none` container/domain231. The
final contract sends robot TF/components at100Hz and probe TF/components at20Hz.
Before injection, no source fault or ready/authorization claim is allowed. The
retained valid robot prefix must reach sequence399; the first rejected source
must be the intentional sequence0/iteration400 packet. Later valid messages
cannot clear failure. The actual observer exits0 onSIGINT and closes1144 events
without drops or writer error. Four seconds of synthetic DDS traffic is not a
whole-mission DDS qualification, real Gazebo contact/cache qualification,
physical stop proof or Native task. No actuator or scene service starts.
The earlier20Hz and100Hz transport passes remain retained; the final test adds
stronger source-prefix/fault-origin assertions and an original driver copy.
