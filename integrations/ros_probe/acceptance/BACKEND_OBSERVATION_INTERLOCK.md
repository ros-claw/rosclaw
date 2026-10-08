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

The pinned Gazebo8.15.0 PosePublisher implementation initializes its publication
clock at0 and skips updates before a positive configured period. Its negative
`update_frequency` leaves period0 and publishes every unpaused step. A20Hz
probe contact producer which emits its first step immediately can otherwise
have a permanent timestamp phase offset from a20Hz pose producer. The new
instrument source therefore declares pose frequency-1, with contacts still20Hz.
This follows the pinned primary source:
https://github.com/gazebosim/gz-sim/blob/gz-sim8_8.15.0/src/systems/pose_publisher/PosePublisher.cc
Source/CDR/SDF checks are not an actual loaded-plugin cadence measurement.

`ProbeSceneGeometry` is a read-only additional spatial constraint. It requires
same-step original scene and native component timestamps, exact model and
complete collision-entity correspondence, actual primitive geometry, a body
envelope within the sealed bound, the declared centered instrument sphere,
measured isolated column, and actual robot/obstacle clearance. Source loss,
paused physics, changed identity/shape, gaps and unsafe clearance latch failure.
Normal articulated Body local poses may change inside the original bound;
this cannot silently change primitive identity/dimensions. New exclusive world
candidates request complete Body collision geometry, leaving original scenes
unchanged. Its20 contracts use explicitly derived synthetic source packets.
It remains unjoined to a controller and original-source closed replay and
declares world ownership, backend/physical admission and authority unverified.

`ProbeSceneJoin` buffers original scene, robot and instrument bytes independently
for at most 300 ms (8 scene / 64 native packets per role). It joins only the same
integer nanosecond SIM step; the spatial validator additionally checks original
SIM values, physical iteration, model/collision identities and poses. All six
cross-stream arrival orders are supported. Adjacent frames, interpolation,
sequence gaps and expired pending scenes fail closed. The original receipts are
preserved in the result; joining later never extends the earliest source TTL.
Robot packets between 20 Hz scene observations remain the separate all-step
contact gate's responsibility. This sampled spatial join cannot certify absence
of one-step contacts. Source ownership, controller IPC and the qualified full
launcher remain independent prerequisites; no scene service is executed here.

The optional spatial observer mode requires all of `--scene-binding`,
`--probe-declaration` and `--scene-directory`. It adds the independent
`/rosclaw_sim/physics_snapshot` subscription, binds both declarations into a new
constraint policy hash, and requires spatial readiness as well as the contact
constraint. `closed_backend_observation` accepts the same two frozen declarations
and replays original scene bytes and joins with the identical online engine.
A legacy actor constraint file cannot enable this mode. The owned policy files
are reopened as regular files; ambiguous/nonfinite/non-UTF8 JSON is refused.
None of these sources certifies world ownership or creates a scene-service ACK.

The isolated DDS driver accepts `--with-spatial`. It uses explicitly synthetic
SDK-derived sources at robot 100 Hz / scene and probe 20 Hz, positive physics
iterations and a fixed 0.5 s transport setup interval before starting sequences.
It requires all 80 original exact scene joins and checks that the first rejected
original robot frame is the intentional sequence regression at iteration 401.
Discovery alone does not prove a data connection. This contract starts no
Gazebo world, scene service, actuator or Native mission.

`prepare_backend_world` assembles both independent `PassiveContacts` instances,
explicitly typed bridges, source-pinned library copies, all-step robot / sampled
probe policies and the combined spatial observer config in an exclusive bundle.
It preserves original inputs. It includes a separately declared passive pose
bridge, expands `topic_name` into identical declared ROS/Gazebo names, and refuses
control/service bridge roles, ambiguous names and missing types. It does not
infer namespaces or live endpoints. The final bundle manifest hashes every
final source; the earlier probe-world manifest is only its preparation substage.

For the exclusive candidate world, conflicting visual names can be renamed
without changing any nonvisual bytes. Collision names remain original; ambiguous
visual frame references are unsupported. This source repair is not runtime
physical equivalence or admission. The known vendor source contract preserves
the full robot's custom `ros2_control` extension and checks its full SDF parse,
as well as a clearly labelled standard-SDF-only schema projection. It never
renames command/state interfaces to satisfy a generic uniqueness check.

The additional compiled `backend_source_parser` invokes installed
`hardware_interface::parse_control_resources_from_urdf` and
`ros_gz_bridge::readFromYamlString` without creating a Node or Hardware instance.
It checks the known differential-drive interfaces and exact bridge type/name/
direction/queue/QoS correspondence after source normalization. Parsing source
is not loading a controller, a Gazebo world, a system plugin or a Native task.
## Original instrument transaction IPC

## Owned world source and original instrument wire

`WorldSourceOwner` is a read-only Linux source checker. The final bundle must
reopen every manifest hash, the actual owned process PID/UID/starttime must
match its Gazebo launch arguments and private `GZ_PARTITION`, and that same
process must map the exact pinned physics/contact ELF inodes plus the installed
Gazebo 8.15 core and scene command system. Source file mutation or loss latches
a fault. Procfs size zero is handled with a bounded read. Unrelated unlinked
shared memory is ignored; it cannot match a required ELF. This checker starts
no World and does not itself admit backend health or authorize movement.

`backend_instrument_service/owned_instrument_service` has an explicit
`--source-validate-only` mode that constructs no transport Node. Its separate
`--owned-runtime` mode is for a future qualified owned SIM fixture controller:
it requires the exact private partition, discovers only the declared world's
set-pose service and can target only the declared disjoint instrument and exact
lift coordinates. It cannot accept a runtime-injected service reply. Requests
use the installed SDK's `RequestRaw` API with a 100ms RPC timeout; retained
request/reply wire bytes are decoded with that SDK without replacing originals
by a reserialized representation. Source-only parsing can never arm a lift.

`instrument_service_evidence.py` retains the original source record and can
reparse its wire bytes using the same frozen executable in source-only mode.
This proves byte/projection correspondence, not publisher authentication,
actual probe motion, cache clearance, World ownership or physical acceptance.
The compiled executable, its source and the original failed normalization
attempt are retained with the source contract evidence.

`backend_observer.py` optionally accepts both `--controller-pid` and
`--controller-uid`, only with the complete spatial source mode. The owned
observer directory must be mode 0700; its exclusive `probe-controller.sock`
is a mode 0600 Linux `SOCK_SEQPACKET` socket. Kernel PID/UID and process start
time pin one live controller. This is local process correspondence, not
publisher authentication or independent Gazebo world admission.

The controller must first send an original exact `backend_probe_lift_begin`
request. Fresh stable ground, current complete robot contact observation and
the exact-step spatial constraint are required. The observer retains original
IPC bytes, their hashes and the kernel peer identity before returning a source
event acknowledgement with `authorization=false` and world admission false.
Only a separately qualified owned world controller may execute the declared
instrument scene service; this observer never executes it.

An accepted begin retains original probe components for at most 0.2 seconds
while original independent poses continue to arrive. A matching original
successful reply resumes those original frames without altering their clocks.
The service completion timestamp is distinct from observer receipt. Frames
captured before completion cannot prove the required post-completion measured
lift. Clear cache and stable ground recontact remain mandatory; a successful
reply alone does not complete a cycle. Missing, late, mismatched or failed
replies latch a fault, as do incomplete original source or IPC audit records.
A closed replay rejects any unfinished transaction, even after an earlier
successful cycle, and reparses original IPC bytes to check retained projections.

`backend_observer_dds_contract.py --with-spatial --with-controller-ipc` tests
the actual isolated observer process, DDS transport and private IPC against
explicit synthetic SDK-derived packets and an explicitly synthetic service
reply. It starts no Gazebo world, scene service, robot actuator or Native task.
It cannot establish physical cache behavior or admit a runtime world.
