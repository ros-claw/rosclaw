# Additional SIM backend observation interlock

This is source preparation, not physical acceptance. No new Gazebo world,
DDS delivery, probe scene operation or Native episode has been verified here.
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
