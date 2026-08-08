# AgileX LIMO ROS 1 Robot Pack

This Pack binds the `limo` e-URDF Body to the independently versioned
`ros-claw/limo-ros-mcp` adapter at commit
`0c7cf9a92510dc8867faf404a946d8131ba956ff` (MCP 0.10.3).

The first REAL capability is `limo.set_initial_pose`. The Agent submits a
validated map-frame estimate to `rosclawd`; the daemon validates an exact
operator permit and starts a fixed-operation ROS Melodic worker. The Agent and
MCP server do not import rospy or publish `/initialpose`.

Revision 0.1.6 adds `limo.navigate_to_pose` and `limo.play_tone`. Both use
in-context confirmation with opaque, exact-action permits. The daemon launches
fixed Python 2 workers from the locked MCP revision.

Revision 0.1.7 adapts those workers to the live LIMO host. Navigation treats
stationary, event-driven AMCL message age as a warning while retaining blocking
covariance and live `map -> odom -> base_link` TF checks. It subscribes to AMCL
before dispatch and requires a post-dispatch pose, move_base `SUCCEEDED`, final
AMCL tolerance, and stopped odometry. A warning-only DEGRADED snapshot remains
usable, but any BLOCK check still prevents dispatch. Tone playback accepts only
440/660/880 Hz, 0.2–1.0 seconds, and 5–25% volume. It uses the uniquely
allowlisted USB PulseAudio sink when PulseAudio owns the sound card, falls back
to direct ALSA otherwise, restores the previous mixer state, and reports only
`DRIVER_CONFIRMED` evidence unless a human separately confirms hearing it.

Revision 0.1.8 fixes completion evidence for the live event-driven AMCL node.
Navigation still prefers a post-dispatch `/amcl_pose`, but when AMCL emits no
new event after `move_base` succeeds it verifies the final map-frame pose from
the live `map -> base_link` TF instead. The same goal tolerances and stopped
odometry checks remain mandatory, and the receipt records which evidence source
was used.

Revision 0.1.9 incorporates the first independent physical-observer feedback.
The USB PulseAudio sink was present but held at 9% / -63 dB, while the old tone
worker attenuated the waveform again. The new worker temporarily uses a reference
output gain, applies the requested volume exactly once in PCM, and restores the
original PulseAudio or ALSA state. Navigation now records the active local-planner
tolerance and pre-dispatch goal error. A `move_base` success only counts as observed
movement when odometry crosses a bounded displacement threshold; a goal already
inside planner tolerance is reported explicitly as requiring no movement.

Revision 0.1.10 requires a bounded onboard-microphone loopback for every REAL
tone. The worker measures a pre-playback baseline, captures during playback,
and verifies the requested frequency against fixed level, gain, and adjacent-band
prominence thresholds. Raw PCM is discarded and a missing acoustic observation
fails the canonical receipt instead of reporting driver-only success.

Revision 0.1.11 locks the current MCP release after the interaction/runtime
deployment, readiness, TF, and camera-summary fixes. The daemon now rejects a
configured adapter whose recorded repository commit differs from this exact
revision, preventing an older installed MCP snapshot from being mistaken for
the signed executor source.

Revision 0.1.12 makes each signed capability input describe the exact
`ActionEnvelope.arguments` accepted by rosclawd. This lets the Native Agent
copy the tone, navigation, and initial-pose contracts without guessing adapter
field names or placing the body snapshot hash inside the arguments object.

Revision 0.1.13 rejects a goal that is already inside move_base's active
tolerance but outside the operator-approved verification tolerance. It also
ignores late delivery of stale AMCL messages and falls back to the live
`map -> base_link` transform for final-pose verification.

Revision 0.1.14 adds `limo.speak_text`. The fixed Python 2 worker calls the
locally installed eSpeak-NG library directly, synthesizes only the approved
1–80 character Chinese or English message into memory, normalizes it to a
bounded 10–25% PCM peak, and uses the allowlisted USB speaker path. The receipt
binds the approved text SHA-256 to synthesis and requires microphone RMS energy
above fixed level and baseline-gain thresholds plus mixer restoration. It does
not claim that the microphone recognized the linguistic content.

Revision 0.1.15 hardens live patrol readiness on the Jetson. LaserScan evidence
is collected through the bounded local ROS CLI instead of competing with the
parallel rosbridge snapshot, and the fixed `map -> base_link` TF probe gets a
five-second listener-fill window. This removes transient false blockers while
retaining the same scan freshness, valid-ratio, clearance, and TF predicates.

Revision 0.1.16 seals the LaserScan observation after the bounded TF preflight,
alongside the high-rate rosbridge observations. This keeps the scan inside its
two-second freshness limit and the coherent snapshot window without weakening
either readiness threshold.

Revision 0.1.17 orders readiness collection by latency: fixed TF listener fill,
then LaserScan, then the final costmap and high-rate rosbridge window. The
snapshot therefore stays coherent even when starting a ROS CLI subscriber is
slow on the Jetson.

Revision 0.1.18 bounds the dynamic `/tf` topic summary to five messages. The
separate fixed `tf_echo map base_link` probe remains the authoritative chain
predicate, so the shorter topic sample removes a redundant five-second wait
without reducing TF verification.

Revision 0.1.19 sequences the remaining slow global-costmap and LaserScan CLI
reads after TF preflight and before the final high-rate rosbridge window. This
prevents slow metadata transport from aging otherwise live status and odometry
at snapshot closure.

Revision 0.1.20 replaces ROS CLI YAML serialization for the freshness-critical
global costmap and LaserScan with a fixed-topic, read-only ROS Melodic worker.
It returns only bounded metadata and scan ranges, preserving live header times
without exposing a generic ROS command or publication surface.

Revision 0.1.21 starts the two fixed ROS Melodic samplers concurrently after TF
listener preflight. This removes serial ROS-node registration skew before the
final high-rate rosbridge evidence window is sealed.

Revision 0.1.22 schedules the tighter-budget LaserScan sampler alongside the
final high-rate rosbridge reads, while the global costmap remains the bounded
preflight sample. This seals the laser header near snapshot closure without
changing the two-second freshness threshold.

Revision 0.1.23 multiplexes all final high-rate observations over one bounded
rosbridge websocket after the parallel fixed costmap and LaserScan workers.
Per-topic receipt times are preserved while connection fanout no longer widens
the evidence window on the Jetson.

Revision 0.1.24 includes the bounded 450-sample LaserScan in the same final
rosbridge batch. Only the multi-megabyte global costmap uses the compact ROS
Melodic worker, eliminating the last process-startup delay from laser freshness.

Revision 0.1.25 keeps LaserScan on the fixed ROS Melodic worker because its
non-finite no-return values are not reliable through rosbridge JSON. The laser
freshness budget is calibrated to 2.5 seconds from the documented 0.7-second
driver stamp lag plus roughly 1.7 seconds of bounded Jetson collection time;
the navigation executor still samples live clearance immediately before dispatch.

Revision 0.1.28 locks MCP 0.10.0 after live chassis diagnostics exposed a
stalled-navigation failure mode. The navigation worker now polls the action
result, cancels the move_base goal when it receives SIGTERM or SIGINT, and
fails within three seconds when a sustained non-zero velocity command produces
no odometry response. Cancellation verifies stopped odometry before the worker
exits, preventing an expired daemon lease from leaving a live navigation goal.

Revision 0.1.29 locks the cold-boot patrol and person-interaction fixes verified
on the physical LIMO. Navigation completion is checked against the stopped live
`map -> base_link` TF pose, with AMCL retained as auxiliary evidence, so a stale
mid-motion AMCL sample cannot create a false failure. The adapter adds bounded
robot-pose and Dabai frame workers for ROS Melodic, fixes operator-runtime
readiness, and records person responses through the fixed PulseAudio bridge while
draining the capture continuously and rejecting truncated audio.

Revision 0.1.30 makes the formal installed MCP entrypoint expose its complete
35-tool surface by default. This aligns the signed adapter declaration with the
runtime wrapper generated by ROSClaw, so camera, audio, robot-pose, peripheral,
and runtime diagnostic tools remain available after installation. Constrained
clients can still explicitly select the bounded 11-tool `core` profile.

Revision 0.1.31 adds a repository-root `limo-ros-mcp` launcher to the locked
adapter source. The launcher bootstraps the source tree and starts the MCP
server with the selected Python interpreter, so ROSClaw's isolated source
installer can execute the signed checkout without first mutating its managed
Python environment to install a console-script shim.

Revision 0.1.32 makes read-only audio inspection and bounded microphone-level
measurement use the exact allowlisted PulseAudio bridge. This lets the isolated
MCP account observe the USB speaker/microphone without direct ALSA control-device
access; captured PCM stays in memory, is truncated to the requested interval,
and is discarded immediately after level analysis.

H1 means only that the signed contract and executor tests pass. H4 requires a
real LIMO, an AMCL subscriber, a post-navigation map-frame pose from AMCL or the
live TF chain, a `map -> odom` transform, a canonical TASK_VERIFIED receipt,
odometry displacement evidence when movement was expected, and independent review.

Revision 0.1.2 waits for a converged AMCL sample within the bounded
verification window, so an immediate transient `/amcl_pose` sample does not
produce a false-negative receipt.

Revision 0.1.3 adds in-context MCP operator confirmation for REAL initial-pose
requests and allows five seconds for read-only ROS CLI startup on LIMO ARM
hosts. Clients without MCP form elicitation now fail immediately instead of
waiting for a confirmation response they cannot render. Permit material remains
internal to the trusted ROSClaw host boundary.

Revision 0.1.4 matches the Dabai U3 streams launched by
`astra_camera/dabai_u3.launch`: `/camera/color/image_raw`,
`/camera/depth/image_raw`, and `/camera/depth/points`. The MCP returns bounded
metadata summaries and never exposes the raw image or point-cloud arrays.
ROS CLI array placeholders are decoded into their true bounded byte and field
counts, rather than reporting the placeholder string length.

Revision 0.1.5 locks the adapter revision that expands read-only inspection to
29 MCP tools and 27 ROS observations. It adds bounded Dabai device and
color/depth/IR camera-state summaries, host audio playback/capture inventory,
in-memory microphone level measurement, display/touch inventory, USB peripheral
inventory, and platform health. Microphone samples are discarded immediately:
no recording or raw audio content is retained or returned. IR endpoints may be
present but are reported inactive when the driver publishes no frames. Front
OLED and chassis RGB lights remain declared but unbound until a stable host or
ROS interface is available.
