# AgileX LIMO ROS 1 Robot Pack

This Pack binds the `limo` e-URDF Body to the independently versioned
`ros-claw/limo-ros-mcp` adapter at commit
`781b0d873bbb2bfe36eb91b907ea15d4808cde3f` (MCP 0.8.7).

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
