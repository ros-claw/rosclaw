# N04: inactive bootstrap sensor observation

World 200913 retained its original FAIL: the declared laser had no received
messages in the captured three-second discovery window. Clock and independent
ground-truth messages were received. This does not establish absence of laser
messages throughout the whole bootstrap, or identify a rendering failure.
The existing shared Gazebo command already has `--headless-rendering`; adding
another headless option was a rejected diagnosis, not a fix.

The optional operator fixture declaration `discovery_duration_sec` now selects
a finite 1–60 second read-only observation window; the default remains 3 seconds.
The value participates in the frozen declaration hash and launch plan. It
does not change the immutable outer runtime deadline. Observation still starts
only after both inactive spawners and the strict controller-source probe succeed.
It does not activate controllers, admit a Body, dispatch tasks or modify any
source sensor, bridge, World or Nav2 configuration.

`generic_bootstrap_observation.py` reuses the existing ROS read-only probe in
the owned SDK child. Every second it retains an original snapshot and a sidecar
in `bootstrap-discovery.checkpoints/`, then atomically updates the latest
`bootstrap-discovery.json` and `bootstrap-discovery.observation.json`. The
sidecar records the snapshot SHA-256, actual monotonic start/deadline/recording
times, checkpoint count and `IN_PROGRESS` or `COMPLETE` window status.
Interrupted runs retain their earlier checkpoints. A sidecar/hash mismatch is
incomplete evidence. A complete window says only that its observation period
elapsed; it does not declare sensor freshness or physical acceptance.

Targeted tests use fake read-only observations and real filesystem checkpoints.
They cover immutable deadlines, interruption, snapshot failure, no overwriting
of existing evidence, source-path aliases and invalid durations. ROS SDK launch
construction and future actual inactive startup require separate evidence.

No new World run is authorized by this component alone. Future diagnostic
startup must have a separately frozen source/image/declaration/seed/deadline,
wait for the registered N02 physical evaluation to release its environment,
and retain original laser signals and any failures. Current laser root cause:
**UNKNOWN**. Held-out Body, controller activation and cleaning: **NOT_RUN**.


## Original stopped-container logs and diagnostic export

Read-only copies from World 200913's original stopped container recovered its
Ogre, server and SDFormat logs without restarting or executing the container.
The original acceptance remains FAIL. GLX display warnings preceded successful
EGL/Mesa OpenGL initialization. The server recorded the rendering thread ready
at 12:43:38.128 UTC and advertised `/declared_gz_lidar` at 12:43:38.129 UTC.
The original ROS observation ended at 12:43:40.726 UTC without laser messages.
Topic advertisement is not evidence of published samples or ROS delivery;
the missing delivery's root cause remains UNKNOWN. A permanent renderer
initialization failure is inconsistent with these recovered logs.

The owned bootstrap host now copies three fixed SDK diagnostic locations after
verified container teardown. Each copy rechecks the exact container ID, owner,
kind, pinned image and stopped state. It records command outcomes and hashes
of regular files and rejects copied symlinks. Missing files, copy timeouts or
incomplete exports remain diagnostic failures and never replace the original
bootstrap outcome or physical acceptance. It never restarts or execs a container.
This export can recover retained container-layer files; it cannot recover files
from a discarded tmpfs. A later actual startup must still test export integration.

## Rate-only synthetic source diagnostic, 2026-10-09

Worlds 200914 and 200915 retained their FAIL with complete 30-second ROS
observation windows. In 200915 a registered extra Gazebo subscriber timed out
after 12 seconds with zero stdout bytes; Gazebo advertised a LaserScan publisher
and the bridge subscriber. Topic advertisement alone was insufficient.

A separately registered development World 200917 changed exactly one original
synthetic operator input: the lidar declares `update_rate=20`. `always_on` stays
absent. Robot geometry, URDF, controller, Nav2, map, World and bridge outputs
remain byte-identical. This is an operator fixture correction, never an automatic
change to an unfamiliar robot or a third hardcoded Body profile. The synthetic
test-source builder now records that explicit rate too.

World 200917 passed its inactive manager and graph/sensor observation: all ten
Nav2 lifecycle nodes remained UNCONFIGURED; the strict inactive controller probe
passed; ROS received approximately 20.15 Hz LaserScan with an 8.20 ms last-message
age after the complete 30.01-second window. The registered Gazebo probe received
one original JSON LaserScan. The source/image/declaration and raw evidence are
recorded separately under recovery_2026-10-09 in the operator harness workspace.

This supports the sensor scheduling hypothesis in this synthetic fixture, but
does not prove a Gazebo implementation root cause. The additional subscriber
may affect lazy sensor activation; a no-extra-subscriber follow-up is required.
The captured scan has 360 ranges, all Infinity in this empty synthetic scene,
and a Gazebo-scoped frame. Useful obstacle returns and matching ROS scan/TF frame
identity are not yet verified. No controller activation, Body admission, held-out
integration or cleaning task was performed. Original failures are unchanged.

World 200918 repeated the same rate-only source with **no additional Gazebo
subscription or exec**. It passed another complete 30-second ROS observation
at approximately 20.00 Hz. The original ROS header frame is
`synthetic_source_model/range_frame/actual_source_lidar`, whereas the observed
static TF is `platform -> range_frame`. Historical discovery at the original
capture time correctly remains UNKNOWN, including `tf.base_to_lidar`; inactive
odometry and map are also unavailable. This is not Body admission.

The synthetic operator source additionally declares `gz_frame_id=range_frame`
to identify its actual URDF sensor link explicitly. Gazebo Sensors 8's
[Sensor.cc](https://raw.githubusercontent.com/gazebosim/gz-sensors/gz-sensors8/src/Sensor.cc)
describes that source extension; branch documentation is not proof of the
installed SDK's behavior. A separately registered inactive SDK World must
verify the resulting ROS header and TF using the actual pinned image. The
production source generator still preserves supplied unfamiliar source bytes;
it never repairs another robot's frame name by guessing.
