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
