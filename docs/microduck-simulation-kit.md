# Microduck simulation kit (candidate integration)

This extension registers the 14-DoF Pollen Robotics Microduck as a SIMULATION-only body. It connects ROSClaw's native action channel to the external `microduck-lab` MCP executor. It does not implement hardware control, infer calibration, promote policies, or grant permission for real execution.

The body profile records the native MJCF joint limits, 15 link masses (0.73724318 kg total), actuator torque bounds and asset digest. Four instances are registered with `body create --robot microduck --name microduck-{lavender,cream,sky,graphite}`. The leader body is `microduck-lavender`; a single action operates a shared simulated arena with four independent motor runtimes, and returns references to all participating profiles.

Install `microduck-lab[rosclaw]` into the ROSClaw environment, and prepare its assets and its own MuJoCo 3.12.0 environment. Set `MICRODUCK_ROOT` to the asset directory and `MICRODUCK_PYTHON` to that environment's Python. These are explicit kit environment references. Build the native agent from this checkout before using `rosclaw chat --workspace <microduck-repository> --mode SIMULATION --basic`.

`microduck.get_game_status` is an observation. `microduck.start_game` is an action, routed through the existing SimActionChannel and its existing DEV_SIM_ONLY admission policy. The manifest adds its own typed observation schema, timeout and verifier note; the UR5E output schema remains unchanged. A game mutates simulation and records evidence, so it must not be recategorized as an observation or dispatched through shell.

The adapter launches a bounded external worker with argv, never shell. Its receipt includes the actual run ID, body profile digests, audit path/digest, outcome and contact quality. Native ROSClaw retains mission, task, grant and receipt linkage. The external worker's archived controls, full-rate states, contacts and decisions can be regenerated with `scripts/replay_duckverse.py`; this demonstrates deterministic simulator consistency, not independent hardware validation.

A 600-second timeout accommodates offline full-rate evidence recording. This game is not claimed to run in real time. Motor control uses existing stand/walk ONNX policies at 50 Hz; the tactical controller uses public current warnings and ideal simulator state. There is no new Jev training or automatic Darwin promotion.

Validation: targeted kit/body/action-channel tests and the existing practice suite. Repository-wide lint has pre-existing violations; only changed files should be formatted in this patch.
