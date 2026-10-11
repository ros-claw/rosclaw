# ROS Expert simulation acceptance

This disposable fixture uses official Jazzy Gazebo/Nav2 packages and pinned
opennav_coverage. The robot model comes from the installed Nav2 simulation
package; its cleaner is an explicit simulated attachment. No REAL executor is
registered. No host APT or ROS dependency replacement is required.

Build from the repository root:

```bash
.venv/bin/python integrations/ros_probe/acceptance/build.py
mkdir -p /tmp/ros-expert-run/golden
docker run --rm --name ros-expert-golden -p 19090:9090 \
  -v "$PWD:/workspace:ro" -v /tmp/ros-expert-run/golden:/evidence \
  rosclaw/ros-expert-jazzy:acceptance \
  bash -c 'source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && python3 /workspace/integrations/ros_probe/acceptance/stack.py --controller-watchdog'
```

Two explicitly supported fixture profiles are available: `--profile waffle`
(default) and `--profile burger`. Burger requires the actual installed ROBOTIS
`turtlebot3_description` URDF and `turtlebot3_gazebo` sensor/drive specification;
the image build installs both. It preserves vendor collision/inertial geometry
and uses the vendor 0.160 m wheel separation and 0.033 m wheel radius. Waffle
uses the Nav2 minimal simulation model. Both cleaners are simulation attachments.

Use a fresh owned evidence directory, unused loopback port and isolated DDS
domain for each stack. To run Burger, pass `--profile burger` to `stack.py`, then
run `run.py --directory <directory> --endpoint ws://127.0.0.1:<port>
--profile burger --mission-timeout 1800`. The Agent-side runner verifies the
actual URDF model identity before compiling its immutable Body. A profile/model
mismatch fails before daemon startup. Without `--profile`, the runner identifies
one of these two supported vendor models from the supplied URDF; it does not
infer or support unknown robot geometry.

The fixed profile defines physical radius, simulated cleaner polygon and legal
recovery centers consistently for the Body, denominator and independent witness.
Burger uses 0.150 m physical/recovery clearance and a 0.175 m cleaner half-width;
Waffle preserves its 0.250 m physical radius, 0.300 m recovery clearance and
0.275 m cleaner half-width. Cleaner size never alters collision clearance.
`fixture_profile.json` records the observer configuration and is validated
against the supported profile before observations start. Ground truth uses a
separate depth-one SENSOR_DATA bridge and matching best-effort subscription,
preventing old queued poses from being treated as current. Controller activation
completes before Nav2 startup. No external overlay or monkeypatch is required.

The official controller watchdog is enabled by default. `--no-controller-watchdog`
is reserved for explicit historical fixture replay; it is not the current
safety acceptance configuration.

The stack writes its measured map and actual robot URDF into the owned evidence
directory. For a fresh episode, restart only this owned container. Wait for the
native probe, Nav2 lifecycle nodes and independent contact streams to be ready.
In another terminal:

```bash
PYTHONPATH=src .venv/bin/python integrations/ros_probe/acceptance/run.py \
  --directory /tmp/ros-expert-run/golden --mission-timeout 900
```

`run.py` starts a separate rosclawd, requests actions through the existing MCP
wrapper, and saves canonical localization, coverage and memory receipts. Nav2
plans the coverage and connecting paths. Repairs use measured missed cells;
only stamped Gazebo poses with the cleaning state enabled contribute coverage.
The fixed denominator is computed from the measured map and configured geometry.
Contact-stream or ground-truth loss fails the mission. Coverage must reach 98%
with zero contacts and no trajectory gaps. Failure artifacts and failed Practice
episodes are retained, rather than converted into successful receipts.

For actual-model Native acceptance, use `run.py --prepare-only` in a new owned
directory, generate its `home/config.yaml` from the compiled Body with
`configure_native.py --directory <directory> --endpoint <same endpoint>`,
and provision the existing Native model settings/authentication separately in
that isolated home. The generator binds both the active Body and MCP
`required_body_types` to the compiled instance, rejecting stale/forged Body
declarations before writing. It does not overwrite an existing config unless
`--overwrite` is explicit, and never copies credentials. Do not commit credentials. Use the same model/provider and
budget for any comparison. With the repository's development test dependencies
and Node build installed:

```bash
PYTHONPATH=src .venv/bin/python integrations/ros_probe/acceptance/native.py \
  --directory /tmp/ros-expert-run/native
```

For an isolated loopback port, pass the same `--endpoint` to `native.py` and
the configured `native_tools.py` MCP command. Model identity is recorded from
actual SDK usage rather than a fixed runner label.

The runner sends only “完成整个房间清扫。” and acts as the explicitly configured
SIM test operator for exact independent authorization cards. The model chooses
the observer, localization, coverage and memory calls; MCP physical function
bodies cannot execute them. The runner requires canonical Memory completion and
existing TaskKernel success, and saves safe SDK token usage separately from
Core usage accounting. A failed independent artifact fails the journey.

During a mission, a passive temporary obstacle can be injected only clear of
the independently observed robot:

```bash
docker exec ros-expert-golden bash -c \
  'source /opt/ros/jazzy/setup.bash && python3 /workspace/integrations/ros_probe/acceptance/faults.py dynamic --x -0.8 --y -0.8 --dwell 20'
```

The obstacle's creation/removal and independent observations are recorded.
This local zero-contact subtest is not a complete-mission PASS: coverage,
canonical receipts, Practice, Memory and TaskKernel must still complete.

Accepted and failed evidence is documented in
[the implementation report](../../../docs/reports/ros-expert-harness/FINAL_IMPLEMENTATION_REPORT.md).

For live safety tests, start the owned fixture with `--fault-acceptance` as well.
This explicitly preserves physics when rosbridge/probe processes fail, so stop
evidence remains observable. Use a fresh fixture for each case:

```bash
PYTHONPATH=src .venv/bin/python integrations/ros_probe/acceptance/safety.py \
  observer_stop --directory /tmp/ros-expert-run/safety-observer \
  --fixture /tmp/ros-expert-run/golden
```

Cases are `daemon_kill`, `clock_pause`, `bridge_kill`, `observer_stop`. The final
case uses a second passive pose observer and checks the official controller's
0.2-second command timeout while the primary observation/lease process is
paused. The auxiliary live witness reader ignores only an unfinished append at EOF;
it rejects malformed completed records and preserves freshness checks. Canonical
mission trajectories are never repaired or filtered by this reader.
Unknown stop evidence fails acceptance. Daemon SIGKILL cannot produce a
terminal receipt; missing receipt is retained explicitly.

For the owned Isaac ROS 5 / Lyrical GPU fixture, source the pinned upstream
`cuda_buffer_backend` test-component build and run in a separate non-robot
DDS domain:

```bash
ROS_DOMAIN_ID=201 python3 isaac_resize.py --output /evidence/resize.json
```

This loads the actual upstream ResizeNode and native CUDA publisher/validator
in separate processes. The installed shared CUDA allocation library is
explicitly preloaded to bind embedded component and transport allocation
symbols to the same process-wide pool. A real validated cold-start frame
precedes 300 frames at 20Hz. The observer checks all content/backend metrics and
actual output dimensions; process death, stale IPC descriptors or incomplete
metrics fail acceptance. Input metadata observation forces CPU fallback and
native output pixel validation copies to CPU. These copies are explicit: this
is a GPU Resize / CUDA IPC subtest, not an Isaac Sim cleaning or zero-copy graph
acceptance. Failed and successful evidence is retained in RH13/isaac-live/resize.

Native probe freshness regressions run with ROS-host Python: `python3 integrations/ros_probe/acceptance/probe_cache.py`. They create no Node or DDS connection. In the inactive owned Golden fixture, `probe_pause.py` pauses and resumes only the acknowledged Gazebo world service and requires new wall-time probe captures showing clock progress true→false→true. The captured snapshot can be replayed through Core diagnosis at its original capture time; replay is historical evidence.

`run.py --reject-smaller-scope` checks that the configured rectangular whole-room executor returns canonical BLOCKED for a smaller requested area before physical dispatch. The Core resource scheduler lease is separate from the cleaning actuator lease. Independent physics evidence must still show standstill, disabled cleaning and no active cleaning lease.

Native execution RPCs retain their response channel for the dispatcher maximum
3600-second action deadline plus 60 seconds for receipt delivery. Read-only bridge
queries retain their five-second timeout. This transport bound does not extend
action authorization, leases, or verification deadlines.

## Passive coverage causal audit

The unchanged coverage/recovery algorithm now records bounded asynchronous
Nav2 feedback, exact consumer offsets at the main-pass boundary, and goal
intervals. `actions/coverage-audit-*.jsonl` is append-only and hash chained;
its separate summary records dropped events and writer errors. An incomplete
audit cannot satisfy the observability acceptance gate. It does not change
canonical coverage, motion policy, deadlines, or the independent observer.

The fixture observes the pinned server's existing coverage plan, field boundary,
planning field and swath debug topics. `plan-events-*.jsonl` preserves every
captured version, its frame, simulation/header time, wall capture time and
orientation. Plans carry no action ID on those ROS topics: the offline auditor
binds them only to an unambiguous serialized daemon goal interval; otherwise
the binding is UNKNOWN. Unexposed internal route/controller stages remain
UNKNOWN. The former overwritten `coverage_path.json` / `navigation_path.json`
are superseded by this event stream.

The daemon freezes public code/config hashes in `source-freeze.json`. This
contains no credentials, ledger secrets or private model reasoning. Dirty
working trees are identified explicitly; historical episodes without the new
events do not acquire invented provenance or phase timestamps.

After the task and fixture have stopped and flushed their audit summaries:

```bash
PYTHONPATH=src .venv/bin/python integrations/ros_probe/acceptance/coverage_audit.py \
  --directory /tmp/ros-expert-run/golden \
  --output /tmp/ros-expert-run/golden-audit
```

The output directory must be new. This read-only tool validates the canonical
artifact hash, reruns the original verifier and reports plan prediction
separately from measured coverage. The plan's ideal brush-on sweep never
enters canonical credit. Missing heading or causal observations remain UNKNOWN,
including old runs' precise phase times. Predicted-but-missed cells are not
automatically called tracking errors. Hash chains reject mutation/reordering
and incomplete trailing records rather than silently repairing them.
# Precise intermediate repair waypoints (opt-in diagnostic)

`paired_efficiency.py --precise-repair-waypoints` requires the same explicit
boolean in the preregistered protocol and the continuous repair strategy.
It applies only to the candidate arm. The owned stack retains its installed
Nav2 through-poses BT and source hashes, changes intermediate pruning from
0.7 m to the existing 0.025 m goal-checker tolerance, and checks pruning outside
the original planner rate controller. Planning frequency, controller limits,
recovery actions, leases, mission budgets, and independent coverage accounting
are preserved. Unexpected installed BT structure is refused before launch.

Source tests and registration with installed BehaviorTree.CPP/Nav2 plugins
are compatibility evidence. Physical waypoint arrival and efficiency require
fresh independently observed SIM runs; a reduced remaining-goal counter grants
no coverage credit. Interrupted or failed diagnostic seeds are never replaced.
