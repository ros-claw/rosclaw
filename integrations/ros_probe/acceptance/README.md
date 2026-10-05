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
directory, configure its `home/config.yaml` with `native_tools.py` as the SIM-only
MCP source, and provision the existing Native model settings/authentication in
that isolated home. Do not commit credentials. Use the same model/provider and
budget for any comparison. With the repository's development test dependencies
and Node build installed:

```bash
PYTHONPATH=src .venv/bin/python integrations/ros_probe/acceptance/native.py \
  --directory /tmp/ros-expert-run/native
```

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
paused. Unknown stop evidence fails acceptance. Daemon SIGKILL cannot produce a
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
