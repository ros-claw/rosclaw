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

## Generic fixture Python dependencies (offline image build)

The compiled generic Body loader needs Pydantic; the original known-body image
contains ROS fixture dependencies only. `Dockerfile.generic-runtime` adds the
five exact dependency wheels in `generic_runtime_requirements.lock`. The lock
currently targets the owned Jazzy CPython3.12/aarch64 fixture. Another ABI or
architecture requires its own reviewed lock; do not discard hash checks.
No robot asset, profile, simulator or actuator is launched by this build.

Prepare a fresh small context, copy the Dockerfile and lock, and download the
matching wheels before the offline Docker build:

```bash
mkdir -p /tmp/ros-generic-context/generic_runtime_wheels
cp integrations/ros_probe/acceptance/Dockerfile.generic-runtime /tmp/ros-generic-context/Dockerfile
cp integrations/ros_probe/acceptance/generic_runtime_requirements.lock /tmp/ros-generic-context/generic_runtime_wheels/requirements.lock
.venv/bin/python -m pip download --only-binary=:all: --require-hashes \
  --platform manylinux_2_17_aarch64 --python-version 312 --implementation cp --abi cp312 \
  -r /tmp/ros-generic-context/generic_runtime_wheels/requirements.lock \
  --dest /tmp/ros-generic-context/generic_runtime_wheels
docker image inspect --format '{{.Id}}' rosclaw/ros-expert-rebuilt:dad31022
docker build --network none --pull=false -t rosclaw/ros-expert-generic-runtime:1008-v1 /tmp/ros-generic-context
docker image inspect --format '{{.Id}}' rosclaw/ros-expert-generic-runtime:1008-v1
```

Freeze and verify the parent image ID before and after building; the validated parent is
`sha256:c31355f34739eb4ea8b60de1414c57ce0eea8225aa9e3ce854b7ceb66ca4376e`.
Freeze the resulting image ID in each future generic protocol. Import/reopen
checks use `--network none`, the frozen source read-only and a synthetic compiled
Body fixture read-only, and initialize no ROS Node or DDS connection. Passing
those checks is no held-out model, source/contact or L0–L4 physical acceptance.
### Frozen dynamic Native episodes

`dynamic_native_episode.py` accepts a closed
`rosclaw.dynamic_native_episode.v1` protocol for D2 or D4. Register exact source,
prior reviewed P0 merge, image ID, plugin/vendor hashes, seed, scene target,
model/provider and deadlines before launch. The entry refuses concurrent owned
physical episodes and requires PR 624 to be merged at the recorded commit.
It opens a new evidence directory and a fresh Native home using only the three
existing model configuration files; credentials remain private local files.

Robot actions still flow through the actual Native Agent, MCP and rosclawd.
The scene controller changes only the preloaded fixture obstacle. D2 requires
actual withdrawal, free-cell revisit and canonical whole-room acceptance. D4
keeps the obstacle present until the genuine FAILED/BLOCKED root closes, then
requires a canonical BLOCKED partial temporal artifact and complete source
replay. D4 is an expected safe failure, never a successful cleaning task.

After the owned simulator stops, D4 acceptance checks the exact introduced
component packet, continuing obstruction, occupied unclean cells, complete
zero-contact observations and an advancing independent stop window. The
separate actuator's closed genesis-to-end audit must bracket that same window
with brush OFF and no live lease. Native outcome, service acknowledgement and
zero requested velocity do not substitute for these observations. Missing SDK
usage after a Native process starts leaves model provenance UNKNOWN (`null`).

These entry and refusal tests are offline contracts. D1, D3, D5 and D6 do not
have full physical episode entry support here; do not label their preparation
or these synthetic tests as dynamic Native simulation acceptance.

### Independent contact evidence for observation faults

`contact_evidence.py` prepares an explicit policy from the actual owned
`robot.sdf`, `bridge.yaml`, brush binding and physics binding. The operator must
supply continuous support topics and exact ground collision names. Each Body
collision needs an explicit SDF contact sensor and a distinct Gazebo-to-ROS
bridge. An explicit independent pose topic supports namespaces; no robot
profile or name heuristic chooses support streams.

`contact_observer.py --policy <frozen-json> --duration <60..1920>` is a separate
passive process. It has no publishers, services or action/lease authority. It
retains original serialized ROS Contact and TFMessage bytes, hashes and receipt
times in a closed Core audit. Every declared contact stream, including streams
with empty contact lists, must actually arrive and remain fresh. Source silence
is UNKNOWN, not a zero-contact measurement. The first valid all-stream sample
admits this evidence source; subsequent loss or malformed input latches failure.
`independent-contact-latest.json` is a freshness/readiness projection only.

`closed_contact_evidence.closed_contact_window` uses the installed official ROS
CDR decoder to reopen the original messages. It checks parsed projections,
source/Body identity, genesis-to-close hashes, lossless writer closure and both
original SIM/wall brackets across the requested window. Each selected sample
must be complete, advancing and contact-free, with no gap over 300 ms. It does
not set canonical task state or prove brush state or standstill. The enclosing
negative Native episode must independently bind the prepared source, original
mission interval, canonical failed task/receipt, actuator OFF/lease evidence
and independent stop poses before making a physical safety claim.

The prepared source is not proof that the live simulator emits every stream.
If a Gazebo sensor publishes only on contact and supplies no fresh empty
messages, this strict independent observer does not admit that stream. A full
negative physical acceptance must resolve that source limitation rather than
infer zero contacts from silence. No D5 physical episode is accepted by this
component alone; the owned fault controller and full episode integration remain
required.

The official serialization/replay contract can be run inside the prepared
Jazzy image, with no Node/DDS/network/server:

```bash
docker run --rm --network none --entrypoint bash \
  -v "$PWD:/workspace:ro" rosclaw/ros-expert-generic-runtime:1008-v1 \
  -c 'source /opt/ros/jazzy/setup.bash && PYTHONPATH=/workspace/src:${PYTHONPATH} python3 /workspace/integrations/ros_probe/acceptance/contact_cdr_contract.py'
```

It serializes synthetic Contacts/TFMessage with official ROS types and tests
original-byte replay plus fifteen rejection cases. This is offline SDK
contract evidence, not simulator or robot acceptance.

`contact_observer_contract.py` additionally exercises the real process callbacks,
exclusive latest-file writes, audit closure and official-byte replay with a
synthetic delivery loop. Its seven cases include source loss, foreign contact,
contact, timestamp regression and malformed stamps. It starts no real Node or
DDS endpoint and remains offline contract evidence.
