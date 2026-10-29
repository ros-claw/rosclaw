# Daemon-owned public coverage slice

This is a bounded capability extension and a bundled capability-only App. It
does not establish complete-house cleaning, hardware support, recharge/resume,
or an integrated HomeClean simulation qualification.

`public-coverage-slice` calls `coverage.execute_slice` once. This distinct ID
cannot route to the older simulation-witness `coverage.execute` executor.
Within that one canonical action, the trusted executor selects public work,
transfers with its brush off, enables the brush, follows a bounded path, and
finishes with fresh stopped/off observations. The existing daemon job, Session,
Action Lease, exclusive Body lease and Receipt cover the intervals between
internal primitives. No nested daemon actions, second task kernel, Agent ROS
transport or new authorization entry are created.

## Installation boundary

An operator-owned Driver supplies `PublicCoveragePort`. It is not an Agent
plugin and cannot be selected through action arguments. After compiling the
Body and installing the trusted port, daemon setup can register:

```python
binding = PublicCoverageBinding.from_body(compiled_body)
executor = PublicCoverageExecutor(binding=binding, port=trusted_port, ledger=ledger)
runtime.action_gateway.register_executor("coverage.execute_slice", binding.execution_mode, executor)
runtime.register_driver("public_coverage", executor)
```

The port must use existing native Driver primitives internally under the outer
action's lease. Each result must carry the exact parent action ID and dispatch
ID given by the executor. The port retains the original native goal through
cancellation/result reconciliation, honors the supplied absolute deadline and
stop event, and implements the existing live-path and sensor watchdogs. Its
emergency stop must latch the native Driver against later motion. An RPC ack
does not prove stop or native-goal termination.

`snapshot(work=False)` must collect original public facts without requiring or
building work geometry. Stop/off verification uses this mode. Cooperative
planning also uses this inexpensive mode while checking every declared source,
the original deadline, native motion state, and battery/dock reserve. The held
planning grid is only proposal input; dispatch still obtains and validates a
fresh full work snapshot. Thus a planner callback cannot rebuild a whole grid
repeatedly or let a missing map block observation of an actual stop.

The bundled App grants no capability, Driver access or permit by installation.
It currently requires an already undocked robot, sufficient battery reserve,
and current public graph registration. Charging, return/docking and recovery
remain integration work. It is not the full mission controller.

## Compiled Body and observations

The required `provider_interfaces.public_coverage` object has schema version
`rosclaw.public_coverage.v1`, FIXTURE or SIMULATION execution mode, a Driver
observation-contract SHA256, odom/graph epochs, maximum work count, finite wall
duration, minimum work SOC, physical/brush radii and planning margin in
`geometry`, and an explicit `sources` array. These fields are inside the actual
EffectiveBody hash; uncompiled profile-only metadata is insufficient.

Each source declares its semantic role, topic, ROS message type, frame, pinned
publisher endpoint GID and original receive/header age limits. TF contracts
also bind the child frame. Pinning a GID inside this configured non-hardware
environment is not cryptographic hardware identity. Neither bindings nor this
App promote generic Sense state to REAL or grant execution authority.

Endpoint bytes retain the native RMW storage width: 16 bytes (including Jazzy)
or 24 bytes. Missing, all-zero, malformed and other-width identifiers are
rejected. No padding, truncation or node-name inference is allowed. The
trusted Driver must receive the original per-message publisher GID; a Python
client that omits it needs a native metadata bridge before enrollment.

Required roles are odom, brush, occupancy, global/local costmaps, public graph,
map/base TF, scan, depth, battery and dock state. Receive time uses monotonic
time with no future allowance. Header time uses the public native clock.
`max_header_future_s` is compiled into each source and defaults to 100ms;
sensor/map roles cannot exceed 100ms. TF roles may explicitly retain the
native Driver's one-second absolute TF guard (including normal SLAM future
dating). Their original receive and past-header limits still apply. Original
message timestamps are never rewritten. Headerless Bool state keeps frame/header
time absent. MarkerArray graph events can explicitly use receipt freshness
and a common marker frame without manufacturing a container header clock.

Work admission requires the complete live work set, known settled previous
native motion, the exact compiled footprint and both costmaps. Stop checks
need new odometry/brush observations and do not depend on unrelated depth,
map or battery health. Estimated brush credit uses actual ordered odom stamps,
brush observations and public graph/TF registration, validated as captured;
draining older history must neither renew freshness nor invent interpolation
across unknown registration. Missing data, source/Body/epoch changes, stale
facts, wrong primitive receipts and out-of-order odom fail closed.

## Evidence scope

The supplied tests use an explicit software port, real geometry/ledger code,
and actual authenticated Unix daemon RPCs. They are FIXTURE evidence, not ROS
or physical evidence. They cover the brush-on interval between native actions,
Session loss, active cancellation refusal followed by emergency stop, and the
real AppRunner -> daemon -> executor path.

Canonical limited-evidence classifications are preserved, including DEGRADED
fixture receipts. The App checks the finite-slice verification result and
scope; success means its bounded workflow completed. `whole_house_success`
remains null and no Judge result is read. A fresh ROS Driver adapter, live
fault qualifications, restart/worker recovery and full mission acceptance are
still required before declaring integrated delivery.
