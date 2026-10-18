# RK-02: competing hypotheses and read-only measurements

Baseline: PR #723, main `b21cf1c8288b724d2ea510d5a144cf5b98d5d10a`.
This increment adds observational diagnostic proposals and falsification, not
causal verification, new Runtime/Body/Memory, staging writes or execution tools.

## Contracts and behavior

The graph compiler also accepts native observations without a navigation context:
`compile_operational_graph(model, source_ref=..., now=...)`. In this mode it never
calls the navigation Case compiler and does not require URDF or Nav2 roles.
`case_hash`, `source_urdf_sha256`, and `requirements_hash` are explicitly null;
source identity remains UNKNOWN and no Body/run/World admission is inferred.
Native stream findings use `signal.observed_steady_receive_window` and are scoped
to the observer, not downstream delivery or producer root cause. Latched or
incomplete windows remain UNKNOWN. Partial navigation inputs or attempted source
admission without navigation context are rejected. Supplying both original URDF
and requirements retains the existing navigation Case behavior. Non-null old
v1 graphs preserve their roundtrip hashes; older decoders must upgrade to opt
into the new explicit-null native mode.

`engineering.hypothesis.propose_hypotheses(graph, issue_ids=...)` proposes bounded
competing explanations for F-SCAN, F-TF, observed QoS/TF/time constraints and an
explicit unresolved explanation for unsupported issues. At most 64 hypotheses;
overlarge scopes fail explicitly and require selecting issue IDs.

`engineering.probe_planner.plan_diagnostic_probes(...)` seals a
`RosDiagnosticExperimentV1`. It prefers available native providers, then rechecks currently supported
explanations, followed by unknown candidate coverage and deterministic operation
order. Under a one-window budget the same receive-gap symptom selects endpoint
QoS, stream timing or clock measurements according to observed competing evidence. Every request is
read-only, proposed, unexecuted and has no authority. Each window costs five
seconds, total budget 1–30 seconds, maximum eight groups. Windows that do not
fit remain explicitly deferred. No probability or calibrated information gain
is claimed. Candidate coverage counts proposed predicate checks, not measured
information gain or mutually exclusive causes.

Each hypothesis includes supporting and contradicting provenance references,
confounders and concrete positive/negative measurement predictions. Receive
age uses steady time plus capture age. QoS compares observed requested/offered
policies only, not delivery or complete RMW compatibility. QoS explanations
retain exact topic scope even when unrelated topics share the same publisher and
consumer nodes. A pair-specific QoS issue retains its publisher and consumer;
an incompatible sibling consumer cannot keep an already resolved path accused. Clock disagreement
covers observed nodes only. TF ages respect declared future tolerances. Missing
attribution never establishes unique authority. Several explanations can be
supported together; eliminating one does not prove the remaining one.
Clock findings require each original per-node parameter value to agree with its
projection and its own aware readback timestamp to be within 500 ms. Simulated
clock receive age includes elapsed capture time. System-time TF age additionally
requires an observed system-time probe; a missing or simulated probe clock cannot
be silently mixed with system-time node stamps. Simulated TF age uses the native
clock stream independently of the probe node parameter. Old v1 graphs with envelope-only
clock provenance remain UNKNOWN in the hypothesis engine and need new evidence;
they cannot silently inherit the stronger clock checks.

`assess_diagnostic_followup(experiment, before, after, now=...)` verifies the
original deterministic proposal, new snapshot, newer capture, budget, source
reference, URDF, requirements, available run/world/Body identities, evidence
domain and aware review clock. Evidence must itself postdate the proposal:
a fresh envelope containing old facts is insufficient. Changed source/World
is rejected; isolated counterfactual Worlds require the separate RK-03 protocol.
The same logical `source_ref` must be supplied by the capture owner for both
snapshots; this strict API does not automatically reconcile differing file paths.
Absent run/Body/World identities remain absent and never establish physical
continuity. Supported explanations can be withdrawn when current measurements
contradict their narrow predicate. Missing or conflicting evidence stays UNKNOWN.
Graph hashes establish integrity, not authenticity or actual probe execution.
Assessment explicitly reports `NOT_ATTESTED_BY_GRAPH` and
`causal_verification=NOT_ESTABLISHED`. Source review remains caller supplied.

## Native provider observations

The existing ROS sidecar adds actual discovery endpoint GIDs, liveliness lease
duration, GIDs on incompatible pairs, and a bounded raw TF event window with
receive times and available sequence numbers. When actual message publisher
GID exists, TF cache keys preserve distinct authorities. Missing, invalid or
all-zero GIDs stay null; node names or publisher counts are never substituted.
Cache/window bounds are 1024 with explicit drop counters; dynamic output/window
receives expire after five seconds. No completeness or uniqueness is inferred
from this observer's window. This does not implement full RMW compatibility.

Important platform boundary: the checked Jazzy Python subscription implementation
returns a metadata dictionary with timestamps and sequence numbers, without a
publisher GID. This provider can preserve raw events and discovery GIDs but cannot
attribute those individual Jazzy messages. Such authority checks remain
`PROVIDER_EXTENSION_REQUIRED`; future native providers need separate acceptance.
Primary sources: [Jazzy subscription implementation](https://github.com/ros2/rclpy/blob/jazzy/rclpy/src/rclpy/subscription.cpp),
[Jazzy endpoint info](https://github.com/ros2/rclpy/blob/jazzy/rclpy/rclpy/topic_endpoint_info.py),
[Jazzy callback dispatch](https://github.com/ros2/rclpy/blob/jazzy/rclpy/rclpy/subscription.py).
Callback metadata is required explicitly so rclpy does not silently dispatch a
one-argument callback. Core modules do not import ROS client libraries.

## Offline CLI

Use the repository-pinned CLI, with `PYTHONPATH=src:.` in a source worktree:

```bash
.venv/bin/python -m rosclaw.entrypoint ros diagnostic-plan \
  --operational-graph graph.json --issue-id F-SCAN-001 \
  --budget-sec 20 --output experiment.json --json
.venv/bin/python -m rosclaw.entrypoint ros diagnostic-assess \
  --operational-graph graph.json --experiment experiment.json \
  --followup-graph newer-graph.json --reviewed-at 2026-10-18T02:00:00Z --json
```

Commands only read bounded regular non-symlink inputs. Output is an exclusive new
artifact; existing files cannot be overwritten. Historical evidence retains
`SEALED_REPLAY`. These commands do not discover ROS or execute a probe.
The offline benchmark module `replay_diagnostic_experiments` accepts 1–128
SHA-pinned graphs and explicit issue scopes. It rechecks inputs, writes exclusive
outputs and labels them proposals, never physical or model efficacy results.

## Acceptance and remaining gates

Tests include same scan symptom with QoS versus clock explanations, fresh sensor
receive versus missing consumer delivery, TF age versus attributed authority,
wrong source/Body/requirements, old facts in new snapshots, replayed proposals,
budget exhaustion, forged predictions, no transport/file I/O in the planner,
CLI dispatch, sidecar metadata callback shape, invalid GIDs, cache bounds and
Jazzy missing attribution. These are synthetic protocol tests.

Independent original capture/replay and exact source/main verification records
are stored under workspace `recovery_2026-10-18`; preserve development failures.
Do not infer real fault injection from synthetic fixtures, historical replay,
or a healthy native provider capture. Generic consumer delivery, executor and
plugin parameter diagnosis remains unresolved by this graph version; it requests
an extension instead of silently declaring an issue absent. No RK-03
before/intervention/after/rollback acceptance or new model benchmark is claimed.
The prior G5 and 1016 NO-GO results remain frozen. Next gates are an independently
preregistered new-mechanism native/SIM protocol, guarded counterfactual execution,
new Body application compilation, transferable mechanisms and the new benchmark.
