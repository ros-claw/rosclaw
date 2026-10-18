# Operational diagnostics in the new RSK client

`OperationalRskACI` composes an existing new B/C/E interface, including its
application/Memory features. It connects the accepted Core operational graph,
hypotheses and diagnostic planner to model-visible operations without changing
G5, Core, ROS drivers, executors or navigation admission.

- `inspect_operational_graph` pages original QoS, TF, time and dataflow findings.
  It reports omitted target counts explicitly; original raw files remain equally
  readable in A/B/C/E. The graph uses original capture/review clocks and never
  renews old evidence into current L1.
- `plan_diagnostic_experiment` selects actual graph findings. Core returns
  competing narrow hypotheses, discriminating measurements, unknowns and deferred
  scope. Plans are hashed, unexecuted and carry no authority. The new interface
  requires a10–30 second plan budget so a full five-second ambient follow-up and
  capture overhead fit the unchanged Core experiment budget.
- `run_diagnostic_probe` is explicit and C/E-only. It selects a previously issued
  experiment/probe identity, rechecks the deterministic original plan and source
  entry, and uses the same single probe counter as legacy named investigations.
  Missing/unsupported providers and insufficient wall budget do not run a probe.

An optional provider requires all four original pilot gates, exact source/run,
URDF/requirements/staging policy and an exclusive owned operator. Synthetic
clients cannot use or acquire it. A diagnostic-only provider hides the legacy
investigation callback path and rejects that path before consuming its budget.
Failures after SDK dispatch retain the attempt, original failure file, owned
container cleanup and consumed slot. There is no automatic retry.

## Initial provider scope

The initial provider is deliberately **sensor-only isolated DDS**, not a general
robot probe. Its closed operator creates only frozen LaserScan test endpoints
inside the existing pinned SDK image, with network none, read-only filesystem,
no devices, dropped capabilities, non-root UID and bounded CPU/memory/PIDs.
It creates no robot driver, Runtime, navigation action server or actuator topic.
A configured producer stop completes before the observation window. Publisher
QoS and task configuration do not change during the window.

Standalone sensor-component acceptance uses `NativeDiagnosticContext`: original
model/source/run plus a snapshot hash, with URDF, navigation requirements and
staging bindings explicitly absent as a complete group. It compiles native-only
Core graphs and never fills fake navigation roles to fit the old Case schema.
The composed navigation facade keeps its real navigation context when present;
this distinction does not create a generic non-navigation model episode runner.

The SDK records two exact Native captures, independent consumer counts/steady
window, source file hashes and original container inspections. The reviewer reads
those originals, checks the current stopped container against its recorded
identity/config/mounts, enforces source/config/URDF/requirements/run bindings and
compares the actual endpoint identity/QoS in both captures against frozen inputs.
Owner booleans alone are insufficient. Source paths and all file references are
host-owned, bounded and SHA-pinned; models cannot choose commands or paths.

The registered diagnostic facts are common to A/B/C/D/E, including a frozen
provider policy when that case offers a probe. Host validation independently
recomputes the graph and binds source/run/snapshot/URDF/requirements/staging for
every arm. An attached provider must additionally match that exact policy.
Providerless arms accept the same facts without gaining a provider or any Expert
operation. In particular A retains generic files/staging and D retains its
zero-model compiler; their schemas and execution permissions do not change.

Each Native snapshot call must fit the original Core request's five-second
maximum. The ambient interval between captures is separately recorded in full
and must be at least five seconds and within the declared diagnostic experiment
budget. Original timestamps are never retimed. Motion500ms/250ms/100ms gates and
physical constraints are untouched.

A new replica is not compared directly with an old World as if it were a
same-World follow-up. The original invariant/exact target scope is resolved in
the new probe's own before capture, its plan is independently rebound there, and
Core assesses that probe's before/after captures under the same run/source URI.
Missing target scope abstains. The result records the original intent and new
rebound plan separately. The original case is not rebound or admitted.

Successful original replay is only an observational update. It does not establish
causality, navigation/physics success, live Body state, L1 or Memory promotion.
Sensor-only SDK containers have zero physics Worlds; the shared probe counter
reserves one budget slot, so `new_Worlds_requested` is distinct from the explicit
`actual_new_Worlds=0` in this provider's result. Other measurement providers remain
unavailable/UNKNOWN until separately implemented and validated.

## Context and remaining acceptance

New-RSK workbook projections summarize whole-file application before/after text
with exact length/SHA and preserve diagnostic hypothesis statuses. Full original
tool results remain unchanged in per-turn artifacts. This prevents generated
files from unnecessarily exhausting the decision context and avoids the old
workbook's generic fresh-probe view hiding a diagnostic assessment. Old G5's
workbook/context is unchanged.

Synthetic tests and original replay are not model or SDK success. Actual source
and exact-main native acceptance, first CI and public original sealing remain
separate checks. The provider's direct component acceptance is not a pilot gate
or model episode. New independent case/provider registration and original entry
attestations must still be frozen before a scientific pilot can start. In
particular this sensor-only implementation is not evidence of generic probes
against an unfamiliar live robot or autonomous repair of arbitrary new faults.
