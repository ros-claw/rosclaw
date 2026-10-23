# Supplemental original-input preparation — 2026-10-23

This new operator preparation does not reopen or overwrite the 2026-10-21 entry
NO-GO. It collects new evidence before any separately registered model study.
No model calls are authorized by this preparation protocol.

Use the existing fixed 8cc3 SDK, rosbot-xl source, filtered sensor configuration,
source witness and canonical MCP → rosclawd navigation workflow. Preserve all
Body limits, source checks, permits, leases, watchers and stopping verification.
Only the SDK-owned acceptance fixture accesses native ROS; no Agent process does.

Maximum two new isolated Worlds: environment seeds 101 and 202. Each first runs
the existing two small healthy canonical navigation goals with independent stop
verification. No new goals may run during subsequent input capture. Require the
robot parked and its action Lease closed before changing any configuration.

Freeze healthy startup Nav2 YAML and a full runtime fault override YAML before
starting either World. They differ only in AMCL `tf_broadcast=true/false`. After
healthy navigation, collect healthy original facts, apply the one runtime
parameter override through the owned SDK fixture, collect the first admissible
single-TF-fault facts, restore the original parameter and collect healthy facts
again. The fault YAML is the source of a runtime override, **not** evidence that
Nav2 cold-started with that YAML. No lifecycle activation, TF workaround,
additional publisher, geometry change or threshold adjustment is permitted.

Each capture has a 20-second acquisition budget and retains its status timeline,
original native/system/physics observations and complete ECM prefix. Any observed
additional non-ACTIVE selected lifecycle node during the fault invalidates the
single-fault preparation. The external operator independently inspects the owned
container and recomputes actual Source binding and all Engineering Case facts.
Healthy/restored cases require all constraints SATISFIED; the fault case requires
only F-TF-001 VIOLATED and every other constraint SATISFIED. Host refusal is not
model abstention. No report grants current L1 or execution authority.

The fault input must also be accepted by the existing stage-only TF repair
compiler against the frozen runtime override YAML, with exactly one false→true
change and an exact rollback. The fixture never invokes this candidate as an
action or claims it came from a model.

Stop and retain every failure if a World or phase fails. Corrections require a
new preparation version and separate retained attempts; no selective rerun may
be represented as a clean first attempt. At most 600 seconds per World. No model
experiment or model physical success is inferred from successful preparation.

Engineering Case TF freshness is aligned with the existing declared transform
future tolerance used by diagnosis and Operational Graph. The 500 ms stale bound
and capture freshness remain unchanged. Future stamps outside the declared
bound, stale facts and competing authority remain unaccepted.
