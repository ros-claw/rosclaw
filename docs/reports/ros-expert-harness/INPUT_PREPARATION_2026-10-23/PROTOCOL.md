# Supplemental original-input preparation — 2026-10-23

This new operator preparation does not reopen or overwrite the 2026-10-21 entry
NO-GO. It collects new evidence before any separately registered model study.
No model calls are authorized by this preparation protocol.

Use the existing fixed 8cc3 SDK, rosbot-xl source, filtered sensor configuration,
source witness and canonical MCP → rosclawd navigation workflow. Preserve all
Body limits, source checks, permits, leases, watchers and stopping verification.
Only the SDK-owned acceptance fixture accesses native ROS; no Agent process does.

Current preparation v4 registers two new isolated stationary Worlds: environment
seeds 101 and 202. Require independently matched Source/Body, complete fresh L1,
parked physics, a closed Lease and zero control messages before capture. It sends
zero navigation goals. Earlier healthy navigation evidence and the seed-202
motion failure remain separate; stationary input success is not mission success.

Freeze healthy startup Nav2 YAML and a full runtime fault override YAML before
starting either World. They differ only in AMCL `tf_broadcast=true/false`. After
independent stationary Source/L1 review, collect healthy original facts, apply the one runtime
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
Healthy/restored cases require all constraints SATISFIED. The fault case requires
F-TF-001 VIOLATED and Scan/Lifecycle/Geometry/Route SATISFIED. The existing Source
binder also depends on TF and must retain its actual UNKNOWN result with the
sole reason `ROS_RUNTIME:tf.map_to_odom`; this dependency is not missing identity
evidence. Require initial independent MATCHED Source/Body, current full stream
continuity, matching identities/geometry and no Source mismatches. Keep all raw
Source reviews visible. This makes a **conditional decision input**, never a
new physical admission. No Source gate is modified, no UNKNOWN is relabelled
MATCHED, and fresh full Source/L1 is still mandatory before any future action.
Host refusal is not model abstention. No report grants L1 or execution authority.

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

Preparation v1 stopped on seed 101 after both healthy goals passed: the new
capture used segment 0 after ECM rollover, so independent Source review correctly
refused stale evidence. No fault was injected. Preserve this attempt. Forward
preparation v2 registers two new Worlds, captures the latest 64 original records
across bounded segments, and independently checks the full original hash chain
from the initially admitted prefix using the existing SourceStreamReader. It
does not relax freshness or continuity. These preparation failures and later
successes must remain separate; no model trial is rerun.

Preparation v2 passed healthy admission and captured an actual single TF fault
with all four non-Source constraints satisfied. It stopped because its original
operator criterion demanded MATCHED Source even though the unmodified Source
binder itself reports the TF failure. Preserve that failed criterion and raw
capture. Preparation v3 registers two new Worlds with the explicit dependency
criterion above, following the old FC-00 non-Source P1 test. It must preserve
UNKNOWN Source, validate the full chain, and verify a legal stage-only proposal.
The two failed preparation Worlds remain counted separately from the v3 pair.

Preparation v3 completed healthy/fault/restored originals and a legal stage-only
TF proposal for seed 101. Seed 202 completed its position goal, then its second
task was stopped by UNSAFE_PLANAR_LOCALIZATION_COVARIANCE; TF injection never
started. Keep that physical failure. V4 separates zero-command input acquisition
from motion stability, preserving all safety thresholds. It does not relabel v3
as PASS, rerun a model, or claim seed-202 navigation stability was repaired. Four
earlier preparation Worlds plus the two v4 stationary Worlds are all accounted.
