# ADR 0018: Fresh SIM navigation readiness

Status: accepted for implementation; physical acceptance remains separate.

G1 source matching intentionally defers localization freshness. A new pure,
read-only reviewer consumes the original pinned source inputs again, at the
current review time, and applies every navigation observation check. It never
accepts a caller-supplied G1 PASS object or refreshes historical timestamps.

An independently supplied operator policy declares its identity, rationale,
maximum capture age, planar localization variance limits, and the two explicit
costmap topic/node associations. Its content hash must be pinned by the caller.
Policy variance limits cannot exceed the existing 0.25 ceiling or the squared
immutable Body goal tolerances. Complete original covariance, finite symmetric
planar covariance and positive semidefiniteness are required. These checks do
not certify a localization algorithm or probabilistic accuracy.

Fresh native GetParameters reads must bind each costmap's base/global frames,
clock, actual graph publisher, obstacle source and conservative footprint to
the discovered Body. Supported footprints are circles or convex polygons
containing the complete URDF conservative collision circle. Unsupported or
incomplete geometry remains UNKNOWN. The source binder separately verifies
the loaded Gazebo collision lineage. The original published footprint must also conservatively contain that circle
after transformation into the Body base using fresh original numeric TF.
Missing transforms or readback remain UNKNOWN.

The independent source-only physics witness must show zero contact, no lease,
no control messages, and a stopped Body. No Runtime, DDS access, authorization,
workspace writes or new Agent action tools are introduced. A positive result
is READY_FOR_SIM_TEST, with execution_authority NONE and L2 NOT_RUN. It is a
point-in-time observation, never daemon admission or an action permit.

Unknown evidence produces UNKNOWN; established source mismatch, unsafe
covariance, undersized footprints or movement produces BLOCKED. Negative
tests must re-pin mutated captures so semantic failures are exercised.

Known and held-out live Worlds, stopped sensor, frozen clock, missing TF,
inactive lifecycle, QoS and namespace faults are separate acceptance gates.
Synthetic contracts and replay cannot close G2. G3 additionally requires a
held-out robot selected after generic source/protocol freeze, canonical daemon
SIM receipts, independent motion/stop witnesses and fault acceptance. G4 needs
five staging repairs and fresh revalidation; G5 needs actual paired model runs.
REAL and coverage tuning remain outside this workstream.
