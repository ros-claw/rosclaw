# Owned source-only Nav2 diagnostic provider

The two explicit SDK observation roles capture before localization initialization
or after the operator's original localization initialization. They accept no
fault label, require no gold parameter value, inject no fault and accept all
observed PROPOSED/UNKNOWN/BLOCKED candidate states. Neither compiles a navigation
workspace, enrolls a Body or sends a goal. Existing engineering fault roles keep
their previous readback, readiness and rollback boundaries.

The closed host owner verifies a frozen recipe, complete SHA/size inventories
for original assets and the observer build, the original URDF and exact clean
source. Recipes accept no script, callback, command or model-supplied path. A
model selects only a registered probe ID. The owner creates at most one fixed
SDK physics World with network none, read-only root, dropped capabilities,
non-root UID, no devices and bounded Mesa/CPU/memory/PIDs/shm. It preserves the
staged Nav2 and bringup configuration; only new-run observer/initialization
metadata and the declared initial pose/observation role are prepared.

Original dispatch intent precedes SDK creation. The owner keeps the existing
capture loop alive while copying original Native files into the bounded window,
then acknowledges capture and stops its own container. Preparation/start/window
failures retain original intent/failure/logs and consume the one attempted slot,
with no automatic retry. Cleanup also handles a timed-out create using only its
unique name, fixed image and expected evidence mount.

The provider checks the original source/run/snapshot/URDF/requirements/staging,
recomputes the registered Core plan, checks the complete new replica target
scope and independently reviews stopped-container and original window records.
The physics original and single same-run ECM World are retained as source-only
observations. Each Native original-file read fits the request's existing duration;
the ambient follow-up stays inside the Core experiment budget. These timings
describe file reads from the owned capture loop, not new ROS snapshot RPCs.
SDK startup and cleanup are separately charged to the real host deadline.

The registered policy exposes the observation phase, scope and startup budget
to all five arms. A/D receive no Expert operations; B receives graph/constraint
inspection, while C/E add competing hypothesis plans and explicit probes. The
existing four original entry attestations are checked before provider binding
and again at dispatch. The provider shares the existing one-probe slot, costs
one physics World and leaves at most one canonical validation World. Missing
or changed recipe/context and insufficient time are rejected before that slot.
Synthetic clients cannot acquire SDK execution through this interface.

The old case is never rebound to the new replica. The independently compiled
replica graphs support observational follow-up only. Body/L1 admission,
canonical navigation, causal repair, Memory promotion and model performance
remain separate unaccepted claims. Frozen-source actual SDK observations and
independent process reviews are required in addition to synthetic contracts;
the scientific pilot still needs independent post-harness case/provider/Memory
registration and all four actual original attestations.
