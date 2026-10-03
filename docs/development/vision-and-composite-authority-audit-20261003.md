# Camera evidence and composite body authority audit

This recovery follows the host reboot on 2026-10-03. Historical Kimi scores and
raw sessions remain immutable. Previously verified V02/V03 runs used weak
evidence gates and do not qualify visual reasoning or camera calibration.

## Independent visual evidence

The old V02 gate accepted any nonempty label together with any render artifact.
The old V03 gate accepted `consistent=true` with any two renders, or
`consistent=false` without observations. Neither compared labels, pixels,
camera identity, calibration, or a common world state.

The revised tasks require content-addressed `rosclaw.sim.camera.v2` observation
manifests and lossless raw NPY arrays. V02 names an integer object ID, object
type, and an actual pixel within the blue block's mask. V03 requires depth and
segmentation observations from both named cameras, and the projected cube
center in pixels.

A separate native MuJoCo renderer reconstructs the original compiled scene
and complete `mjSTATE_INTEGRATION` initial state without importing the
producer's camera worker. The oracle checks raw segmentation exactly, metric
depth within 1e-6 absolute/relative tolerance, model/state digests, categorical
channels, intrinsics/extrinsics, visible target masks, and camera projection
within half a pixel. It produces no physics steps. Typed false consistency,
duplicate cameras, wrong projections, channel substitution, raw-mask forgery,
incorrect labels/types/pixels, changed state binding and false calibration are
rejected.

Actual EGL rendering verified that all original target scenes are visible:
the V02 blue block has 1,021 visible pixels; the V03 cube has 1,202 and 4,375
pixels in the two cameras. These are renderer fixture results, not new Kimi
task success. Fourteen native-render oracle tests pass. A fresh native agent
wave is required to claim agent visual task success under this stronger
contract. Observation evidence is inspectable execution evidence, not remote
attestation against a hostile agent controlling the host filesystem.

## Composite body boundary

The trusted embodied envelope is constructed from the mission's body binding
and the service's active body source, not from an arbitrary loaded MJCF:
`src/rosclaw/agentd/pi_bridge/context.py:build_embodied_context`.
The envelope describes one body ID/effective hash. Workspace registries can
store multiple body entries but expose one current body pointer; they do not
automatically turn a G1 plus M20 scene into a single trusted composite body.

`rosclaw_inspect(kind="robot")` returns `UNKNOWN_ROBOT` when the ecosystem
index cannot find an authoritative robot chain. A local upstream M20 model
and SDK provenance receipt are valuable simulation provenance, but are not
the missing canonical index chain or a verified physical execution provider.
G1 catalog capabilities are static capability descriptions, not evidence that
the current 29-DOF hand-racket controller works. The formal mission's existing
sim/ur5e binding must not silently be relabeled as G1/M20.

The current native tennis implementation therefore remains an external native
MuJoCo experiment operated through the task's generic process/evidence tools.
Its source, body and dynamic receipts must explicitly identify the G1/M20
scene and retain this authority limitation. Promoting it into an embodied
capability requires a separate simulation-only composite body/profile,
content-addressed source/body/asset binding, an explicit capability provider,
and a declared action/observation contract validated through the framework.
None of those steps grant REAL access, establish M20 commercial calibration,
or justify altering the live mission's body binding during an active task.

No live body, physical driver, catalog support status, or original simulation
receipt was modified by this audit.
