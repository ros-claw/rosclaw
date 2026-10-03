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

The fresh native Kimi wave exposed a public API gap that the original renderer
fixtures did not cover: `SimulationRuntime.snapshot(model_ref)` still created
a legacy v1 snapshot, while the low-level fixtures explicitly called
`initial_state_v2`. Correct labels/raw pixels therefore lacked the required
full initial-state binding. Both V02/V03 historical wave results are retained.
The public default now captures actual FULL_INTEGRATION v2 state without a
physics step. Explicit legacy input references remain LEGACY_PARTIAL; missing
history is not fabricated. Three actual runtime/CLI/MCP regression cases
failed against the old default and now pass, including public CLI
load→snapshot→observe→independent pixel verification. Independent reset/forward
and `mj_getState` reproduce the entire captured vector exactly.

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

## Renderer environment recovery

Wave 9's public full-state camera evidence exposed an evaluator environment bug:
operator `PYOPENGL_PLATFORM=osmesa` survived into a worker requesting `MUJOCO_GL=egl`.
EGL import failed and the evaluator silently rendered with OSMesa. Cross-renderer
edge pixels and depth then differed from the producer's declared EGL arrays.
Original source and stored MJCF compiled camera/body/geom/visual fields matched.

The independent evaluator now binds both GL variables to the observation's declared
renderer and performs exactly one backend attempt. An unavailable renderer or
worker timeout is `INFRASTRUCTURE_FAILURE`, with no verified success and no agent
false-success attribution. Known invalid initial-state/target evidence remains a
validation failure. Exact segmentation, depth tolerances, source/model/state,
calibration and target projection checks are unchanged.

A real public load→default snapshot→segmentation observation regression first
failed with `CAMERA_RAW_PIXELS_MISMATCH` when the operator shell was switched to
OSMesa after producing EGL evidence; it passes after the fix. Negative tests prove
no fallback and honest infrastructure classification. Existing wave 9 V02/V03
raw evidence is rechecked read-only; historical scores and transcripts remain
unchanged. No physics steps or model reruns are required for this correction.
