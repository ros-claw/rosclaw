# Coverage Cleaning: offline verifier and mission proposals

The Golden Task is **not yet accepted**. Nav2, opennav_coverage, Gazebo, daemon action executors and independent live collision monitoring still need integration.

`compile_mission()` uses existing TaskGraphV1/TaskNodeV1 contracts for precheck, localization, coverage planning, cleaning enable, guarded execution, independent verification, missed-region recovery, cleaning disable, final verification and Practice capture. Nodes remain PENDING; compiling a graph does not install capabilities or dispatch an action. Failure cleanup and conditional recovery execution remain runtime-integration work.

## Independent coverage

CoverageVerifier accepts an explicit reachable-cell mask, map frame/origin/resolution, a cleaning polygon and timestamped poses with cleaning-enabled flags. It rasterizes transformed cleaning footprints and interpolates bounded continuous segments; it does not generate coverage routes. Disabled cleaning, gaps, excessive speed, NaN, empty reachable area and frame mismatches cannot create swept coverage.

The denominator is the original accessible mask. Temporary blocks retain accessible area and missed components until actually revisited. Four-neighbor connected missed regions include cells, area, centroid and deferred status. MissedRegionRecovery proposes bounded retries with action-ID idempotency and per-cell attempt bookkeeping; it does not dispatch them or mark cells clean from action success.

Masks: 0 missed, 1 cleaned, 2 nonaccessible, 4 temporarily blocked. Permanent obstacles/exclusions must already be reflected in the supplied accessible mask. Cell-center rasterization is resolution-dependent; tiny holes and boundary uncertainty require a conservative map/resolution policy in live acceptance. Arbitrary complex/self-intersecting footprints and map reachability derivation are not supported yet.

## Mission proof

`verify_mission` recomputes coverage from supplied geometry/trace and requires zero collisions, complete independent collision observation, no trace gaps and >=98% coverage. A calculation alone yields NOT_VERIFIED. PASS additionally requires canonical daemon receipts with matching body/action/mission and independent evidence hashes bound to the exact evidence payload. Current executors do not emit these coverage bindings yet. FIXTURE and SHADOW receipts cannot establish physical coverage success. The frozen ExecutionReceipt contract is unchanged; the new artifact references it.

Replay:

```bash
PYTHONPATH=src .venv/bin/python scripts/ros_expert_replay.py --output /tmp/ros-expert-replay
```

This produces a synthetic 80%-blocked/100%-recovered mask, trajectory, diagnostics and mission proposal. Collision count is null and mission verification remains NOT_VERIFIED. It is not a Nav2 simulation result.
