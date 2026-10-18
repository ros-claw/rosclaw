# RK-03 bounded intervention and experiment records

`ros intervention-plan` compiles an untrusted parameter hypothesis and original
Native readback into a proposed single difference in isolated SIM staging.
It consumes a sealed operational graph/system model, original Nav2 YAML,
bringup JSON and a typed proposal. It never opens ROS, patches a live workspace,
starts a World, authorizes an action, or declares a repair successful.

```sh
PYTHONPATH=src:. .venv/bin/python -m rosclaw.entrypoint ros intervention-plan \
  --operational-graph graph.json --snapshot system.json \
  --staging-nav2 nav2.yaml --staging-bringup bringup.json \
  --proposal proposal.json --output plan.json --json
PYTHONPATH=src:. .venv/bin/python -m rosclaw.entrypoint ros experiment-result \
  --plan plan.json --phases phases.json --output result.json --json
```

The CLI accepts bounded regular inputs, rejects symlinks and YAML aliases, and
creates outputs exclusively. Offline replay retains its original evidence domain
and capture times. A newly written plan cannot renew a parameter observation.

The first bounded semantics are a concrete node's `transform_tolerance` (0–5 s),
a literal `.sensor_frame`, or a nonempty `.plugin` class candidate. An explicit
sensor-frame override can be removed with `RESET_COMPONENT_DEFAULT`, predicting
an empty readback. That prediction still requires the installed component's
actual default/frame checks in the independent After World. A string naming a
plugin is not proof that its class is installed. There are no robot-model branches.
Other parameters, Body limits, controls, controllers and geometry require their
existing compilers and cannot be changed through this proposal surface.

A proposal must match one concrete original YAML path and the current per-node
Native readback. Missing/stale readback requests a read-only probe, a contradictory
baseline abstains, and an unchanged value returns `NO_CHANGE`. A concrete plan
binds the exact baseline, one declared After difference, byte-semantic rollback,
original provenance, frozen experiment revision and bounds: three Worlds, two
After goals, two total goals and a declared total wall-time budget. Its intrinsic
validator rejects extra differences even if the envelope is rehashed.

The operator reviews the concrete diff before starting new owned isolated Worlds.
Before and Rollback may be unadmitted source-only fault captures with zero goals;
this must remain explicit. Every After action requires fresh full L1, the existing
source/Body binding and canonical MCP → rosclawd guarded execution. Independent
stop, zero contacts, source continuity, consumer effects and negative hypotheses
remain mandatory. The 500 ms source / 250 ms proof and Body limits are unchanged.
All failures stop the trial and preserve original artifacts.

`experiment-result` binds three ordered closed-before-next phase records. It
checks unique World/run identities, unchanged source/image/URDF/scene/controllers/
requirements/limits, exact staging hashes, original parameter identity/value/time,
wall budget and distinct action/receipt pins. Missing proof roles are explicit.
Its result is only `INCOMPLETE_ORIGINAL_EVIDENCE` or
`NEEDS_INDEPENDENT_ORIGINAL_VERIFICATION`: the pure contract does not read the
pinned originals, authenticate a source, attest physics, establish causality,
promote mechanism memory or grant execution authority. Existing independent SDK
verifiers must replay the actual originals separately.

The adjacent generated JSON schemas describe the proposal, plan and result.
Synthetic tests validate boundaries; source/main original replay, new World
experiments and PR delivery are distinct evidence gates, never implied by this
implementation document.
