# SimForge CoreBench v1

SimForge CoreBench is ROSClaw's fail-closed simulation benchmark and bounded
policy-evolution layer. It is intended to discover physical/runtime failures,
produce immutable parameter candidates, compare them with paired evidence, and
promote only candidates that satisfy Statistical Gate V3.

It does not authorize real-hardware execution. Simulation evidence always uses
the `SHADOW` evidence domain and must never be relabeled as hardware evidence.

## Core contracts

The machine-readable contracts live under `benchmarks/simforge/`:

- `schema/` defines task, scenario distribution, evolution, and evaluation
  bundle JSON schemas.
- `suites/core_v1/` defines ShieldReach, ContactPush, GuardedBase, ROS2Chaos,
  and BodyMutation.
- `holdout/` documents the hidden-evaluator trust boundary.

Scenario cases are separated into Discovery, Development, Validation, Holdout,
and Counterexample Regression partitions. Seed derivation is HMAC based;
candidate-visible manifests disclose commitments rather than hidden seeds.
Holdout execution occurs in a spawned worker with private mode-0600 inputs and
returns an Ed25519-signed aggregate, never raw cases or seeds.

Scenario distributions support deterministic random, Latin hypercube,
boundary, and pairwise coverage sampling. Candidate search supports bounded
random, CMA-ES-style, cross-entropy, and Bayesian-style strategies. Every
candidate compiles to an immutable `CandidatePatch`; paths and values outside
the explicit whitelist are rejected, and arbitrary source edits cannot be a
candidate action.

## Evidence and promotion

Baseline and candidate evaluations use the same scenario and seed. Binary
outcomes include exact McNemar evidence; continuous robustness uses paired
bootstrap intervals, P05, and lower-tail CVaR. Gate V3 evaluates all checks
G1–G14, including:

- complete, independently verified physics evidence;
- paired scenario/seed identity;
- at least 200 Validation pairs, 200 Holdout pairs, and 1000 stress worlds;
- success, collision, unsafe-allow, false-block, and robustness bounds;
- hidden Holdout, counterexample regression, strict replay, hashes,
  cross-backend labels, data quality, and complete shards.

Absent or undersized critical evidence returns `NEED_MORE_EVIDENCE`. A safety
or quality regression returns `REJECTED`. Only a complete pass returns
`SIM_CHAMPION`.

`DataBudgetManager` bounds event/trace size, nesting, strings, episode/run/
workspace use, and fails closed for recursive or oversized input. Raw physics
states and private Holdout inputs must be written outside the source checkout.

## ROS command boundary

`GenericMobileBaseSimulationExecutor` is a SHADOW-only executor. A ROS command
sink must be bound to the same `daemon_*` identity as the executor. The
rosbridge adapter exposes only bounded Twist motion and stop; it captures a
fresh pose before and after motion. The canonical live route is:

```text
MCP request_action → rosclawd Unix socket → ActionGateway
→ daemon-owned executor → ROS command → fresh observation → receipt
```

Provider-side direct velocity publication remains blocked. A sent command with
no valid fresh observation can reach only `DISPATCH_CONFIRMED`, never
`TASK_VERIFIED`.

## Reproduction

Use the repository checkout on `PYTHONPATH` with the required project launcher:

```bash
PYTHONPATH=/code/rosclaw/rosclaw_test/src \
  /code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python \
  -m rosclaw.entrypoint simforge suite validate --json

PYTHONPATH=/code/rosclaw/rosclaw_test/src \
  /code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python \
  -m rosclaw.entrypoint simforge scenarios generate \
  --task shield-reach --output-dir /path/outside/checkout/scenarios

PYTHONPATH=/code/rosclaw/rosclaw_test/src \
  /code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python \
  -m rosclaw.entrypoint simforge evolve shield-reach \
  --output-dir /path/outside/checkout/evolution \
  --scale-curve /path/outside/checkout/scale_curve.json
```

The four-GPU stress runner is `scripts/simforge/run_scale_curve.py`; the finite
Isaac Lab UR10 qualification is `scripts/simforge/isaac_reach_smoke.py`.

## Safety interpretation

Falsification that finds no counterexample is not a mathematical proof of
safety. A discovered counterexample is, however, a concrete, content-addressed,
strictly replayable engineering defect. `SIM_CHAMPION` means the candidate met
this benchmark's simulation gate; it does not authorize deployment to a real
robot.
