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

External MJWarp scale evidence is also Ed25519-signed. The evolution command
verifies it against an independently supplied public key, then recomputes the
generated candidate's shield decision from the signed per-world risk and
collision labels. Each MJWarp shard runs the same controls through CPU MuJoCo
outside the GPU throughput timing window and requires exact agreement on the
critical collision label. A self-consistent JSON document is not trusted
evidence. The verifier also requires comparable workload dimensions, nested GPU
identity sets, one model hash, one device model, and no failed shards across the
1/2/4-GPU curve.

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
`SIM_CHAMPION`. In-process verification seals bind each Gate input to the
specific content that passed receipt, signature, and provenance checks;
caller-constructed booleans, counts, or modified bundles are treated as
missing evidence.

Phase 3 adds evidence-derived `ModuleProof` levels E0-E5,
`FailureSignatureV2` with eight safety-first routes, immutable
`PracticeDatasetSnapshot` artifacts, a learned ContactPush contextual policy,
and a body-scoped simulation Champion Registry. A ContactPush promotion also
requires same-seed failure-to-success evidence, a measured Memory ON/OFF
benefit, Know filtering with zero safety overrides, and candidate/stress/
dataset/Body identity agreement. Activation, ordinary Champion use, Canary,
freeze, and rollback are hash-chained after promotion.

`DataBudgetManager` bounds event/trace size, nesting, strings, episode/run/
workspace use, and fails closed for recursive or oversized input. Raw physics
states and private Holdout inputs must be written outside the source checkout.

## ROS command boundary

`GenericMobileBaseSimulationExecutor` is a SHADOW-only executor. A ROS command
sink must be bound to the same configured `daemon_*` instance identity as the
executor. This is an application binding, not a cryptographic daemon identity.
The rosbridge adapter accepts only turtlesim or explicit simulation-namespace
topics, exposes bounded straight-line Twist motion and stop, and captures a
fresh pose before and after motion. Angular commands remain blocked until an
orientation predicate is implemented. Every path after dispatch attempts a
zero-velocity stop, including observation exceptions. The canonical tested
route is:

```text
MCP request_action → rosclawd Unix socket → ActionGateway
→ daemon-owned executor → ROS command → fresh observation → receipt
```

Provider-side direct velocity publication remains blocked. A sent command with
no valid fresh observation can reach only `DISPATCH_CONFIRMED`, never
`TASK_VERIFIED`.

## Reproduction

Run these commands from the repository checkout. Store all keys and evidence
outside the checkout. The qualification operator must keep the private signing
key unavailable to candidate generation and pin the corresponding public key
independently:

```bash
rosclaw simforge suite validate --json

rosclaw simforge scenarios generate \
  --task shield-reach --output-dir /path/outside/checkout/scenarios

rosclaw simforge key create \
  --private-key /path/outside/checkout/simforge-private.pem \
  --public-key /path/outside/checkout/simforge-public.pem

.venv/bin/python scripts/simforge/run_scale_curve.py \
  --signing-key /path/outside/checkout/simforge-private.pem \
  --output-dir /path/outside/checkout/scale

rosclaw simforge evolve shield-reach \
  --output-dir /path/outside/checkout/evolution \
  --scale-curve /path/outside/checkout/scale/scale_curve.json \
  --scale-curve-public-key /path/outside/checkout/simforge-public.pem
```

The CoreBench four-GPU scale runner is
`scripts/simforge/run_scale_curve.py`; the Phase 3 ContactPush runner is
`scripts/simforge/run_contact_push_four_gpu.py`. Both require the separately
provisioned `.venv-mjwarp` worker environment. The finite Isaac Lab UR10
qualification is `scripts/simforge/isaac_reach_smoke.py`.
For the Phase 3 flagship and Gazebo process-chaos commands, evidence layout,
and interpretation rules, see
[Failure-to-Success Arena](failure_to_success_arena.md).

## Safety interpretation

Falsification that finds no counterexample is not a mathematical proof of
safety. A discovered counterexample is, however, a concrete, content-addressed,
strictly replayable engineering defect. `SIM_CHAMPION` means the candidate met
this benchmark's simulation gate; it does not authorize deployment to a real
robot.
