# Failure-to-Success Arena

The ROSClaw Failure-to-Success Arena is a simulation-only causal validation
workflow. It demonstrates that a recoverable ContactPush failure changes
downstream behavior through Practice, Memory, Know, How, Auto, Darwin, and the
simulation Champion Registry. It also provides a separate Gazebo GuardedBase
workflow for the canonical MCP-to-rosclawd command path and real process
faults.

Neither workflow authorizes a real robot. All actions use the `SHADOW`
evidence domain, raw evidence must stay outside the checkout, and a
`SIM_CHAMPION` is not a hardware safety approval.

## What the ContactPush run proves

A full run performs these steps without human candidate selection:

1. Execute and independently replay 120 MuJoCo Practice episodes.
2. Build an immutable, body-scoped Dataset Snapshot with HMAC-grouped
   development, validation, and private Holdout partitions.
3. Train a contextual policy from the public development partition.
4. Reproduce a fixed-policy overshoot, classify it with
   `FailureSignatureV2`, and compile a whitelist-only How intervention.
5. Compare Memory OFF and Memory ON on the same scenario, seed, and Body.
6. Execute parameter, trajectory, Skill Graph, and learned-policy candidates.
7. Reject an unsafe Candidate A in the physics Sandbox.
8. Evaluate Candidate B on 200 Validation pairs, 200 process-isolated hidden
   Holdout pairs, Counterexample Regression, and 1,000 worlds sharded across
   exactly four physical GPUs. Physics-receipt bundles and the independently
   signed Holdout are process-sealed; the four-GPU aggregate must carry a
   valid Ed25519 signature bound to the run trust key.
9. Require Statistical G1-G14 and the Phase 3 causal gates before producing
   `SIM_CHAMPION`.
10. Activate the Champion in a body-scoped simulation slot, run an ordinary
    task through that active slot, detect a promoted but regressed Canary,
    freeze it, roll back, and retry successfully.

Module evidence levels are derived from evidence fields:

```text
E0 exists
E1 invoked
E2 valid output
E3 changed a downstream decision
E4 passed fault injection
E5 strict replay
```

The level cannot be supplied by a caller. Promotion requires every core
ContactPush module to reach E3; Sandbox, Practice, and Darwin must reach E5
before activation. Registry E5 is added only after activation, ordinary use,
Canary, and rollback have completed.

## Run the flagship workflow

Inspect GPU ownership first. The runner never kills unrelated GPU work and
uses one bounded shard per requested device.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv

.venv/bin/python \
  -m rosclaw.entrypoint demo run failure-to-success \
  --task contact_push \
  --profile full \
  --output-dir /code/rosclaw/phase3-evidence \
  --live-dashboard
```

`--profile smoke` keeps all contracts but reduces the statistical sample sizes;
it is not a full acceptance run. The output directory must not already exist
and must be outside the source checkout.

The module-causal route runs the same evidence-producing workflow and can
enforce an explicit module subset:

```bash
.venv/bin/python \
  -m rosclaw.entrypoint proof run module-causal-v1 \
  --task contact_push \
  --profile full \
  --modules body,provider,failure_router,sandbox,practice,memory,know,how,auto,darwin,registry \
  --output-dir /code/rosclaw/phase3-proof-evidence
```

Inspect a completed proof tree:

```bash
.venv/bin/python \
  -m rosclaw.entrypoint proof show /code/rosclaw/phase3-evidence --tree
```

## Dashboard and export

`--live-dashboard` exports complete before, after, and split-screen MP4 traces,
key receipts, a self-contained `evidence.html`, `technical-report.json`, and a
hash manifest. It does not silently start a long-lived server.

Open the exported HTML directly, or serve the live repository dashboard:

```bash
ROSCLAW_EVOLUTION_ARENA_REPORT=/code/rosclaw/phase3-evidence/phase3-run.json \
  .venv/bin/python \
  -m rosclaw.entrypoint dashboard --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765/evolution-arena`.

An existing run can be exported again:

```bash
.venv/bin/python \
  -m rosclaw.entrypoint evolution export \
  /code/rosclaw/phase3-evidence \
  --format showcase \
  --output /code/rosclaw/phase3-showcase
```

## Run Gazebo process chaos

Build and validate the Fortress image first:

```bash
.agents/skills/rosclaw-simforge/scripts/verify_gazebo.sh
```

The verifier runs the real diff-drive, odometry, laser, command-source,
deadman, and bridge processes under `launch_testing`. The canonical experiment
then adds MCP, rosclawd, lease/session handling, rosbridge loss, reconnection,
and receipt verification:

```bash
.venv/bin/python \
  -m rosclaw.entrypoint chaos run gazebo-guarded-base \
  --faults agent-kill,rosbridge-loss,odom-stale,worker-crash \
  --output-dir /code/rosclaw/gazebo-guarded-base-evidence
```

The run fails unless all four named faults are requested. It uses actual
`SIGKILL`/`SIGTERM`, requires a bounded deadman stop, limits an
observation-loss receipt to `DISPATCH_CONFIRMED`, rebinds fresh rosbridge
handles after restart, requires a new Action ID, and verifies that the old
action is not replayed.

## Evidence interpretation

The primary machine-readable files are:

- `phase3-run.json` — complete ContactPush result and all identity hashes;
- `00-failure-router-acceptance.json` — all eight labeled routes and retry
  limits;
- `01-flywheel/dataset/snapshot.json` — public Dataset Snapshot;
- `04-evaluation-champion/evaluation.json` — Validation and signed Holdout
  aggregates;
- `03-four-gpu-champion/summary.json` and
  `contact-push-stress-signing-public.pem` — signed stress aggregate and the
  pinned run trust key;
- `11-final-proofs/proof-bundle-final.json` — derived module evidence tree;
- `10-activation/activation-canary-rollback.json` — D8/D9 and rollback chain;
- `raw-evidence-manifest.json` — hashes, byte sizes, and modes for the complete
  raw run tree;
- `showcase/hashes.json` — exported presentation artifact hashes;
- `gazebo-guarded-base-report.json` and `hashes.json` — Gazebo acceptance and
  sealed raw-evidence manifest.

Private Holdout inputs and signing keys are mode `0600`. They are never copied
into the showcase or committed to Git. A missing shard, identity mismatch,
invalid signature, data leakage, safety regression, stale observation, or
insufficient evidence fails closed.
