# ROSClaw SimForge Phase 3 implementation report

Date: 2026-07-23 (Asia/Shanghai)

Rebased onto: `2d4aa89f4d89e1ba2a7b2055a80a9c3a5a2a6cdf`

Branch: `simforge-phase3`

Scope: local simulation, four-GPU stress, ROS 2, and Gazebo validation only.
No real robot was connected or commanded. Unrelated GPU processes were left
running. Raw evidence is outside the Git checkout.

## Outcome

The three Phase 3 priority lines are complete:

1. ContactPush now executes a causal failure-to-success data flywheel rather
   than a preselected second parameter set.
2. Gazebo GuardedBase now exercises a real diff-drive, odometry, laser,
   deadman, canonical MCP-to-rosclawd action path, and actual process faults.
3. A statistically promoted but targeted-regression Candidate is activated as
   a Canary, frozen, rolled back, and followed by a successful retry through
   the restored active Champion.

The final signed flagship full run completed in 295.00 seconds and returned
`SIM_CHAMPION`. Its machine report is:

```text
/code/rosclaw/phase3_postrebase_full_20260723_signed/phase3-run.json
```

This run was repeated after Phase 2 PR #101 introduced trusted Gate inputs.
Validation and Counterexample Regression are now sealed only after physics
receipt verification, hidden Holdout remains independently signed, and the
four-GPU summary is Ed25519-signed against a pinned run public key before it
can become process-sealed `StressEvidence`. Caller-supplied counts and booleans
cannot qualify a Candidate.

The Gazebo canonical and process-chaos report is:

```text
/code/rosclaw/phase3_evidence_20260723_full_v2/13-gazebo-canonical-v5/gazebo-guarded-base-report.json
```

## Implemented contracts and runtime paths

Phase 3 adds:

- evidence-derived `ModuleProof` E0-E5 levels and hash-bound `ProofBundle`;
- `FailureSignatureV2`, eight safety-first failure classes, bounded retry
  budgets, and a deterministic route-acceptance suite;
- immutable, Body-scoped `PracticeDatasetSnapshot` partitions with HMAC
  grouping, label provenance, leakage checks, and private mode-0600 Holdout;
- real MuJoCo ContactPush contact, force, trajectory-state, and strict-replay
  evidence;
- Memory ON/OFF, Know ON/OFF, executable How, parameter/trajectory/Skill
  Graph/learned-policy candidates, Provider counterfactuals, and Darwin
  with/without Holdout;
- a process-isolated Ed25519-signed hidden evaluator;
- exactly four physical MJWarp shards with identity, completeness, dedicated
  Ed25519 signature, pinned-key, and tamper checks;
- a body-scoped simulation Champion Registry with hash-chained activation,
  ordinary use, Canary, freeze, rollback, and ledger verification;
- `/evolution-arena`, before/after/split-screen video export, receipts,
  self-contained HTML, technical JSON, and hash manifests;
- a Gazebo Fortress SDF with differential drive, odometry, GPU laser, command
  heartbeat, and deadman;
- `launch_testing` process supervision plus actual `SIGKILL`/`SIGTERM` for
  Agent, worker, odometry bridge, and rosbridge faults;
- fresh rosbridge-generation rebinding, new Action ID recovery, and old-action
  replay prevention;
- a sealed raw-evidence manifest and updated product capability truth source.

The product CLI routes are:

```text
rosclaw demo run failure-to-success
rosclaw proof run module-causal-v1
rosclaw proof show
rosclaw evolution export
rosclaw chaos run gazebo-guarded-base
```

Champion activation and rollback are intentionally kept inside the
evidence-bound demo transaction: direct standalone activation without the
matching Candidate, Dataset, Body, evaluation, stress, and promotion
identities is not exposed.

## ContactPush causal results

The immutable Practice Dataset contains 120 independently verified,
strict-replay MuJoCo episodes:

| Partition | Rows | Public rows disclosed |
| --- | ---: | --- |
| Development | 84 | yes |
| Validation | 18 | yes |
| Holdout | 18 | no |

All four quality rates are 1.0 and split leakage is false. The trained
contextual policy references the exact Dataset Snapshot hash.

The flagship fixed policy overshot on a low-friction scenario. Memory OFF
needed four bounded search attempts; Memory ON used a body-scoped recalled
recovery and succeeded in one attempt. The executable How patch then succeeded
on the exact same scenario, seed, initial state, and Body hash.

| Causal check | Result |
| --- | ---: |
| Failure capture, predefined eight-class suite | 100% |
| Failure routing accuracy | 100% |
| Unrecoverable stop rate | 100% |
| Infinite retry count | 0 |
| Memory attempts, OFF → ON | 4 → 1 |
| Wrong/stale-memory harmful retrievals | 0 / 2 |
| Wrong-memory hurt rate | 0% |
| Know invalid candidates removed | 2 |
| Know safety overrides admitted | 0 |
| Same-seed retry | failure → success |

Provider output was also causal: a valid, high-confidence output selected the
learned policy and changed MuJoCo failure to success; timeout routed to safe
stop, illegal output was rejected, and low confidence was rejected. Strict
replay matched.

Candidate A was not promoted: real MuJoCo contact reached 34.426 N against a
30 N limit, and the physics Sandbox returned
`FORCE_LIMIT_EXCEEDED`. Candidate B produced:

| Evaluation | Baseline success | Candidate success | Unsafe allow |
| --- | ---: | ---: | ---: |
| Validation, 200 paired episodes | 0% | 100% | 0% |
| Hidden Holdout, 200 paired episodes | 0% | 99% | 0% |
| Counterexample Regression, 20 pairs | 0% | 100% | 0% |

Statistical Gate G1-G14 and every Phase 3 causal gate passed. The four-GPU
stress used devices `0,1,2,3`, 1,000 unique scenarios, and 1,250,000
world-steps. All shards were complete and finite, exact CPU/MJWarp label
agreement was at least 99.6%, and critical backend disagreements and CPU force
violations were both zero.

The active-slot test then proved D8/D9: an ordinary task resolved and executed
the promoted learned Candidate. A second Candidate scored 97% Validation and
95% Holdout and was independently promoted, but a targeted 10-episode Canary
scored 0%. It was automatically frozen and rolled back; the restored Champion
succeeded on the same Canary scenario. The final active hash equals the
original Champion, the receipt ledger verifies, and a wrong-Body slot remains
empty.

## Module evidence

The final proof tree is:

| Module | Level | Decision impact | Strict replay |
| --- | --- | --- | --- |
| Body | E4 | yes | no |
| Provider | E5 | yes | yes |
| Failure Router | E5 | yes | yes |
| Sandbox | E5 | yes | yes |
| Practice | E5 | yes | yes |
| Memory | E4 | yes | no |
| Know | E4 | yes | no |
| How | E4 | yes | no |
| Auto | E4 | yes | no |
| Darwin | E5 | yes | yes |
| Registry | E5 | yes | yes |

E4 modules passed matched counterfactual and fault-injection evidence but are
not relabeled E5 when they do not have a strict-replay claim. The promotion
gate requires Body, Provider, Failure Router, Sandbox, Practice, Memory, Know,
How, Auto, and Darwin at E3 or above; Sandbox, Practice, and Darwin must be E5
before activation.

## Gazebo and ROS 2 results

The final Gazebo run passed all 13 acceptance predicates. The canonical route
was:

```text
MCP RuntimeClient → rosclawd → ActionGateway → daemon-owned ROS sink
→ rosbridge → ROS 2 → Gazebo Fortress → odometry/laser → TASK_VERIFIED
```

The 0.2 m/s by 0.5 s command produced 0.0992 m observed displacement against
0.1 m expected and stopped at zero velocity. The laser delivered 181 samples,
including 34 finite ranges and a 1.641 m nearest obstacle.

Fault evidence:

| Fault | Observed result |
| --- | --- |
| rosbridge `SIGKILL` | deadman stop in 0.589 s; maximum evidence `DISPATCH_CONFIRMED`; never `TASK_VERIFIED` |
| rosbridge restart | fresh southbound handles, new Action ID, new session required, zero old-action replay |
| Agent `SIGKILL` | session loss, orphaned receipt, physically observed E-stop, bounded stop in 0.546 s |
| launch Agent `SIGKILL` | deadman stop in 0.421 s |
| worker crash | real exit code 73, bounded stop in 0.749 s |
| odometry bridge kill | no fresh observation and maximum truthful evidence `DISPATCH_CONFIRMED` |

`launch_testing` supervised six real processes. The Gazebo proof bundle derives
rosclawd E5 and Runtime E4. The raw manifest contains 22 files and was
independently recomputed with zero size or hash mismatches. No managed
`rosclaw-phase3-gazebo-*` container remained after the run.

An earlier canonical attempt reused a severed WebSocket after rosbridge
restart and correctly failed recovery. That negative artifact is retained at
`13-gazebo-canonical/gazebo-guarded-base-failed.json`; the passing code binds
every command, stop, and observation handle to the new connection generation.
A later proof attempt exposed that missing executor evidence is truthfully
`FAILED/EXECUTOR_UNAVAILABLE`, not `BLOCKED`; the acceptance predicate was
corrected rather than weakening the runtime.

## Evidence identities

| Artifact | SHA-256 or content identity |
| --- | --- |
| Signed ContactPush machine report | `10c6669fe1148047540cd9b36c036c2a0464174791d8d27f3cd4b07c3457cd3e` |
| Raw signed ContactPush evidence manifest | `de57874ddc24672954573bf60017cbd31e7f6dab5a8ab2f8e4a18c2095813ecb` |
| Dataset Snapshot | `sha256:9da9feb383bbd6aaa414296803abd171003c0cbf34e4c079f2ca8a641d11df32` |
| Learned Candidate | `sha256:e9534305e6e4edbad14965b2803d4d8ed3c4b90028bb8fa4192245a3561918e8` |
| Provider causal output | `sha256:726e4d537d0e26d7764c6f20112b495c68f3911c2e4fce740a85b7006e4e5724` |
| Signed four-GPU stress source | `sha256:e99ca8466845e58118da348df068e0991fe4cd50f53839f0a160c75383da8420` |
| Four-GPU unsigned summary commitment | `sha256:27c427d89a1a741deca07173ee74148c413e4641220a1df12284836ef7c24299` |
| Four-GPU trust-key fingerprint | `sha256:1d7e046c180755c2d6c6e69f62d1fbb4a6a0719a8917cff3ddc2c2f838b84eb5` |
| Final logical ProofBundle | `sha256:9c5abc6dbd7a285acc5f0defa440badac914e1498ab7c2a784540a9929e0db5a` |
| Separately sealed showcase HTML | `0e8f2e650efe0822bcc44079f351f01a2dd1729b47def85cd3c8d272ca566963` |
| Gazebo report | `f6e8edd0d7fd08f34aabd659f4329d4b58a56ad2806c82a07d84842acc4fdfdc` |
| Gazebo logical ProofBundle | `sha256:0843dba7baea7ecb2519e5b90366ff830ee30e2be193765cace411bc69ae0130` |
| Gazebo raw manifest | `b0f5d522f2962e7664c33c2cc7d2e38ee9ae3e6f7e0d96702bc03107c3596f0b` |
| Gazebo launch-testing result | `2013f62f4b9249357420bc31b88228177e146b85c2a1410c684e6a4a4e324613` |

The signed ContactPush raw manifest covers 5,474 files and 88,795,095 bytes
with zero
recomputed mismatches. The showcase is separately sealed because it is
generated after the raw run. Private Holdout files and signing keys are mode
`0600`.

## Verification

Final repository validation:

- Ruff over changed source, tests, and scripts: passed;
- mypy over all 40 `src/rosclaw/simforge` source files: no issues;
- CoreBench schema validation: all five core tasks and the
  `failure_to_success_v1/contact_push` suite valid;
- focused Phase 3 contracts: 15 passed, including signed-stress tamper
  rejection;
- focused SimForge/Product/Dashboard regression: passed;
- full repository suite: 4,996 passed, 59 skipped, 25 deselected, 0 failed.

The first post-rebase full-suite run had four LeRobot integration failures:
collection saw the user-level ready runtime, while the isolated test home did
not inherit that path. Binding the fixture explicitly to the already verified
LeRobot 0.6.1 interpreter made all four targeted tests pass; the full suite was
then rerun from start to the green result above. No product-code skip or
fallback was added.

An earlier pre-rebase run also diagnosed six SeekDB filter failures caused by
an external interpreter drifting to pyseekdb 1.4.0 while `pyproject.toml`
pins 1.3.0. Restoring the pinned dependency made those tests and that full
suite pass. No source workaround for unpinned 1.4.0 behavior was added.

## Evidence ceiling and remaining work

- `SIM_CHAMPION` applies only to the bounded ContactPush simulation task and
  does not authorize hardware.
- The Body path proves hash-scoped invalidation and prevents old-skill reuse;
  automatic geometry adaptation and re-promotion for a changed Body remain a
  separate future milestone.
- Gazebo proves bounded guarded motion and process recovery, not a complete
  Nav2 checkpoint mission.
- Provider causal routing is covered; the lower-priority active-perception
  camera-view Demo is not part of this change.
- External independent Agent H5 acceptance and real robot tests remain
  unclaimed.
- The recommended standalone Practice/train/activation convenience commands
  are represented by the single evidence-bound flagship transaction; unsafe
  unbound Champion activation is intentionally not exposed.

Falsification that finds no counterexample is not a mathematical proof of
safety. A discovered counterexample is retained as replayable negative
evidence.
