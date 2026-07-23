# ROSClaw SimForge Phase 2 implementation report

Date: 2026-07-23 (Asia/Shanghai)

Base: merged `main` commit `d79507b26669d3455855a429816f3b4cbfc0a1ff`

Scope: local simulation and ROS validation only. No real robot was connected.
Raw evidence is intentionally outside the Git checkout at
`/code/rosclaw/rosclaw_phase2_evidence` (about 40 MiB, including superseded
qualification runs retained for audit).

## Outcome

The Phase 2 implementation adds CoreBench schemas and five task definitions,
partitioned scenario generation, SeedLedger, robustness monitors, bounded
candidate search, paired statistics, a signed hidden-Holdout worker,
falsification/counterexample storage, cross-backend label comparison,
DataBudgetManager, Evolution state handling, Gate V3, and three `simforge` CLI
routes.

ShieldReach completed a fully automatic parameter evolution:

- generator: bounded cross-entropy plus generated coverage grid, 60 evaluations;
- parent threshold: 0.82; selected threshold: 0.50;
- candidate: `sha256:1826d82d9fdc160e42238d00a114488dca9053d1175ab30b8d0b39cb31157816`;
- all four human-involvement flags: false;
- Validation (200 paired MuJoCo episodes): success 0.64 → 1.00,
  unsafe-allow 0.36 → 0, false-block 0;
- hidden Holdout (200 paired MuJoCo episodes): success 0.66 → 1.00,
  unsafe-allow 0.34 → 0, false-block 0;
- Counterexample Regression (20 disjoint-seed collision episodes): complete,
  strict replay 100%, candidate unsafe-allow 0;
- G1–G14: all passed; decision: `SIM_CHAMPION`.

The final 1/2/4 A6000 MJWarp run used 256 worlds per GPU and 100 steps:

| GPUs | Worlds | World-steps | World-steps/s | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 25,600 | 500.51 | 1.000× |
| 2 | 512 | 51,200 | 982.44 | 1.963× |
| 4 | 1,024 | 102,400 | 1,899.18 | 3.794× |

All shards reported finite state, correct mixed physical collision labels, and
zero candidate unsafe allows. Across the full scaling curve, 1,792 worlds had
zero critical MuJoCo↔MJWarp collision-label disagreements. Missing-shard and
killed-worker campaigns both return incomplete evidence and a non-zero process
status; incomplete shards cannot promote.

The canonical ROS test issued real bounded motion in turtlesim through:

```text
MCP RuntimeClient → rosclawd Unix socket → ActionGateway
→ GenericMobileBaseSimulationExecutor → daemon-owned rosbridge sink
→ /turtle1/cmd_vel → /turtle1/pose → TASK_VERIFIED receipt
```

Direct Provider motion remains blocked. The observation-loss case returns only
`DISPATCH_CONFIRMED`. Gazebo Sim 6.18.0 was run headless and `ros_gz_bridge`
delivered a live `/clock` sample. ContactPush used real MuJoCo contact and force
data. ROS2Chaos generated 1,000 fault cases, and BodyMutation validated 1,000
expected accept/reject outcomes.

Isaac Lab ran `Isaac-Reach-UR10-Play` with 64 environments on GPU using the
kitless Newton/MJWarp backend: 1,920 finite env-steps at 1,908.71 env-steps/s.

Final verification on the repository-pinned dependency set:

- Ruff: passed;
- mypy: 22 source files, no issues;
- final targeted SimForge/sandbox regression: 58 passed, 1 deselected;
- live ROS integration: 7 passed;
- complete repository suite: 4,967 passed, 58 skipped, 25 deselected, 0 failed.

The first complete run found six embedded SeekDB failures because the local
environment had drifted to pyseekdb 1.4.0 while `pyproject.toml` pins 1.3.0.
Restoring 1.3.0 made all six pass, and the complete suite was then rerun to a
green result. No third-party 1.4.0 workaround was added to ROSClaw source.

## Evidence hashes

| Artifact | SHA-256 |
| --- | --- |
| ShieldReach evolution report | `f9b2a7fd4fc955c2dfeaacdab928947041ac3975ae7b602cd6bdbf6de0336c09` |
| 1/2/4 GPU scale curve | `6b42cd0eb572d4e0a4956496a238b528051014739f836e6c273a597691458ae3` |
| missing-shard rejection | `12226f90bada88dc1e93c000b5f615f7e02a08f05053703375b2a00905c01ab2` |
| killed-worker rejection | `c4c98051a6112ac360f3799bf7048576eb2f173cef8d147be37a430097fbe29f` |
| Isaac UR10 qualification | `b9bcca524bc445b91cb355fb497b0c978db6e84d852993fbc21400f264069fba` |

The final evolution directory includes 440 request artifacts and 440
independently replayed trajectory-state artifacts for Discovery, Validation,
Holdout, and Counterexample Regression.
Private Holdout cases and signing keys are mode 0600 and are not committed.

## Known boundaries

- `SIM_CHAMPION` applies to this bounded ShieldReach risk-threshold benchmark;
  it is not real-robot deployment approval or a formal safety proof.
- Gazebo qualification currently proves headless simulation and a real ROS
  bridge clock. Actual canonical base motion is proven in turtlesim; a Gazebo
  diff-drive body with odometry, laser, cancellation, and client-loss deadman is
  future work.
- ROS2Chaos currently validates evidence classification for generated faults;
  it is not yet a full `launch_testing` multi-process recovery campaign.
- Isaac PhysX was unavailable in the kitless environment; Newton/MJWarp was
  used as the second implementation. Newton reported an upstream UR10
  `ee_link` inertia approximation warning and a fixed-root velocity warning.
- The Isaac Hub client panicked on an EOF response, then safely fell back to
  NVIDIA's S3/CloudFront asset provider. The task completed, but the Hub panic
  should be treated as an upstream reliability issue.
- Deliberate OOM was not triggered because it could destabilize concurrent
  workloads. Missing-GPU and hard worker-crash paths were exercised instead.
- External RoboLab/ManiSkill/RoboCasa/BEHAVIOR suites are not part of this
  Phase 2 change set.

Falsification that finds no counterexample is not proof of safety. Every stored
counterexample remains content-addressed and replayable.
