# ROSClaw SimForge Phase 4 implementation report

Date: 2026-07-23 (Asia/Shanghai)

Base: `origin/main` at
`2d4aa89f4d89e1ba2a7b2055a80a9c3a5a2a6cdf`, through the Phase 3 commit
`56281ed3685dce36cceb20ac19b3d22e1267d732`

Branch: `simforge-phase4`

Scope: local simulation, four-A6000 CUDA screening, CPU MuJoCo physical
verification, and isolated Unitree DDS loopback. No real robot transport was
opened. Existing unrelated GPU workloads were not stopped. Raw trajectories,
Private Holdout rows, signatures, and receipts are outside the Git checkout.

## Outcome

Phase 4 implements the G1 GoalForge failure-to-success loop:

```text
MuJoCo failure
→ FailureSignatureV3
→ Memory / Know / How
→ bounded Auto Candidate
→ Sandbox
→ same-seed physical retry
→ Twin residual update
→ Practice Dataset Snapshot
→ learned Shot Adapter
→ signed Private Holdout
→ Darwin / Registry
→ ordinary Champion use
→ Promotion Gate V4
```

The final Promotion decision is `SIM_CHAMPION`. G1-G16 all pass. This is a
simulation qualification only and grants no hardware authorization.

The final machine report is:

```text
/tmp/rosclaw-goalforge-promotion-v7.qwiU3n.json
```

Its evidence hash is
`sha256:24c8c0536cd7d0d7a390b59a0940667457451104e7026d0222e02e25718800d7`.

## Public-project qualification

The external projects are retained outside ROSClaw and are referenced by
commit and content hash:

| Project | Qualified commit | Use |
| --- | --- | --- |
| Unitree `unitree_rl_mjlab` | `1425b15f73bd4095f0df53709d7c389c3eb9e790` | G1 motion-imitation and multi-GPU contract |
| Unitree `unitree_mujoco` | `ae6a8403e272733e9996ef59990880330496177f` | official 29-DoF model and SDK2/DDS semantics |
| Unitree `unitree_sim_isaaclab` | `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | optional visual/DDS second-backend contract |
| OpenDriveLab RoboNaldo Deploy | `f60f24459aaabc3aea9187a2b13f8923049b629c` | 29-joint G1 kick prior, motion, and ball scene |

The qualified Body hash is
`sha256:1525aafc953293dd340475e8660b58fae7ea1a22715222ab06331b644340626c`.
The fixed kick-prior hash is
`sha256:b952af481c47402c40a883ea214635368b44ee36de356780258d04601eaa3774`.

The Doctor verified four distinct RTX A6000 UUIDs, MuJoCo, ONNX Runtime,
PyTorch, Unitree SDK2 Python, CycloneDDS, the official Unitree model, and the
external source contracts. `real_hardware_opened` is false. The optional Isaac
Lab checkout contract passes, but the Isaac Lab runtime is not installed and
is not represented as executed.

Doctor report:

```text
/tmp/rosclaw-g1-doctor-v5.9LZ3G4.json
```

## Flagship physical demo

The public demo runs the fixed RoboNaldo ONNX policy in the G1-with-ball
MuJoCo scene at 500 Hz physics and 50 Hz policy control. Each of its three
shots executed 6,520 physics steps and produced independently verified,
strict-replay receipts.

| Shot | Status | Target error | Support slip | Fall / torque violation |
| --- | --- | ---: | ---: | --- |
| fixed prior | `TARGET_MISS_RIGHT` | 0.505 m | 0.0308 m | 0 / 0 |
| same-seed bounded retry | `SUCCESS` | 0.418 m | 0.0125 m | 0 / 0 |
| new-location first shot | `SUCCESS` | 0.430 m | 0.0272 m | 0 / 0 |

The failure router classified the first shot as a recoverable target error,
assigned a retry budget of two, and routed it through Memory, Know, How, and
Auto. The same scenario and seed were preserved for the retry. An aggressive
wrong Candidate was rejected.

Demo report:

```text
/tmp/rosclaw-goalforge-demo-v5.52SHcx/run/goalforge-demo.json
```

## Practice-to-policy flywheel

The final acceptance flywheel uses twelve distinct physical scenarios:

| Partition | Distinct scenarios | Physical Practice records | Public rows |
| --- | ---: | ---: | --- |
| Development | 8 | 16 | yes |
| Validation | 2 | 4 | yes |
| Private Holdout | 2 | 4 | no |

Baseline and Candidate records from the same scenario count as one teacher
context, not two training samples. The learned `G1ShotAdapter` therefore has
eight distinct teacher contexts. Every output is bound to Dataset Snapshot
`sha256:17862702d7e4665210f9b501baecbcfa3419a5e3cc484fd26638c5ba1cf9d24e`
and model
`sha256:b8440136986d01b702c099d79b7d330ae5e24565b885b93d9365ed58baf2b38a`.

All Practice records are complete, independently verified, hash-complete, and
strict replayable; all four rates are 1.0 and split leakage is false.

| Comparison | Result |
| --- | ---: |
| Fixed Prior success on Validation + Holdout | 75% |
| Learned Adapter success on Validation + Holdout | 100% |
| Online teacher success | 100% |
| Learned fall / torque violation | 0% / 0% |
| Mean learned inference | 0.126 ms |
| Mean online search | 7,638 ms |

The Ed25519-signed Holdout aggregate contains two undisclosed scenarios:
100% goal and target-zone success, 0.354 m mean target error, positive
0.042 m mean COM margin, zero fall, zero joint-limit violation, and zero
torque violation. Ordinary task execution resolved the activated Champion
through the Registry and physically succeeded.

Flywheel report:

```text
/tmp/rosclaw-goalforge-flywheel-v6.oDqvnI/run/goalforge-flywheel.json
```

## Recovery, Memory, four-GPU, and continual evidence

The dense CPU MuJoCo recovery validator attempted 115 scenarios to collect
100 matched failure-recovery pairs. It executed 220 physical episodes,
retained every Sandbox attempt, and did not hide five unsafe first Candidates.

| Recovery metric | Result |
| --- | ---: |
| Recoverable failure capture | 100% |
| Final same-seed Retry success | 100% |
| Unrecoverable stop | 100% |
| Infinite retries | 0 |
| Independent verification | 100% |
| Final safe executions | 100% |

Memory ablation used 100 ON/OFF pairs. Mean search attempts changed from 5.03
to 1.00, an 80.12% reduction. Wrong-memory hurt rate and wrong-Body reuse were
both zero.

The CUDA screen used all four physical A6000 UUIDs and 1,000 independent
scenario commitments, 250 per GPU, with complete signed shards and G0-G10
coverage. Private Holdout case rows were not disclosed. CUDA output is
screening evidence, not physical truth. A disjoint balanced CPU MuJoCo check
executed 24 scenarios after excluding 24 calibration rows and achieved 95.83%
agreement for both safety and success labels. The one disagreement is retained
as a torso-overshoot counterexample.

The G0-G10 continual screen improved first-attempt success from 0.67% to 2.0%
and reduced mean retries from 0.993 to 0.980. These are conservative
`CUDA_SCREENING` proxy rates, not MuJoCo policy-training claims. Critical
safety forgetting is zero and mean historical success delta is +0.994
percentage points.

Evidence:

```text
/tmp/rosclaw-g1-recovery-100-v4c.ugdk2D/recovery-100.json
/tmp/rosclaw-g1-memory-ablation-v4.tiyd1U.json
/tmp/rosclaw-g1-four-gpu-v3.J9hnZC/run/four-gpu-summary.json
/tmp/rosclaw-g1-cpu-gpu-agreement-v3.7nWQc9.json
/tmp/rosclaw-g1-continual-v3.hzr25u.json
```

## DDS canonical execution and chaos

The isolated DDS path uses the official Unitree `unitree_hg` `LowCmd` and
`LowState` messages on `lo`, DDS domain 76, and canonical `rt/lowcmd` and
`rt/lowstate` topics. It observed 50 commands, 382 states, 490 physics steps,
29 actuators, finite state, and IMU feedback.

All eleven fault cases passed, including Agent kill, worker crash, DDS loss,
LowState/IMU stale, policy timeout, cancel before kick, cancel during
recovery, stale verification, daemon restart, and a fresh post-restart
session. Old trigger replay and stale `TASK_VERIFIED` counts are both zero.
Every permit is `SHADOW`, and hardware authorization is false.

Chaos report:

```text
/tmp/rosclaw-goalforge-chaos-v4.5tp1pt/run/goalforge-chaos.json
```

## Proof and Promotion hardening

The final Proof Bundle contains Body, Provider, Failure Router, Sandbox,
Practice, Memory, Know, How, Auto, Darwin, Registry, and rosclawd at E5.
Independent replay recomputes the bundle hash and re-derives each level from
matched counterfactual, fault-injection, and replay primitives. It does not
trust a stored `level: E5` string.

Proof Bundle:

```text
/tmp/rosclaw-goalforge-proofs-v7.MwzD7u/run/proof-bundle-final.json
```

Logical bundle hash:
`sha256:d8ed1c48fd3187641cd9fbea5e4c4323b89273f37bd95b17b8d6575cce5200a9`.

Promotion was tightened during final review:

- G3 requires signed Holdout task improvement and zero safety violations;
- G13 uses a separate Doctor report and checks Doctor, Champion, and Proof
  Body hashes plus the Doctor/Champion kick-prior hash;
- G15 requires all twelve modules at replay-verified E5;
- missing or mismatched qualification evidence fails closed.

## Showcase

The dependency-free showcase contains 652 recorded trajectory frames for each
of the baseline, same-seed retry, and new-location shots. It renders ball
paths, COM and support-slip overlays, the causal module chain, verifier
metrics, receipt hashes, and Champion evidence.

```text
/tmp/rosclaw-goalforge-showcase-v4.H3wDza/export/index.html
/tmp/rosclaw-goalforge-showcase-v4.H3wDza/export/showcase.json
```

The HTML JavaScript passed `node --check`.

Phase 4 also provides a real MuJoCo offscreen video export. It reconstructs the
recorded pelvis pose, all 29 joint positions, and the ball pose in the
qualified scene, with contact slow motion, a target marker, ball trail, and
verifier-derived subtitles. The final H.264 artifact is 1280x720, 30 fps,
36.2 seconds, and 1,086 frames:

```text
/tmp/rosclaw-goalforge-video-final.4HJQqk/g1-goalforge.mp4
/tmp/rosclaw-goalforge-video-final.4HJQqk/g1-goalforge.json
```

Video hash:
`sha256:68dc4e7775437a7b3fbd695f54994f5304166fb23d7c62861cb0c83b68f563a3`.

Both exports are recorded-evidence visualizations and cannot produce task
labels or Promotion evidence.

## Phase 4.1 nominal-success optimization

On 2026-07-24, the bounded Shot Adapter search was extended with coupled
stance/pelvis/foot lateral candidates while retaining the same immutable
whole-body prior and safety limits. The new balanced CPU MuJoCo validation
contains 30 static-ball cases: three ball offsets, five goal columns, and two
reachable heights.

| 30-case CPU MuJoCo metric | Fixed | Context-only | Optimized |
| --- | ---: | ---: | ---: |
| Success rate | 53.33% | 60.00% | 100.00% |
| Mean target error | 0.5025 m | — | 0.2441 m |

All 30 selected candidates were safe, independently trajectory-verified, and
bit-for-bit strict replayable. Mean target error decreased by 51.4%. The
validator report passed its 95% success, +30 percentage-point improvement,
zero-unsafe-selection, independent-verification, and replay gates.

```text
/tmp/rosclaw-goalforge-success-v2-final.vsiSi5/nominal-success-30.json
```

File SHA-256:
`e4c0061ab19177dc624476243616f1200a52636eab5bfd7c56a961997e229d6d`.

A fresh four-A6000 screening run also passed with four distinct GPU UUIDs,
1,000 scenario commitments, and complete G0-G10 coverage. Its disjoint CPU
check executed 24 physics cases and retained the observed 79.17% success-label
and 87.50% safety-label agreement (83.33% mean); this conservative broad
screen is not used as CPU task truth. Memory ON/OFF ablation again reduced
mean attempts from 5.03 to 1.00 (80.12%) without wrong-memory harm.

```text
/tmp/rosclaw-goalforge-4gpu-v2.J6tHBa/run
```

The high-success claim is intentionally limited to the nominal grid: static
ball, nominal mass/friction, zero latency/noise/disturbance, lateral targets
within ±0.75 m, and target heights of 0.20 m or 0.55 m. Exploratory broad
randomized disturbances and the 0.90 m target are retained as open failure
modes rather than included in the 100% figure.

The refreshed four-shot H.264 replay adds the difficult edge-angle success
after the original miss/retry/new-location sequence. It is 1280x720 at 30 fps,
48.27 seconds, and 1,448 frames. The optimized shot reached the opposite goal
edge with 0.337 m target error, 0.0021 m support slip, no fall, and no torque
violation.

```text
/tmp/rosclaw-goalforge-video-v2-final-20260724/g1-goalforge-optimized.mp4
/tmp/rosclaw-goalforge-video-v2-final-20260724/g1-goalforge-optimized.json
```

Video SHA-256:
`f7a3b7d34b60f561a3b02cbd4ed2519c0d71cbba1b2e05db1953845baea7593e`.

## Verification

Final repository verification:

- Phase 4 contract tests: 16 passed;
- real GoalForge MuJoCo/video + Unitree DDS integration tests: 2 passed in
  86.94 s;
- full repository suite with the verified LeRobot 0.6.1 interpreter:
  5,012 passed, 59 skipped, 27 deselected, 0 failed in 984.19 s;
- Ruff over 52 Phase 4 Python files: passed;
- Ruff format over the same 52 files: passed;
- mypy over 49 Phase 4 source files: no issues;
- repository-prescribed mypy over 118 source files: no issues;
- `compileall` over `src` and `tests`: passed;
- Practice suite: 175 passed, 4 skipped;
- `git diff --check`: passed.

The first full-suite run had four LeRobot integration failures because test
collection saw the user-level external runtime while the isolated test home
did not inherit its path. Re-running with
`ROSCLAW_TEST_LEROBOT_PYTHON` bound to the already inspected LeRobot 0.6.1
interpreter produced the clean full-suite result. No source skip or fallback
was added.

Whole-repository Ruff still reports 244 pre-existing issues under the untouched
`examples/rh56_rps` tree. Whole-repository Ruff format also reports 110
pre-existing unformatted files, including that example and other untouched
paths. Whole-`src` mypy still reports six pre-existing errors in
`practice/bridges/sandbox_bridge.py` and `practice/episode_recorder.py`. The
scoped Phase 4 checks are clean; these unrelated files were not modified.

## Evidence identities

| Artifact | File SHA-256 |
| --- | --- |
| Doctor | `d947d57367197a71e9cce392c523a13824ea14cbf8dbab0c81c7fd2fda7e1e89` |
| Flagship demo | `b9cf7187ab6a5289cd663ba02afa6fe1046125b96fc6c0dc31a2a5e19ee50e87` |
| Practice flywheel | `f4685a34b8d9c568075ab0a0f77a65b78192f9ad734e5614c6fe5148c6dc9426` |
| 100-pair recovery | `9cc24de84ca7552259c99f19caf62207712e7a98591c747c8b0971907dead10b` |
| Four-GPU summary | `80d23af41baa094113081e5594e047fcdfeb9ae082aee2cf7db8e1374ba95eda` |
| CPU/GPU agreement | `a86a8b6d4c5878eddb96eceba8309920466f503dd72cf0dbd4588d870c988f6d` |
| Memory ablation | `df2df14054f81eabc8f5edeea28a867609e4e8d4b5cb81a5bb92dc1587ac9122` |
| Continual G0-G10 | `681a7ddbfb0c0533eb9ef17dfc2df0145603b2ed543bc786ab25d641195fb9f0` |
| DDS/chaos | `7faf6645c08376474f2d722f7ad6098d0a2679825667f4b0917061a4789a5527` |
| Proof Bundle file | `dc2872b29aeee5dd6d51ebd3d1d0335651cfe7a85f63a75da76959d44e324a4a` |
| Promotion report | `774013bfe3f709eb6c4e00bbb264ae6fe824aa5bb747ce9472399622d81619d7` |
| Showcase HTML | `92961ecbbeaef8f19e59b1f544ff61f867af8d8854574231a9d291d1ee2b4074` |
| GoalForge H.264 video | `68dc4e7775437a7b3fbd695f54994f5304166fb23d7c62861cb0c83b68f563a3` |
| Nominal-success 30 | `e4c0061ab19177dc624476243616f1200a52636eab5bfd7c56a961997e229d6d` |
| Fresh four-GPU summary | `c2d90ba55faa2f711edb3a53a8a49968d6a40721ae510ee2efa90259c338bb3e` |
| Refreshed four-shot video | `f7a3b7d34b60f561a3b02cbd4ed2519c0d71cbba1b2e05db1953845baea7593e` |

## Evidence ceiling

- `SIM_CHAMPION` is limited to the qualified RoboNaldo/Unitree G1 simulation
  assets and bounded Shot Adapter parameters.
- CPU MuJoCo provides task truth. Four-GPU CUDA results are prioritization,
  falsification, and Holdout-screening evidence.
- The two-scenario learned-policy Private Holdout is intentionally modest; the
  separate four-GPU Private Holdout has 250 undisclosed scenarios, while final
  physical validation has 100 paired cases.
- Isaac Lab execution, camera-based perception, hardware latency, real turf,
  actuator thermal behavior, and robot falls are not proven.
- No real G1 was connected, commanded, or authorized.
- Falsification cannot prove mathematical safety. Retained counterexamples and
  simulation success do not replace supervised real-robot qualification.
