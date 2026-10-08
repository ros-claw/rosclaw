# N02 main planning candidates and paired SIM protocol

Status: five Waffle candidate pilot pairs have completed physical acceptance;
the latest stateless boundary candidate improves this pair below the30% target.
The first startup attempt failed before dispatch and remains retained. No frozen-series median or P0 closure claimed.
N01 predecessor PR #621 merged at8a9ae84; its actual gates are recorded separately.

The actual Waffle baseline uses 0.5 m headland, 0.45 m operation width and
0.5 m robot width. Burger uses 0.3 m headland and 0.3 m operation/robot widths.
The original offline Burger comparison erroneously used Waffle's 0.5 m
headland; the corrected snapshot explicitly supersedes it. No physical trial
used that erroneous reference.

The [corrected pinned Fields2Cover predictions](offline/n02-003/candidate-scores.json)
cover six variants for each robot, preserving the installed revision. Waffle
predicts 59.87% at baseline, 77.49% with 45-degree swaths and 76.17% with
0.45 m headland. Burger predicts 80.46% at its true baseline. Its diagonal
candidate predicts 91.05% but has negative conservative circle-to-wall
clearance and is rejected before dispatch. The larger 0.35 m Burger headland
predicts 78.71%, with more planned clearance. It may trade coverage for safety;
it is not claimed to be an efficiency improvement.

Only these known SIM fixtures are configured; this does not implement unseen
Body adaptation. Plan polyline clearance in the convex rectangular room is
bounded by endpoints; physical controller tracking is not proved by that
geometric fact. New candidates use a declared 50 mm offline clearance screen.
The unchanged Burger reference has only 24 mm planned clearance and fails that
new-candidate screen; its prior physical success is retained as historical
reference rather than reclassified as an optimized safe candidate.

`stack.py --seed` now passes the seed to the installed `gz sim --seed` option.
`--coverage-preset baseline|diagonal|headland` applies only the four declared
planning parameters after the existing profile setup. Body, brush, fixed
measured coverage denominator, speed, observation and stop guards are retained.
No upstream dependency upgrade or new actuator entry is introduced.

The [pretrial protocol](N02_EXPERIMENT_PROTOCOL.json) fixes five pilot seeds
and ten later evaluation seeds per robot. Each arm starts in a fresh owned
container, evidence directory, ROS domain, port and clean state. Arm order
alternates by seed parity. All failed episodes are retained. No evaluation
run is admitted until an explicit source/configuration freeze matches the
actual commit and candidate. Evaluation seeds cannot be used for tuning.

`paired_efficiency.py` launches this fixture and invokes the existing
`run.py` MCP request-action journey. It independently checks canonical replay,
Memory persistence/reopen, Practice, contacts, complete observations and three
seconds of post-cleanup standstill before recording a PASS. Shutdown flushes
passive streams; an incomplete audit fails the pair. Every pair records image
ID, source commit, protocol hash, measured distance, SIM span and wall mission
duration separately. One successful pair cannot establish median improvement.

Configuration preflight generates five configurations inside the unchanged
rebuilt image, with no ROS action. Exact robot/world/controller/map bytes and
all non-planning Nav2 settings match across presets; only temporary controller
YAML reference paths in SDF are normalized for comparison. Local ROS regression:
308 passed / 11 deselected after the portability and startup fixes; focused
protocol tests: 4 passed after adding the startup regression. Changed-file lint/format checks pass. These counts
are not summed.

Planned pilot command (clean committed worktree, output must not exist):

```bash
PYTHONPATH=src .venv/bin/python integrations/ros_probe/acceptance/paired_efficiency.py \
  --directory /tmp/ros-expert-next/n02-waffle-diagonal-100801 \
  --protocol docs/reports/ros-expert-harness/next/N02_EXPERIMENT_PROTOCOL.json \
  --profile waffle --candidate diagonal --seed 100801
```

Main coverage, bounded pose-aware repair optimization and their eventual paired
evaluation remain separate steps. The target is at least 98% measured coverage,
zero contact, complete stop/evidence gates and a 30% reduction in both median
observed distance and SIM span. Until actual frozen paired evaluation exists,
`v1_done=false` and efficiency acceptance remains open.

## Later candidate outcomes (same first pilot seed, independent pairs)

| Waffle candidate | Baseline distance / SIM | Candidate distance / SIM | Result |
| --- | --- | --- | --- |
| Headland0.45 |72.427m /697.60s|57.634m /595.24s|Main failed104; reductions20.43% /14.67%, rejected.|
| Five boundary through-poses targets |57.700m /531.52s|71.632m /671.64s|Main/boundary succeeded; primary76.90%; worse24.15% /26.36%, rejected.|
| Nine sequential boundary targets |63.734m /643.24s|63.712m /697.56s|Main failed105; boundary never exercised; reductions0.0346% /-8.4447%.|
| Nine sequential targets plus RPP stateless |73.275m /723.76s|63.183m /575.00s|Main/boundary succeeded; primary91.1633%; reductions13.7724% /20.5538%, below target.|

All10 physical runs retain original>=98% coverage, zero contacts/trace gaps,
complete audit/canonical replay, Practice/Memory and independent stop. Different
rows cannot be combined into a synthetic best baseline or improvement estimate.
Latest source1c107929188b939d965b311fe49a27a642bda5db is unchanged across its
two arms. Its candidate final coverage is98.1823%, but still uses23 repair goals.
The boundary segments sum to119.60SIMs after successful original main coverage;
measured primary coverage is distinct from original main57.5503%.
See [stateless physical results](runs/n02-waffle-stateless-100801/paired-efficiency-results.json)
and [unexercised sequential boundary results](runs/n02-waffle-sequential-100801/paired-efficiency-results.json).
Complete raw artifacts remain in the matching local evidence archives; SHA
manifests and per-stage SVG/JSON are committed beside each result.

Burger stateless seed100801 is running with the same frozen1c source. Its1800s
deadline is the existing documented Burger acceptance deadline, identical for
both arms; Waffle retains its existing900s deadline. None of the ten final
evaluation seeds has been consumed. Repair optimization remains a later PR.

The first `n02-waffle-diagonal-100801` startup was interrupted before any
MCP journey or action artifact. Readiness incorrectly waited for `snapshot.json`,
which `run.py` creates only after startup. The owned container was stopped and
all evidence retained. The corrected check requires fresh independent observer
state, a measured map and active navigation/coverage lifecycle logs. A regression
test proves startup does not depend on the task-created snapshot. A fresh
attempt at the same preregistered pilot seed is separate evidence; the failed
startup is not counted as a physical cleaning pass.

## First paired physical pilot, seed100801

| Measurement | Baseline | Diagonal candidate |
| --- | ---: | ---: |
| Final measured coverage | 98.0425% | 98.0425% |
| Main measured coverage | 57.0749% | 13.3949% |
| Observed distance | 62.713985 m | 64.987093 m |
| Observed SIM span | 573.36 s | 705.64 s |
| Repair goals | 26 | 40 |
| Contact / trace gaps | 0 / 0 | 0 / 0 |
| Audit / canonical replay / independent stop | PASS | PASS |

Frozen source1f41fd4b8dfc22d07392f351d284e534eafddba5, unchanged image,
Body, seed and world, independent fresh directories. The diagonal main goal
returns status6/error105 (`FAILED_TO_MAKE_PROGRESS` from installed FollowPath
contract), then the existing measured repair loop completes the room. Its
distance increases3.6246% and SIM span increases23.0710%. It is rejected.
Physical pair PASS and optimization FAIL are separate claims. This is one
pilot pair, not a median or completed five-seed pilot series.

[Run summaries, overlays and conclusion](runs/n02-waffle-diagonal-100801/README.md)
are committed. Complete public raw inputs, independent post-cleanup observations
and hashes are durably archived under
`/home/nvidia/workspace/rosclaw/rosclaw_harness/evidence/2026-10-08/n02-waffle-diagonal-100801`.
The actual cause of the progress stall remains unassigned: no collision-monitor
state/velocity observation was captured in this episode. Subsequent trials
passively record those existing topics, with no actuator, control or safety
parameter change. The canonical initial Nav2 result is also exposed in the
offline summary. Earlier absent observations are not invented retroactively.
