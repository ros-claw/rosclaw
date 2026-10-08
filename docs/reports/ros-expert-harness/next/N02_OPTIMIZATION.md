# N02 main planning candidates and paired SIM protocol

Status: candidate implementation and configuration tests PASS; physical pilot
NOT_RUN at initial commit. First startup attempt failed before dispatch. No efficiency improvement or P0 closure claimed.
N01 PR #621 remains the predecessor and must merge first.

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
305 passed / 11 deselected before the added owned-fixture failure test; focused
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

The first `n02-waffle-diagonal-100801` startup was interrupted before any
MCP journey or action artifact. Readiness incorrectly waited for `snapshot.json`,
which `run.py` creates only after startup. The owned container was stopped and
all evidence retained. The corrected check requires fresh independent observer
state, a measured map and active navigation/coverage lifecycle logs. A regression
test proves startup does not depend on the task-created snapshot. A fresh
attempt at the same preregistered pilot seed is separate evidence; the failed
startup is not counted as a physical cleaning pass.
