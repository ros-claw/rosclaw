# Example Workflow: Desktop Pick-and-Place

This page walks through a complete ROSClaw closed-loop example: teaching a robot to pick a red cube from a table and place it in a bin.

---

## The Task

```text
Agent: "Pick the red cube from the table and place it in the bin."
```

## Closed Loop

```text
Agent intent
  ↓
Provider selects the pick-and-place skill for the current body.
  ↓
Sandbox validates the trajectory against the e-URDF and safety limits.
  ↓
Runtime dispatches the validated trajectory.
  ↓
Practice records the episode: video, state, events, outcome.
  ↓
Memory indexes the experience for similar future tasks.
  ↓
How intervenes if the grasp fails and requests a retry pattern.
  ↓
Know compiles a TaskCard: "red cube, glossy surface, parallel jaw grip."
  ↓
Auto proposes a grip-force patch.
  ↓
Darwin evaluates the patch across 100 simulated seeds.
  ↓
Promotion gate moves the patch to champion if it improves success rate.
```

## Commands

```bash
# 1. Install and first boot
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot

# 2. Link the robot body
rosclaw body init --robot sim_ur5e

# 3. Run the skill in simulation
rosclaw sandbox run --robot sim_ur5e --world tabletop --task pick_red_cube

# 4. Inspect the recorded episode
rosclaw practice list
rosclaw practice show <last-episode-id>
```

## What happens under the hood

1. **Body Context**: ROSClaw loads the effective body model for `sim_ur5e`, including joint limits, gripper geometry, and camera calibration.
2. **Capability Routing**: The Provider picks a `pick_red_cube` skill candidate and checks compatibility.
3. **Sandbox Validation**: The trajectory is pre-played in MuJoCo against the e-URDF. If a collision is predicted, the action is blocked or modified.
4. **Execution**: The validated trajectory is dispatched to the simulation.
5. **Practice Capture**: The episode is recorded as MCAP + JSONL events.
6. **Memory**: The failure or success pattern is indexed for recall.
7. **How Intervention**: If the grasp slips, How returns an injection with a suggested force or approach adjustment.
8. **Knowledge Compilation**: Know turns the episode into a TaskCard with constraints and evidence.
9. **Auto Evolution**: Auto generates a patch and proposes it for evaluation.
10. **Darwin Evaluation**: The patch is stress-tested across many seeds.
11. **Promotion Gate**: Only patches that improve the success rate become the new champion skill.

## See Also

- [QUICKSTART.md](../QUICKSTART.md) — Get started with ROSClaw.
- [docs/SAFETY.md](SAFETY.md) — Safety model and deployment rules.
- [ARCHITECTURE.md](../ARCHITECTURE.md) — Runtime architecture.
