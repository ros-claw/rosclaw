# Body Module Migration Guide

This guide explains how to migrate an existing ROSClaw workspace or runtime to
the three-layer body system introduced in `rosclaw-v1.0`.

## What Changed

| Before | After |
|--------|-------|
| `rosclaw robot *` commands managed ad-hoc robot state. | `rosclaw body *` commands manage a formal body model. |
| e-URDF profiles were loaded on demand by skills/providers. | e-URDF profiles are linked once via `rosclaw body link-eurdf`. |
| No centralized body state. | `~/.rosclaw/body/body.yaml` is the Body Instance Ledger. |
| Agent inferred robot details from tools. | `EMBODIMENT.md` is the compiled Agent-readable body manual. |
| Skills ran without body compatibility checks. | `SkillExecutor` fails closed on `blocked` or `unknown` compatibility. |

## Migration Steps

### 1. Back up your existing workspace

```bash
cp -r ~/.rosclaw ~/.rosclaw.backup.$(date +%Y%m%d)
```

### 2. Link an e-URDF profile

Choose the profile that matches your physical robot:

```bash
rosclaw body link-eurdf unitree-g1
```

If your robot is not in `e-urdf-zoo`, create a minimal e-URDF profile first and
register it in the robot registry.

### 3. Customize instance state

Edit `~/.rosclaw/body/body.yaml` to record instance-specific facts:

```yaml
body_instance_id: lab-g1-001
nickname: Alice
installed_components:
  sensors:
    head_rgb_camera:
      status: available
  actuators:
    left_arm_actuator_group:
      status: available
    right_arm_actuator_group:
      status: unavailable
      reason: "servo r10 overheated"
capabilities:
  disabled:
    - capability: dual_arm_manipulation
      reason: "right arm unavailable"
      source: human
forbidden_capabilities:
  - free_run_motor_test
```

Prefer the CLI for structured changes:

```bash
rosclaw body update-state \
  --set installed_components.actuators.right_arm_actuator_group.status=unavailable \
  --reason "right arm servo overheated"

rosclaw body update-state --disable-capability dual_arm_manipulation --reason "right arm unavailable"
```

### 4. Verify the effective body

```bash
rosclaw body inspect --json
rosclaw body inspect --agent
```

Check that `EMBODIMENT.md` was generated and contains the expected sections.

### 5. Update skills to declare requirements (optional)

Existing JSON skills continue to work. To opt into body compatibility checking,
add a `.skill.yaml` manifest next to the skill:

```yaml
skill_id: walk_forward
requirements:
  capabilities:
    - locomotion
  min_actuator_groups:
    - legs
  allow_degraded:
    - locomotion
```

`SkillExecutor` will now block execution if the effective body does not satisfy
the requirements.

### 6. Update automation that reads body state

Replace direct reads of `~/.rosclaw/body/body.yaml` with the resolver API:

```python
from rosclaw.body.resolver import BodyResolver

resolver = BodyResolver()
body = resolver.get_effective_body()
print(body.effective_body_hash)
```

Or use `rosclaw://` URIs:

```python
from rosclaw.body.references import resolve

body = resolve("rosclaw://body/current/effective")
```

## Backward Compatibility

- `rosclaw robot *` commands remain functional.
- Existing skills without manifests run as before, with a warning if no body is
  linked.
- If no body is linked, `SkillExecutor` logs a warning and continues.
- e-urdf-zoo directory layout is unchanged.

## Rolling Back

To revert to a pre-body-system state:

```bash
rm -rf ~/.rosclaw/body
# Restore from backup
cp -r ~/.rosclaw.backup.YYYYMMDD ~/.rosclaw
```

## Common Issues

### "No body linked"

Run `rosclaw body link-eurdf <profile_id>`.

### Effective body hash keeps changing

Volatile fields such as `generated_at` and `snapshot_captured_at` are excluded
from the hash. If the hash still changes, check that no absolute paths or
non-deterministic ordering are being stored in `body.yaml`.

### Skill unexpectedly blocked

Check `EMBODIMENT.md` section 5 (Current Capabilities) and section 8 (Known
Faults). A required capability may be degraded or a fault may have disabled it.

### Multiple robots in one workspace

Use the body registry:

```bash
rosclaw body list
rosclaw body switch lab-g1-002
```

## Live ROS Introspection

After linking a body, you can seed `runtime_state` from a live ROS graph:

```bash
rosclaw body update-state --from-ros --reason "initial bringup"
```

This is especially useful when migrating a robot that is already running ROS
and you want the body ledger to reflect the currently advertised topics and
nodes.
