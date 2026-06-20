# Migrating to the three-layer body model

This guide helps users move from older ROSClaw robot configuration files to the `rosclaw.body` three-layer model.

## What changed

Before the body module:

- Robot model data lived in `e-urdf-zoo/` and was loaded directly by runtime code.
- Per-robot instance state was scattered across config files and runtime caches.
- Skills had no formal compatibility check against the current physical body.

With the body module:

| Layer | File / concept | Responsibility |
|---|---|---|
| Physical DNA | `e-urdf-zoo/` → `EurdfProfile` | Model-level definition of joints, links, sensors, actuators, capability hints, safety limits. |
| Body Instance Ledger | `~/.rosclaw/body/body.yaml` | This robot's instance state: installed components, overrides, forbidden capabilities, identity. |
| Compiled Body Manual | `~/.rosclaw/body/EMBODIMENT.md` | Agent-readable rendered summary generated from the **Effective Body Model**. |

## Backward compatibility

- Existing `rosclaw robot *` commands are unchanged.
- `e-urdf-zoo/` directory layout is unchanged.
- Existing JSON skills still work; compatibility defaults to `unknown` until a `.skill.yaml` manifest is provided.
- If no body is linked, `SkillExecutor` logs a warning and proceeds for backward compatibility.

## Creating a body for an existing robot

1. Identify the e-URDF profile ID.

   ```bash
   rosclaw body link-eurdf --help
   ls e-urdf-zoo
   ```

2. Link the profile to your workspace.

   ```bash
   rosclaw body link-eurdf unitree-g1
   ```

3. Review the generated body state.

   ```bash
   rosclaw body inspect
   rosclaw body show --agent
   ```

4. Open `~/.rosclaw/body/EMBODIMENT.md` and read the **Agent Operating Instructions**.

## Migrating per-robot overrides

If you previously edited robot-specific YAML files by hand, move equivalent fields into `body.yaml`:

| Old concept | New location | Example |
|---|---|---|
| Disabled capability list | `body.yaml` → `capabilities.disabled` | `capabilities.disabled: [run_fast]` |
| Installed sensor override | `body.yaml` → `installed_components.sensors` | `head_camera: {status: unavailable}` |
| Calibration offsets | `calibration.yaml` | `joints: {left_arm_joint_1: 0.01}` |
| Forbidden actions | `body.yaml` → `prohibited_capabilities` | `jump_over_obstacles` |
| Maintenance notes | `maintenance.log` (JSONL) | Use `rosclaw body note` |

After editing, run:

```bash
rosclaw body render
rosclaw body validate
```

## Migrating skills

Skills that need body-aware compatibility should provide a `.skill.yaml` manifest next to the skill file.

Example `skills/walk_forward.skill.yaml`:

```yaml
skill_id: walk_forward
skill_version: "1.0.0"
requires:
  robot_class: humanoid
  eurdf:
    compatible_profiles: [unitree-g1]
  capabilities:
    all_of: [walk]
  actuators:
    all_of:
      - group: left_leg
        status: available
      - group: right_leg
        status: available
degradation_policy:
  allow_lower_speed: false
```

Without a manifest, compatibility status is `unknown` and execution is blocked.

## Validation checklist

After migration:

- [ ] `rosclaw body inspect` returns exit code 0.
- [ ] `~/.rosclaw/body/EMBODIMENT.md` exists and contains your robot model.
- [ ] `rosclaw body validate` reports `PASS` or `PASS_WITH_WARNINGS`.
- [ ] Skills you expect to run report `compatible` in `rosclaw body inspect --skills`.
- [ ] Adding a fault downgrades the relevant capability to `degraded` or `blocked`.

## Rollback

To unlink the body and return to the pre-body-module state for that workspace:

```bash
rm -rf ~/.rosclaw/body
```

This removes the body module files. The underlying `e-urdf-zoo/` profile and existing skills are unaffected.

## Files

- `src/rosclaw/body/` — body module source
- `docs/body/EMBODIMENT_FORMAT.md` — EMBODIMENT.md reference
- `docs/body/SKILL_COMPATIBILITY.md` — skill compatibility rules
- `docs/body/URI_SCHEME.md` — cross-module URI scheme
