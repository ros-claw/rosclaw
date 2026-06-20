# EMBODIMENT.md Format Guide

`~/.rosclaw/body/EMBODIMENT.md` is the **compiled body manual** that the Agent
reads before deciding which skills are safe to run. It is generated from the
*Effective Body Model* and should always be treated as the single source of
truth for the current physical robot.

Do **not** hand-edit the managed sections of this file. Use the CLI to change
the body, then let `BodyResolver` regenerate the document:

```bash
rosclaw body update-state --set installed_components.sensors.head_rgb_camera.status=unavailable --reason "camera offline"
rosclaw body note --type incident --affects right_arm_actuator_group "Right arm overheated during test."
```

## File Location

```text
~/.rosclaw/body/
├── body.yaml              # Body Instance Ledger (human/machine editable)
├── calibration.yaml       # Calibration data
├── maintenance.log        # JSONL history of all changes
├── EMBODIMENT.md          # This file
├── skill_compatibility.yaml
└── refs/
    ├── eurdf.lock
    ├── eurdf.profile.yaml
    └── effective_body.json
```

## Sections

The rendered document contains the following sections. Some are always present;
others appear only when the underlying data exists.

### 1. Identity
- `body_instance_id` — unique instance identifier.
- `base_model_id` — linked e-URDF model.
- `nickname` — human-friendly name.
- `effective_body_hash` — deterministic fingerprint of the effective body.
- `generated_at` / `source_files` — audit trail.

### 2. Body Structure
- Kinematic tree summary (links, joints, degrees of freedom).
- Important frames (base, tool, camera, etc.).
- Joint groups (arms, legs, head, gripper, etc.).

### 3. Installed Sensors
- List of sensors from the e-URDF plus instance overrides.
- `Sensor Readiness` sub-section shows which sensors are currently available.

### 4. Installed Actuators and Tools
- Actuators, end-effectors, grippers, and any retrofit tooling.

### 5. Current Capabilities
Derived from the effective body and grouped by status:
- **Enabled** — safe to use.
- **Degraded** — usable only if the skill manifest explicitly allows the
  degradation.
- **Disabled** — explicitly turned off in `body.yaml`.

### 6. Forbidden Capabilities
Capabilities that must never be invoked (e.g. because of a safety incident).

### 7. Safety Limits
- Global safety envelope.
- Workspace limits.
- Contact / force limits.
- Safety gates (required checks before motion).

### 8. Known Faults
Active faults and their impact on capabilities.

### 9. Known Successful Experiences
Reusable practice episodes from memory that match this body hash.

### 10. Known Failed Experiences
Failed episodes that should be avoided or require extra validation.

### 11. Calibration Summary
Calibration status, warnings, and stale entries.

### 12. Maintenance and Modification History
Chronological log of maintenance events, incidents, repairs, and modifications.

### 13. Agent Operating Instructions
Mandatory do / do-not / when-unsure rules generated from safety limits and
faults.

### 14. Machine-readable Summary
A YAML front-matter block containing the effective body hash, capability map,
and key safety flags. Other tools can parse this block without reading the full
document.

### 15. Source Files
Paths and hashes of the files used to compile the effective body.

### 16. Regeneration
A footer reminding the reader that the file is auto-generated and pointing back
to this guide.

## Human Notes Zone

You may add free-form notes in the managed human-notes markers:

```markdown
<!-- HUMAN-NOTES-START -->

My lab robot has a custom mounting bracket on the left wrist.
Always check bracket bolts before running manipulation skills.

<!-- HUMAN-NOTES-END -->
```

The renderer preserves the content between these markers across regenerations.
Notes placed outside the markers may be overwritten.

## Determinism and Hashing

The effective body hash is computed from the normalized effective body dict:

1. Convert all dataclasses to dicts recursively.
2. Drop volatile fields (`generated_at`, `captured_at`, file paths).
3. Sort dict keys and lists.
4. Serialize as compact JSON.
5. Compute SHA-256.

This hash is used for:
- Skill compatibility caching.
- Memory lookups (`body_hash`).
- Cross-module consistency checks.

## Regenerating the Document

```bash
# After any body change
rosclaw body inspect --agent

# Or force a recompile
python -m rosclaw.body.resolver --recompile
```

If you delete `EMBODIMENT.md`, running any `rosclaw body` command that touches
the effective body will recreate it.
