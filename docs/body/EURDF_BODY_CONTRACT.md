# e-URDF / Body Contract

This document defines the contract between the e-URDF model layer and the body
instance layer. It is the reference for implementers adding new robot profiles
or body adapters.

## Design rule: only tighten

The body instance layer (`body.yaml` / `calibration.yaml` / `maintenance.log`)
may only **tighten** or **disable** capabilities declared by the e-URDF. It may
not invent new capabilities, joints, sensors, or safety limits that are not
present in the linked e-URDF profile.

Examples of valid tightening:

- Mark a sensor as `unavailable` because it is physically disconnected.
- Add calibration offsets within the e-URDF limits.
- Disable a capability via `forbidden_capabilities`.
- Reduce velocity/acceleration limits within e-URDF bounds.

Examples of invalid expansion:

- Declaring a sensor that does not exist in the e-URDF.
- Enabling a capability that the e-URDF marks as unsupported.
- Setting joint limits beyond the e-URDF hard limits without a retrofit event.

## e-URDF profile normalization

`EurdfProfile.from_robot_complete_profile(profile)` converts a
`RobotCompleteProfile` into the canonical schema used by the body module:

- `profile_id` and `version`
- `joints` with limits and types
- `frames` with parent/child relationships
- `sensors` and `actuators` with required topics
- `capabilities` with dependencies
- `safety` limits and emergency contacts
- `provider_interfaces` for runtime diagnosis

## Body instance ledger (`body.yaml`)

`body.yaml` contains:

- `body_instance_id`: unique instance identifier
- `profile_id` / `eurdf_uri`: link to the e-URDF profile
- `nickname`: human-readable label
- `installed_components`: per-component status overrides
- `forbidden_capabilities`: explicitly disabled capabilities
- `notes`: free-form human notes

The compiler merges `body.yaml` on top of the e-URDF profile.

## Calibration ledger (`calibration.yaml`)

Calibration data is applied after `body.yaml`. It contains:

- Joint offsets
- Sensor offsets / intrinsics
- Actuator direction / scaling
- Validation bounds

Calibration may only adjust values; it may not change topology.

## Maintenance log (`maintenance.log`)

`maintenance.log` is a JSONL stream of `MaintenanceEvent` records. Each event
may carry:

- `affects`: list of components or capabilities impacted
- `before` / `after`: state diff
- `requires_skill_recheck`: whether skill compatibility must be recomputed
- `requires_render`: whether `EMBODIMENT.md` must be re-rendered

Events are the authoritative history of body changes. The compiler reads the
full log and applies events that describe state changes.

## Structure preservation

When a component fails, the Effective Body Model **keeps the component in the
structure** but marks it `unavailable`. Related capabilities become `degraded`
or `blocked`. This preserves agent expectations about topology while making
runtime safety explicit.

## Retrofit events

A retrofit event records a hardware modification that legitimately changes the
body topology or capabilities. It must include:

- `component`: the modified subsystem
- `retrofit_type`: e.g. `add_sensor`, `replace_end_effector`
- `summary`: human-readable description
- `requires_skill_recheck: true`

After a retrofit, the body should be re-initialized from an updated e-URDF
profile if the model-level definition changed.

## Version pinning

The e-URDF link is pinned via `refs/eurdf.lock`:

```yaml
profile_id: unitree-g1
version: "1.0.0"
uri: rosclaw://eurdf/unitree-g1@1.0.0
```

This lock ensures reproducible compilation. To upgrade a body to a new e-URDF
version, use `rosclaw body link-eurdf <profile_id> --version <version>`.

## Skill compatibility derivation

`SkillCompatibilityChecker` derives compatibility from the Effective Body
Model. A skill manifest declares `requirements` such as:

```yaml
requirements:
  capabilities:
    - locomotion
  sensors:
    - head_rgb_camera
  joints:
    - left_hip_pitch
  actuator_groups:
    - legs
```

The checker evaluates each requirement against the effective body and returns
`compatible`, `degraded`, `blocked`, or `unknown`.

## Provider interface contract

`ProviderBodyBinder` maps the Effective Body Model to provider interfaces. Each
interface declares:

- `name`: topic or interface identifier
- `category`: `sensor`, `actuator`, `safety`, `telemetry`
- `required`: whether the interface is mandatory
- `status`: `available`, `degraded`, `unavailable`, `blocked`

When no runtime set is supplied, `diagnose()` derives status from the
Effective Body Model. When a runtime set is supplied, missing interfaces are
marked unavailable.

## See also

- [BODY_RUNTIME_OVERVIEW.md](BODY_RUNTIME_OVERVIEW.md)
- [SKILL_COMPATIBILITY.md](SKILL_COMPATIBILITY.md)
- [URI_SCHEME.md](URI_SCHEME.md)
