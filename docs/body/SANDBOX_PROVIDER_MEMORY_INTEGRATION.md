# Sandbox / Provider / Memory Integration

This document describes how the body module integrates with the sandbox,
provider, and memory subsystems. All three consume the **Effective Body Model**
only; they never read `body.yaml` directly.

## Sandbox integration

`SandboxBodyAdapter` converts the Effective Body Model into simulator configs.

### Supported simulators

| Simulator | Method | Output |
|-----------|--------|--------|
| MuJoCo | `to_mujoco_config()` | dict with URDF path, mesh dirs, joint limits, disabled actuators |
| Isaac Sim | `to_isaac_config()` | dict with stage path, joint overrides, calibration offsets |

### Required fields

`to_mujoco_config()` returns:

```python
{
    "effective_body_hash": str,
    "eurdf_uri": str,
    "urdf_path": str,
    "mesh_dirs": list[str],
    "disabled_actuators": list[str],
    "joint_limits": dict,
    "safety": dict,
    "collision": dict,
    "calibration_offsets": dict,
}
```

The sandbox uses `disabled_actuators` to disable unavailable actuators and
`joint_limits` to enforce tightened limits from the Effective Body Model.

### Fail-closed behavior

If the Effective Body Model reports a capability as `blocked`, the sandbox
can refuse to load a scene that exercises that capability. Calibration offsets
are applied so that simulated joints match the real robot's calibrated state.

## Provider integration

`ProviderBodyBinder` maps the Effective Body Model to provider interfaces for
runtime diagnosis.

### ProviderInterface

Each interface declares:

- `name`: topic or interface identifier
- `category`: `sensor`, `actuator`, `safety`, `telemetry`
- `required`: mandatory or optional
- `status`: `available`, `degraded`, `unavailable`, `blocked`
- `error`: human-readable error if unavailable
- `topic`: associated ROS topic when applicable
- `provider_ref`: reference to the provider implementation
- `metadata`: extra diagnostics

### Diagnosis

`ProviderBodyBinder.diagnose(available=None)` returns a
`ProviderBodyDiagnosis`:

- If `available` is omitted, status is derived from the Effective Body Model.
- If `available` is a set of runtime-provided interface names, missing
  interfaces are marked unavailable.

Required interfaces that are unavailable produce a `DEGRADED` or `UNHEALTHY`
overall status.

### Example

```python
from rosclaw.provider.body_binder import ProviderBodyBinder
from rosclaw.body.resolver import BodyResolver

body = BodyResolver().get_effective_body()
binder = ProviderBodyBinder.from_effective_body(body)
diagnosis = binder.diagnose(available={"/camera/color/image_raw", "/joint_states"})
```

## Memory integration

`BodyMemoryEventWriter` records body change events to the memory subsystem.

### Event semantics

- Memory writes are **best-effort**. Failure to write a memory event never
  blocks a body update.
- If no memory client is configured, the writer is a no-op.
- Each event stores the full `BodyDiff` dictionary.

### Event types

- `body_effective_changed`
- `body_active_switched`
- `body_skill_compatibility_changed`
- `body_provider_health_changed`

### Example

```python
from rosclaw.memory.body_events import BodyMemoryEventWriter
from rosclaw.body.diff import BodyDiff

writer = BodyMemoryEventWriter(memory_client)
writer.write_effective_changed(diff=BodyDiff(...))
```

## Cross-module contract tests

The following contract tests verify that each subsystem consumes the same
Effective Body Model:

- `tests/provider/test_provider_body_contract.py`
- `tests/provider/test_body_binder.py`
- `tests/sandbox/test_sandbox_body_contract.py`
- `tests/sandbox/test_body_adapter.py`
- `tests/memory/test_memory_body_contract.py`
- `tests/memory/test_body_events.py`
- `tests/body/test_body_update_writes_memory_event.py`
- `tests/dashboard/test_dashboard_body_contract.py`

Each test asserts that the subsystem's output reflects the Effective Body
Model's hash, readiness, or capabilities.

## See also

- [BODY_RUNTIME_OVERVIEW.md](BODY_RUNTIME_OVERVIEW.md)
- [EURDF_BODY_CONTRACT.md](EURDF_BODY_CONTRACT.md)
- [AGENT_VIEW.md](AGENT_VIEW.md)
