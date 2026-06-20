# Testing the body module

This guide covers unit, CLI, and integration tests for the `rosclaw.body` three-layer body system.

## Unit tests

Run the body module test suite:

```bash
pytest tests/body -q
```

Key test files:

| File | Coverage |
|---|---|
| `tests/body/test_schema.py` | `EurdfProfile`, `BodyYaml`, `EffectiveBody`, `CalibrationYaml`, `MaintenanceEvent` schema behavior. |
| `tests/body/test_link_eurdf.py` | CLI `rosclaw body link-eurdf` artifact generation and force semantics. |
| `tests/body/test_inspect.py` | CLI `rosclaw body inspect` text and JSON output. |
| `tests/body/test_effective_body.py` | `EffectiveBodyCompiler` merging of e-URDF, body.yaml, calibration, and maintenance events. |
| `tests/body/test_diff.py` | `BodyDiffer` change categorization and skill-recheck impact detection. |
| `tests/body/test_update_state.py` | `rosclaw body update-state` paths, capability changes, and dry-run. |
| `tests/body/test_note.py` | Maintenance note categorization and skill recheck triggering. |
| `tests/body/test_skill_compatibility.py` | `SkillCompatibilityChecker` statuses and incremental recheck. |
| `tests/body/test_cross_module_references.py` | `rosclaw://body/current/effective` URI resolution across modules. |

All tests use temporary workspaces (`tmp_path` + `monkeypatch.setenv("HOME", ...)`), so they do not touch the real `~/.rosclaw/body` directory.

## CLI smoke tests

End-to-end happy path:

```bash
rm -rf ~/.rosclaw/body
rosclaw body link-eurdf unitree-g1
rosclaw body inspect
rosclaw body list
rosclaw body show --agent
rosclaw body update-state \
  --set installed_components.sensors.head_camera.status=unavailable \
  --reason "camera cable disconnected"
rosclaw body diff
rosclaw body note "Right arm overheated" \
  --type incident --severity warning --affects right_arm
rosclaw body history
```

Expected behavior:

- `link-eurdf` creates `body.yaml`, `calibration.yaml`, `maintenance.log`, `EMBODIMENT.md`, and `refs/*`.
- `inspect` prints the current effective body hash.
- `list` shows the registered body.
- `diff` reports the sensor status change and any capability impact.
- `history` contains at least the initial link snapshot and the update-state snapshot.

## Skill executor regression

The skill manager relies on body compatibility checks. Run:

```bash
pytest tests/test_skill_manager.py -q
```

Verify that `TestSkillExecutorBodyCheck` passes — it confirms fail-closed behavior when the resolver errors or returns `unknown` compatibility.

## Live ROS introspection (`--from-ros`)

The `--from-ros` flag on `rosclaw body update-state` reads runtime topics and parameters from a running ROS graph and updates the body state before recompiling.

### ROS 2

```bash
# Terminal 1 — launch a robot stack or simulator
ros2 launch <pkg> robot.launch.py

# Terminal 2 — introspect and update body state
rosclaw body update-state --from-ros \
  --set runtime_state.online=true \
  --reason "live ROS 2 introspection"
```

### ROS 1

```bash
# Terminal 1
roscore
rosrun robot_state_publisher robot_state_publisher

# Terminal 2
rosclaw body update-state --from-ros \
  --set runtime_state.online=true \
  --reason "live ROS 1 introspection"
```

ROS 1 bridge / port-mapped connections may hit the `rosbridge` HTTP 400 Host-header issue. If so, connect via the container IP on port `9090` or use the local `roscore` directly.

## Continuous integration

Run the full relevant test set before pushing body changes:

```bash
pytest tests/body tests/test_skill_manager.py tests/test_cli.py tests/test_cli_coverage.py -q
```

If your change touches e-URDF loading or ROS connectors, also run:

```bash
pytest tests/integration/test_eurdf_loader.py \
       tests/integration/test_provider_eurdf_integration.py \
       tests/integration/test_sandbox_eurdf_integration.py -q
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No body linked` | `~/.rosclaw/body/body.yaml` missing | Run `rosclaw body link-eurdf <profile>`. |
| Effective hash changes on every render | `body.yaml` contains timestamps in hashed fields | Move volatile fields out of the hash input or exclude them in `EffectiveBodyCompiler`. |
| `rosclaw body list` empty | Workspace mismatch | Pass `--workspace ~/.rosclaw` or set `HOME` appropriately. |
| ROS introspection returns no joints | No `/joint_states` publisher | Start the robot stack; verify `rostopic list` / `ros2 topic list`. |

## Files

- `tests/body/` — body module tests
- `src/rosclaw/body/cli.py` — CLI handlers
- `src/rosclaw/body/resolver.py` — `BodyResolver` used by tests and runtime
