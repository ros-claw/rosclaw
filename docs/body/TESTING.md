# Body Module Testing Guide

This guide covers how to run and extend tests for the `rosclaw.body` three-layer
body system (e-URDF / `body.yaml` / `EMBODIMENT.md`).

## Quick Start

All body tests are isolated: they use `tmp_path` + `monkeypatch.setenv("HOME",
...)` so they never touch your real `~/.rosclaw` workspace.

```bash
# Run the full body test suite
pytest tests/body -q

# Run a specific test file
pytest tests/body/test_update_state.py -q
pytest tests/body/test_update_state_from_ros.py -q

# Run with verbose output
pytest tests/body -v
```

## Test Fixtures

### e-URDF fixture

`tests/body/fixtures/eurdf/unitree-g1/profile.yaml` is a minimal e-URDF
profile used by most tests. It is copied into the isolated workspace when a
test calls:

```python
with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
    assert rosclaw_main() == 0
```

### Isolated workspace fixture

Many tests use a shared fixture:

```python
@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    yield tmp_path
```

This ensures `BodyResolver` resolves `~` to a temporary directory.

## Test Categories

| File | What it covers |
|------|----------------|
| `test_schema.py` | Dataclass construction and serialization. |
| `test_link_eurdf.py` | `rosclaw body link-eurdf`. |
| `test_inspect.py` | `rosclaw body inspect` in text, JSON, and agent modes. |
| `test_effective_body.py` | Effective body compilation and hashing. |
| `test_diff.py` | Change detection and skill recheck triggers. |
| `test_update_state.py` | `--set`, maintenance logging, forbidden fields. |
| `test_update_state_from_ros.py` | `--from-ros` live introspection path. |
| `test_note.py` | `rosclaw body note`. |
| `test_skill_compatibility.py` | `compatible` / `degraded` / `blocked` / `unknown`. |
| `test_cross_module_references.py` | `rosclaw://` URI resolution. |
| `test_capability_management.py` | Capability enable/disable/dedup logic. |
| `test_incremental_skill_recheck.py` | Impact-aware recheck. |
| `test_multi_body_registry.py` | Multiple body instances and switching. |
| `test_list.py` | `rosclaw body list` output. |

## Testing `--from-ros`

The `--from-ros` feature connects to a live rosbridge server. In unit tests we
never open a real network connection. Instead, we monkeypatch
`rosclaw.body.cli.introspect_ros`:

```python
from unittest.mock import patch
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot, RosTopicInfo

snapshot = RosGraphSnapshot(
    ros_version="ros2",
    distro="humble",
    endpoint="ws://127.0.0.1:9090",
    topics=[
        RosTopicInfo(name="/camera/image_raw", msg_type="sensor_msgs/Image", is_sensor=True),
    ],
    services=[],
    actions=[],
    nodes=[{"name": "/robot_state_publisher"}],
    params=["/robot_description"],
    captured_at="...",
)
runtime_state = {"online": True, "ros_version": "ros2"}

with patch("rosclaw.body.cli.introspect_ros", return_value=(snapshot, runtime_state)):
    # run CLI
```

See `tests/body/test_update_state_from_ros.py` for full examples.

## Manual / Integration Testing

If you have a rosbridge server running, you can exercise the live path:

```bash
# Link a body first
rosclaw body link-eurdf unitree-g1

# Update runtime_state from the live ROS graph (dry-run)
rosclaw body update-state --from-ros --dry-run --reason "system bringup check"

# Persist the snapshot
rosclaw body update-state --from-ros --reason "system bringup check"

# Use a non-default rosbridge endpoint
rosclaw body update-state --from-ros --ros-endpoint ws://192.168.1.10:9090 --reason "edge bringup"
```

### ROS1 vs ROS2

`--from-ros` detects the ROS version automatically via `rosapi/get_ros_version`.
The runtime_state patch records `ros_version` and `ros_distro` so that
skills can gate on the ROS distribution.

### Known Caveats

- Some Docker / port-mapped ROS1 setups return HTTP 400 unless you connect via
  the container IP on port 9090 (see memory: ROS1 rosbridge Host header caveat).
- rosapi for ROS1 returns `/rosdistro` values that may differ from the host
  distro; use `ros_version` for branching logic, not `ros_distro` alone.

## Regression Tests

After any body-module change, run the broader regression suite to ensure the
CLI and integration paths still work:

```bash
pytest tests/body tests/test_cli.py tests/test_cli_coverage.py -q
pytest tests/integration/test_eurdf_loader.py tests/integration/test_provider_eurdf_integration.py tests/integration/test_sandbox_eurdf_integration.py -q
```

## Adding a New Body Test

1. Decide whether the test belongs in an existing file or a new one.
2. Use the `tmp_path` + `monkeypatch.setenv("HOME", ...)` pattern.
3. For CLI tests, patch `sys.argv` and call `rosclaw.cli.main`.
4. Assert on the effective body, `body.yaml`, or captured output.
5. Run the new test in isolation, then run the full suite.

## Troubleshooting

- **"No body linked"** — the test did not call `link-eurdf` first, or the
  `HOME` environment was not patched.
- **JSON parse errors in CLI tests** — drain `capsys.readouterr()` between
  multiple CLI invocations.
- **Hash mismatch** — ensure the test does not mutate the effective body
  between hash reads without recompiling.
