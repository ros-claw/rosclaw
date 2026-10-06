"""Known-body acceptance must bind the actual vendor identity before any action."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "integrations/ros_probe/acceptance"
spec = importlib.util.spec_from_file_location("acceptance_profiles", RUNNER / "profiles.py")
profiles = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profiles
spec.loader.exec_module(profiles)


@pytest.mark.parametrize("name", ["waffle", "burger"])
def test_vendor_identity_is_required_even_with_explicit_profile(tmp_path, name):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(f'<robot name="turtlebot3_{name}"/>')
    assert profiles.profile_for_urdf(urdf).name == name
    other = "burger" if name == "waffle" else "waffle"
    with pytest.raises(ValueError, match="does not match"):
        profiles.profile_for_urdf(urdf, other)
    urdf.write_text('<robot name="unverified_robot"/>')
    with pytest.raises(ValueError, match="supported vendor"):
        profiles.profile_for_urdf(urdf)


@pytest.mark.parametrize(
    "name, radius, cleaner", [("waffle", 0.25, 0.275), ("burger", 0.15, 0.175)]
)
def test_prepare_only_compiles_actual_body_and_legal_recovery_centers(
    tmp_path, name, radius, cleaner
):
    # This is an offline Body/denominator contract, not observed physical coverage.
    root = tmp_path / name
    root.mkdir()
    (root / "robot.urdf").write_text(f"""<robot name="turtlebot3_{name}">
      <link name="base_footprint"/><link name="base_link"/>
      <joint name="base_joint" type="fixed"><parent link="base_footprint"/>
      <child link="base_link"/></joint></robot>""")
    width = 64
    occupancy = [
        100 if abs((x + 0.5) * 0.05 - 1.6) >= 1.5 or abs((y + 0.5) * 0.05 - 1.6) >= 1.5 else 0
        for y in range(width)
        for x in range(width)
    ]
    (root / "measured_map.json").write_text(
        json.dumps(
            {
                "width": width,
                "height": width,
                "resolution": 0.05,
                "origin": [-1.6, -1.6],
                "occupancy": occupancy,
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER / "run.py"),
            "--directory",
            str(root),
            "--prepare-only",
            "--profile",
            name,
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout)["status"] == "PREPARED"
    body = json.loads((root / "body.json").read_text())
    config = json.loads((root / "execution_config.json").read_text())
    assert body["physical_radius_m"] == radius
    assert body["cleaning_polygon"] == [
        [-cleaner, -cleaner],
        [cleaner, -cleaner],
        [cleaner, cleaner],
        [-cleaner, cleaner],
    ]
    assert config["body_snapshot_hash"] == body["effective_body_hash"]
    assert config["grid"]["cleaning_polygon"] == body["cleaning_polygon"]
    assert config["grid"]["accessible_cells"]
    recovery = config["recovery_centers"]
    assert recovery and all(max(abs(x), abs(y)) + radius < 1.5 for x, y in recovery)
    if name == "burger":
        assert max(x for x, _ in recovery) == pytest.approx(1.325)
    assert not (root / "daemon_ready.json").exists()
    configured = subprocess.run(
        [
            sys.executable,
            str(RUNNER / "configure_native.py"),
            "--directory",
            str(root),
            "--endpoint",
            "ws://127.0.0.1:19094",
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    import yaml

    native = yaml.safe_load((root / "home/config.yaml").read_text())
    assert json.loads(configured.stdout)["body_snapshot_hash"] == body["effective_body_hash"]
    assert native["agent"]["body_id"] == body["body_id"]
    assert native["agent"]["default_mode"] == "SIMULATION"
    assert native["mcp_servers"][0]["required_body_types"] == [body["body_id"]]
    assert native["mcp_servers"][0]["supported_modes"] == ["SIMULATION"]
    assert native["mcp_servers"][0]["args"][-1] == "ws://127.0.0.1:19094"
    original_config = (root / "home/config.yaml").read_bytes()
    body["effective_body_hash"] = "forged"
    (root / "body.json").write_text(json.dumps(body))
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER / "configure_native.py"),
            "--directory",
            str(root),
            "--overwrite",
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rejected.returncode != 0
    assert "differ from the compiled Body" in rejected.stderr
    assert (root / "home/config.yaml").read_bytes() == original_config


@pytest.mark.parametrize("mutation", ["body", "declaration", "mode", "endpoint"])
def test_native_preflight_rejects_inconsistent_binding_before_startup(mutation):
    spec = importlib.util.spec_from_file_location("acceptance_native", RUNNER / "native.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    config = {
        "agent": {
            "body_id": "ros_expert_burger",
            "default_mode": "SIMULATION",
            "ros_expert": {"endpoint": "ws://127.0.0.1:19094"},
        },
        "mcp_servers": [
            {
                "action_tools": ["coverage.execute"],
                "required_body_types": ["ros_expert_burger"],
                "supported_modes": ["SIMULATION"],
            }
        ],
    }
    native.validate_fixture_config(config, "ros_expert_burger", "ws://127.0.0.1:19094")
    if mutation == "body":
        config["agent"]["body_id"] = "ros_expert_base"
    elif mutation == "declaration":
        config["mcp_servers"][0]["required_body_types"] = ["ros_expert_base"]
    elif mutation == "mode":
        config["mcp_servers"][0]["supported_modes"].append("REAL")
    else:
        config["agent"]["ros_expert"]["endpoint"] = "ws://127.0.0.1:19090"
    with pytest.raises(ValueError):
        native.validate_fixture_config(config, "ros_expert_burger", "ws://127.0.0.1:19094")
