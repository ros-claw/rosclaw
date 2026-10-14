"""Fault triggers require actual measured movement, never just an enabled lease."""

import importlib
import json
from pathlib import Path

import pytest

from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from tests.connectors.ros.test_navigation_only_executor import compiled_body


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("governed_daemon_sigkill_fixture")
    binding = SimulationNavigationBinding.from_body(compiled_body())
    sample = {
        "body_id": binding.body_id,
        "body_snapshot_hash": binding.body_snapshot_hash,
        "run_id": binding.run_id,
        "frame_id": binding.frame_id,
        "evidence_domain": "GAZEBO_PHYSICS",
        "collision_source": "gazebo_contacts",
        "observation_complete": True,
        "collision_count": 0,
        "lease_enabled": True,
        "linear_speed_mps": 0.0,
        "angular_speed_rps": 0.0,
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "time_sec": 1.0,
    }
    return module, binding, sample


def test_enabled_stationary_lease_is_not_motion(fixture):
    module, binding, sample = fixture
    assert not module.is_physical_motion(sample, binding)
    sample["linear_speed_mps"] = binding.stop_speed_mps * 2
    assert module.is_physical_motion(sample, binding)


@pytest.mark.parametrize("fault", ["nan", "overspeed", "wrong_run", "incomplete"])
def test_untrustworthy_or_unsafe_physics_cannot_trigger_a_fault(fixture, fault):
    module, binding, sample = fixture
    sample["linear_speed_mps"] = binding.stop_speed_mps * 2
    if fault == "nan":
        sample["linear_speed_mps"] = float("nan")
    elif fault == "overspeed":
        sample["linear_speed_mps"] = binding.max_speed_mps + 1
    elif fault == "wrong_run":
        sample["run_id"] = "other"
    else:
        sample["observation_complete"] = False
    with pytest.raises(ValueError):
        module.is_physical_motion(sample, binding)


def test_partial_last_line_never_becomes_original_physics(fixture, tmp_path):
    module, _, sample = fixture
    (tmp_path / "witness.jsonl").write_text(json.dumps(sample) + "\n" + '{"incomplete":')
    assert module.last_physics(tmp_path) == sample
    real = tmp_path / "original.jsonl"
    (tmp_path / "witness.jsonl").rename(real)
    (tmp_path / "witness.jsonl").symlink_to(real)
    with pytest.raises(OSError):
        module.last_physics(tmp_path)
