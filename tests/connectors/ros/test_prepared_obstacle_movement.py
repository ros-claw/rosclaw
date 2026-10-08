"""Pure request screening, no world service calls or physics acceptance."""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


@pytest.fixture
def movement(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "faults",
        SimpleNamespace(
            observed=lambda: None, service=lambda *a: pytest.fail("world mutation in offline test")
        ),
    )
    spec = importlib.util.spec_from_file_location(
        "prepared_obstacle", ROOT / "prepared_obstacle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = {
        "schema_version": "rosclaw.sim_physics_fixture.v1",
        "binding": {"body_snapshot_hash": "body", "obstacle_names": ["person"]},
        "obstacles": [
            {"name": "person", "pose": [5, 0, 0.1, 0, 0, 0], "box_size": [0.7, 0.7, 0.2]}
        ],
    }
    body = {"effective_body_hash": "body", "physical_radius_m": 0.25}
    sample = {
        "x": -1,
        "y": -1,
        "collision_count": 0,
        "observation_complete": True,
        "captured_at": datetime.now(UTC).isoformat(),
    }
    return module, fixture, body, sample


def test_preloaded_pose_preserves_height_and_identity(movement):
    module, fixture, body, sample = movement
    request = module.pose_request(fixture, body, sample, name="person", x=0.25, y=0.25)
    assert 'name: "person"' in request
    assert "z: 0.10000000000000001" in request
    assert "orientation { w: 1 }" in request


@pytest.mark.parametrize(
    "fault",
    [
        "close",
        "nan",
        "huge",
        "unknown",
        "injection",
        "body",
        "stale",
        "future",
        "contact",
        "incomplete",
        "rotation",
        "negative",
    ],
)
def test_unsafe_or_unbound_move_refused(movement, fault):
    module, fixture, body, sample = movement
    name, x, y = "person", 0.25, 0.25
    if fault == "close":
        x, y = -1, -1
    elif fault == "nan":
        x = float("nan")
    elif fault == "huge":
        x = 10**400
    elif fault == "unknown":
        name = "new_person"
    elif fault == "injection":
        name = 'person" position {'
    elif fault == "body":
        body["effective_body_hash"] = "other"
    elif fault == "stale":
        sample["captured_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    elif fault == "future":
        sample["captured_at"] = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
    elif fault == "contact":
        sample["collision_count"] = 1
    elif fault == "incomplete":
        sample["observation_complete"] = False
    elif fault == "rotation":
        fixture["obstacles"][0]["pose"][5] = 1
    elif fault == "negative":
        fixture["obstacles"][0]["box_size"][0] = -1
    with pytest.raises(ValueError):
        module.pose_request(fixture, body, sample, name=name, x=x, y=y)
