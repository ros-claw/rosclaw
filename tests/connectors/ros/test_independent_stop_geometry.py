"""Repeated or paused poses cannot establish an actual physical stop window."""

import copy
import importlib.util
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("independent_stop", ROOT / "independent_stop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def poses():
    start = datetime(2026, 10, 8, tzinfo=UTC)
    return [
        {
            "source": "independent_gazebo_ground_truth_subscription",
            "x": 1,
            "y": 2,
            "yaw": math.pi - 0.002,
            "time_sec": 100 + i * 0.1,
            "captured_at": (start + timedelta(seconds=i * 0.1)).isoformat(),
        }
        for i in range(30)
    ]


def test_stationary_advancing_actual_poses_and_wrapped_yaw(monkeypatch):
    rows = poses()
    rows[-1]["yaw"] = -math.pi + 0.002
    result = load(monkeypatch).stop_geometry(rows)
    assert result["physical_acceptance"] == "NOT_VERIFIED"
    assert result["displacement_m"] == 0
    assert result["yaw_change_rad"] == pytest.approx(0.004)


@pytest.mark.parametrize(
    "fault",
    ["repeat", "pause", "regress", "short", "move", "rotate", "source", "nan", "bool", "naive"],
)
def test_incomplete_frozen_wrong_source_or_moving_window_refused(monkeypatch, fault):
    rows = poses()
    if fault == "repeat":
        rows[-1] = copy.deepcopy(rows[-2])
    elif fault == "pause":
        for row in rows:
            row["time_sec"] = 100
    elif fault == "regress":
        rows[-1]["time_sec"] = 99
    elif fault == "short":
        rows = rows[:20]
    elif fault == "move":
        rows[-1]["x"] += 0.02
    elif fault == "rotate":
        rows[-1]["yaw"] += 0.04
    elif fault == "source":
        rows[-1]["source"] = "nav2_odometry"
    elif fault == "nan":
        rows[-1]["x"] = float("nan")
    elif fault == "bool":
        rows[-1]["time_sec"] = True
    elif fault == "naive":
        rows[-1]["captured_at"] = "2026-10-08T00:00:03"
    with pytest.raises(ValueError):
        load(monkeypatch).stop_geometry(rows)
