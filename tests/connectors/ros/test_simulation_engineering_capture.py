"""Synthetic SDK role and freshness boundaries; no physical repair claim."""

import importlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_engineering_capture")


def settings(case):
    return {
        "run_id": "owned-run",
        "generic_source_observer": True,
        "fault": "none",
        "governed_sink": {"bootstrap": "synthetic contract"},
        "engineering_capture": {
            "schema_version": "rosclaw.engineering_failure_capture.v1",
            "case": case,
            "phase": "before_repair",
            "run_id": "owned-run",
        },
    }


@pytest.mark.parametrize(
    "case",
    sorted(
        [
            "map_to_odom_missing",
            "scan_stale",
            "nav2_lifecycle_inactive",
            "footprint_undersized",
            "duplicate_velocity_route",
        ]
    ),
)
def test_explicit_isolated_nonready_capture_roles(module, monkeypatch, case):
    monkeypatch.setattr(module.Path, "is_file", lambda _: True)
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    value = settings(case)
    assert module.validate_engineering_capture(value) == value["engineering_capture"]


@pytest.mark.parametrize("fault", ["host", "domain", "run", "extra", "fixed", "mixed", "sink"])
def test_failure_capture_cannot_authorize_host_or_normal_execution(module, monkeypatch, fault):
    monkeypatch.setattr(module.Path, "is_file", lambda _: fault != "host")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "0" if fault == "domain" else "1")
    value = settings("scan_stale")
    if fault == "run":
        value["engineering_capture"]["run_id"] = "other"
    elif fault == "extra":
        value["engineering_capture"]["permit"] = True
    elif fault == "fixed":
        value["engineering_capture"]["phase"] = "after_repair"
    elif fault == "mixed":
        value["fault"] = "world_reset"
    elif fault == "sink":
        value.pop("governed_sink")
    with pytest.raises(ValueError, match="isolated SDK engineering"):
        module.validate_engineering_capture(value)


def test_ordinary_source_fixture_keeps_existing_role(module):
    assert module.validate_engineering_capture({}) is None


@pytest.mark.parametrize(
    "native_age,physics_age,expected",
    [(0.1, 0.1, True), (0.6, 0.1, False), (0.1, 0.4, False), (-1, 0.1, False)],
)
def test_original_capture_requires_current_native_and_physics(
    module, native_age, physics_age, expected
):
    now = datetime.now(UTC)
    state = {
        "native": {"captured_at": (now - timedelta(seconds=native_age)).isoformat()},
        "physics": {"captured_at": (now - timedelta(seconds=physics_age)).isoformat()},
    }
    assert module.current_capture(state) is expected
