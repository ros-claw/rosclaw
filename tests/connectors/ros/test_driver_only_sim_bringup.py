"""Driver-only bringup is closed to the explicit unadmitted isolated SDK role."""

import importlib
from pathlib import Path

import pytest

from tests.connectors.ros.test_simulation_engineering_capture import settings


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("prepared_source_stack")


def test_default_bringup_preserved_and_owned_driver_only_can_skip_nav2(module, monkeypatch):
    assert module.navigation_stack_enabled({}, {}) is True
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setattr(module.Path, "is_file", lambda _: True)
    assert (
        module.navigation_stack_enabled(
            {"start_navigation_stack": False}, settings("driver_onboarding")
        )
        is False
    )


@pytest.mark.parametrize(
    "fault",
    [
        "host",
        "domain",
        "wrong_role",
        "wrong_phase",
        "wrong_run",
        "no_sink",
        "not_boolean",
        "starts_nav2",
    ],
)
def test_driver_only_cannot_change_other_roles_or_enable_navigation(module, monkeypatch, fault):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "0" if fault == "domain" else "1")
    monkeypatch.setattr(module.Path, "is_file", lambda _: fault != "host")
    value = settings("driver_onboarding")
    descriptor = {"start_navigation_stack": False}
    if fault == "wrong_role":
        value["engineering_capture"]["case"] = "navigation_onboarding"
    elif fault == "wrong_phase":
        value["engineering_capture"]["phase"] = "rollback"
    elif fault == "wrong_run":
        value["engineering_capture"]["run_id"] = "other"
    elif fault == "no_sink":
        value.pop("governed_sink")
    elif fault == "not_boolean":
        descriptor["start_navigation_stack"] = 0
    elif fault == "starts_nav2":
        descriptor = {}
    with pytest.raises(ValueError):
        module.navigation_stack_enabled(descriptor, value)
