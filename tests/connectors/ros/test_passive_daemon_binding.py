"""Synthetic daemon health/pin rejection contracts, never physical proof."""

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_dual_supervision").review_passive_daemon_status


def healthy():
    pins = {
        "daemon_instance_id": "daemon_" + "a" * 32,
        "daemon_pid": 123,
        "daemon_uid": 1000,
        "robot_id": "synthetic-base",
    }
    status = {
        **pins,
        "running": True,
        "process_separated": True,
        "southbound_owner": "rosclawd",
        "client_peer": {"pid": 321},
        "runtime_state": "RUNNING",
        "supervision_state": "DISARMED",
        "watchdog": {"running": True, "healthy": True},
        "queue": {"QUEUED": 0, "RUNNING": 0},
        "permits": {"active": 0},
        "sessions": {"active": 0},
        "hardware_actions_executed": 0,
        "emergency_stop_latched": False,
        "recovery": {"required": False},
        "ledger": {"integrity_verified": True, "write_failed": False},
        "drivers": [],
        "registered_executors": [],
    }
    return status, pins


def test_only_current_independent_passive_daemon_satisfies_subcheck(checker):
    status, pins = healthy()
    assert checker(status, pins, client_pid=321) == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("daemon_instance_id", "daemon_" + "b" * 32),
        ("daemon_pid", 124),
        ("daemon_uid", 0),
        ("robot_id", "different-body"),
        ("running", False),
        ("process_separated", False),
        ("southbound_owner", "agent"),
        ("client_peer", {"pid": 999}),
        ("runtime_state", "STOPPED"),
        ("emergency_stop_latched", True),
        ("recovery", {"required": True}),
        ("ledger", {"integrity_verified": False, "write_failed": False}),
        ("ledger", {"integrity_verified": True, "write_failed": True}),
        ("ledger", None),
        ("drivers", ["unexpected"]),
        ("registered_executors", ["navigation.navigate_to_pose:SIMULATION"]),
    ],
)
def test_changed_generation_health_or_execution_path_is_rejected(checker, key, value):
    status, pins = healthy()
    original = copy.deepcopy(pins)
    status[key] = value
    assert checker(status, pins, client_pid=321)
    assert pins == original
