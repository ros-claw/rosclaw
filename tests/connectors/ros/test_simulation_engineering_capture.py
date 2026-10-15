"""Synthetic SDK role and freshness boundaries; no physical repair claim."""

import importlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

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
            "j100_navigation_onboarding",
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
    "radius,age,polygon,frame,publishers,expected",
    [
        (0.2, 0.1, "[]", "base", ["/costmap"], "BLOCKED"),
        (0.4, 0.1, "[]", "base", ["/costmap"], "PROPOSED"),
        (0.2, 0.6, "[]", "base", ["/costmap"], "PROPOSED"),
        (0.2, -0.1, "[]", "base", ["/costmap"], "PROPOSED"),
        (0.2, 0.1, "[[1,1]]", "base", ["/costmap"], "PROPOSED"),
        (0.2, 0.1, "[]", "other", ["/costmap"], "PROPOSED"),
        (0.2, 0.1, "[]", "base", ["/costmap", "/other"], "PROPOSED"),
        (float("nan"), 0.1, "[]", "base", ["/costmap"], "PROPOSED"),
    ],
)
def test_geometry_failure_requires_original_unambiguous_current_parameters(
    module, radius, age, polygon, frame, publishers, expected
):
    now = datetime.now(UTC)
    review = {
        "status": "PROPOSED",
        "reviewed_at": now.isoformat(),
        "source_snapshot_hash": "synthetic",
        "candidate": {
            "geometry": {"complete": True, "physical_radius_m": 0.369},
            "frames": {"base": "base"},
        },
    }
    native = {
        "graph": {"topics": [{"name": "/costmap/data", "publishers": publishers}]},
        "observations": {
            "node_parameters": {
                "/costmap": {
                    "robot_radius": radius,
                    "footprint": polygon,
                    "robot_base_frame": frame,
                }
            },
            "parameter_captured_at": {"/costmap": (now - timedelta(seconds=age)).isoformat()},
        },
    }
    result = module.review_engineering_failure(
        native, review, {"costmap_topics": ["/costmap/data"]}
    )
    assert result["status"] == expected
    assert result["original_candidate_status"] == review["status"] == "PROPOSED"
    assert result["authorization"] is result["binding_verified"] is False


@pytest.mark.parametrize(
    "native_age,physics_age,expected",
    [(0.1, 0.1, True), (0.6, 0.1, False), (0.1, 0.4, False), (-1, 0.1, False)],
)
def test_original_capture_requires_current_native_and_physics(
    module, native_age, physics_age, expected
):
    now = datetime.now(UTC)
    state = {
        "native": {
            "captured_at": (now - timedelta(seconds=native_age)).isoformat(),
            "completeness": {"graph": True},
        },
        "physics": {"captured_at": (now - timedelta(seconds=physics_age)).isoformat()},
    }
    assert module.current_capture(state) is expected


@pytest.mark.parametrize(
    "case,candidate_status,l1",
    [
        ("map_to_odom_missing", "UNKNOWN", "NOT_READY"),
        ("j100_navigation_onboarding", "PROPOSED", "NOT_EVALUATED"),
    ],
)
def test_unready_capture_preserves_missing_body_and_execution_config(
    module, tmp_path, monkeypatch, case, candidate_status, l1
):
    sink = {
        "run_id": "owned-run",
        "body_snapshot_hash": None,
        "terminal": False,
        "action_id": None,
        "daemon_instance_id": None,
        "lease_expires_ns": 0,
        "lease_updates": 0,
        "received_commands": 0,
        "published_velocity_messages": 0,
    }
    (tmp_path / "governed-sink-status.json").write_text(json.dumps(sink))
    (tmp_path / "initial-source-witness-observer-sources.json").write_text(
        json.dumps({"run_id": "owned-run"})
    )
    (tmp_path / "engineering-original-requirements.json").write_text(
        json.dumps({"costmap_topics": []})
    )
    stamp = datetime.now(UTC).isoformat()
    state = {
        "native": {"captured_at": stamp, "completeness": {"graph": True}},
        "physics": {"captured_at": stamp, "observation_complete": True, "body_snapshot_hash": None},
        "review": {
            "status": candidate_status,
            "candidate": {},
            "reviewed_at": stamp,
            "source_snapshot_hash": "synthetic",
        },
        "model": SimpleNamespace(to_dict=lambda: {"synthetic": True}),
    }
    clock = iter([0, 3])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))

    def spin(condition, _seconds):
        if (tmp_path / "engineering-capture-ready.json").exists():
            (tmp_path / "host_capture_ack.json").write_text("{}")
        assert condition()

    def save(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    module.capture_failure(tmp_path, settings(case)["engineering_capture"], state, spin, save)
    marker = json.loads((tmp_path / "engineering-capture-ready.json").read_text())
    assert marker["L1"] == l1 and marker["L2"] == "NOT_RUN"
    assert marker["body_snapshot_hash"] is None and marker["navigation_workspace_compiled"] is False
    assert not (tmp_path / "execution_config.json").exists()


def test_transient_incomplete_graph_is_not_mistaken_for_the_declared_engineering_fault(module):
    stamp = datetime.now(UTC).isoformat()
    state = {
        "native": {"captured_at": stamp, "completeness": {"graph": False}},
        "physics": {"captured_at": stamp},
    }
    assert module.current_capture(state) is False
