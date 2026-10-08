"""Scene-controller offline contracts; SDK fixtures are not Native physics episodes."""

import hashlib
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.connectors.ros.test_physics_component_packets import packet

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "dynamic_scenario_test", ROOT / "dynamic_scenario.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def binding():
    return {
        "run_id": "synthetic_run",
        "body_snapshot_hash": "synthetic_body_hash",
        "attachment_hash": "synthetic_attachment_hash",
        "world_name": "fixture_world",
        "body_model_name": "anonymous_body",
        "obstacle_names": ["anonymous_blocker"],
        "scene_model_names": ["anonymous_body", "anonymous_blocker"],
        "mission_id": "fresh-mission",
    }


def source_row(p):
    wire = json.dumps(p, indent=1)
    return {
        "kind": "physics_snapshot_received",
        "run_id": p["run_id"],
        "sim_time_sec": p["sim_time_sec"],
        "captured_at": datetime.now(UTC).isoformat(),
        "payload": {
            "packet": p,
            "raw_packet_utf8": wire,
            "packet_sha256": hashlib.sha256(wire.encode()).hexdigest(),
        },
    }


def policy():
    return {
        "schema_version": "rosclaw.dynamic_fixture_scenario.v1",
        "case": "D2",
        "run_id": "synthetic_run",
        "mission_id": "fresh-mission",
        "obstacle_name": "anonymous_blocker",
        "target_xy": [0.25, 0.25],
        "introduce_after_cleaning_sim_sec": 25,
        "dwell_sim_sec": 20,
        "wall_timeout_sec": 900,
    }


def test_retained_packet_preserves_actual_original_bytes(monkeypatch):
    m = load(monkeypatch)
    p = packet()
    p["captured_at_unix_ns"] = time.time_ns()
    row = source_row(p)
    result = m.retained_packet(row, binding())
    assert result["packet_sha256"] == row["payload"]["packet_sha256"]
    assert result["packet"] == p


@pytest.mark.parametrize("fault", ["wire", "parsed", "run", "time", "source"])
def test_retained_packet_consistency_failure_refused(monkeypatch, fault):
    m = load(monkeypatch)
    p = packet()
    p["captured_at_unix_ns"] = time.time_ns()
    row = source_row(p)
    if fault == "wire":
        row["payload"]["raw_packet_utf8"] += " "
    elif fault == "parsed":
        row["payload"]["packet"] = {}
    elif fault == "run":
        row["run_id"] = "different"
    elif fault == "time":
        row["sim_time_sec"] += 1
    else:
        row["payload"]["packet"]["source"] = "agent"
    with pytest.raises(ValueError):
        m.retained_packet(row, binding())


@pytest.mark.parametrize(
    "key,value",
    [
        ("dwell_sim_sec", 9),
        ("dwell_sim_sec", 31),
        ("wall_timeout_sec", 1921),
        ("target_xy", [float("nan"), 0]),
        ("run_id", "old"),
        ("mission_id", "old"),
        ("case", "D6"),
        ("obstacle_name", "new"),
    ],
)
def test_scenario_requires_bounded_preregistered_identity(monkeypatch, key, value):
    m = load(monkeypatch)
    spec = policy()
    spec[key] = value
    with pytest.raises(ValueError):
        m.scenario_policy(spec, binding())


@pytest.mark.parametrize("actual_move", [True, False])
def test_d2_controller_requires_new_actual_positions_after_each_ack(
    tmp_path, monkeypatch, actual_move
):
    m = load(monkeypatch)
    b = binding()
    s = policy()
    (tmp_path / "scenario.json").write_text(json.dumps(s))
    fixture = {
        "schema_version": "rosclaw.sim_physics_fixture.v1",
        "binding": b,
        "obstacles": [
            {"name": "anonymous_blocker", "pose": [5, 0, 0.1, 0, 0, 0], "box_size": [0.7, 0.7, 0.2]}
        ],
    }
    (tmp_path / "physics.json").write_text(json.dumps(fixture))
    (tmp_path / "body.json").write_text(
        json.dumps({"effective_body_hash": b["body_snapshot_hash"], "physical_radius_m": 0.25})
    )
    (tmp_path / "plan-events-1.jsonl").touch()
    active = {}
    moves = []
    position = [5, 0]
    tick = [0]

    class Cursor:
        def __init__(self, *a, **kw):
            pass

        def poll(self):
            tick[0] += 1
            p = packet()
            p.update(
                sequence=tick[0],
                sim_time_sec=tick[0] * 10.0,
                paused=False,
                captured_at_unix_ns=time.time_ns(),
            )
            p["body"]["world_pose"][:2] = [-1, -1]
            p["obstacles"][0]["world_pose"][:2] = position
            row = source_row(p)
            active.update(
                sample={
                    "x": -1,
                    "y": -1,
                    "time_sec": p["sim_time_sec"],
                    "captured_at": datetime.now(UTC).isoformat(),
                    "cleaning_enabled": True,
                    "collision_count": 0,
                    "observation_complete": True,
                    "physics_packet_sha256": row["payload"]["packet_sha256"],
                }
            )
            return [row]

    def service(name, kind, request):
        assert name == "set_pose" and kind == "gz.msgs.Pose"
        moves.append(request)
        if actual_move:
            position[:] = s["target_xy"] if len(moves) == 1 else [5, 0]
        return "data: true"

    monkeypatch.setattr(m, "AuditCursor", Cursor)
    monkeypatch.setattr(m, "observed", lambda: active["sample"])
    monkeypatch.setattr(m, "service", service)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    clock = [0.0]

    def monotonic():
        clock[0] += 0.1
        return clock[0]

    monkeypatch.setattr(m.time, "monotonic", monotonic)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scenario", "--directory", str(tmp_path), "--scenario", str(tmp_path / "scenario.json")],
    )
    if actual_move:
        m.main()
    else:
        with pytest.raises(ValueError, match="lacks independent actual pose confirmation"):
            m.main()
    rows = [
        json.loads(line)
        for line in (tmp_path / "dynamic-scenario-events.jsonl").read_text().splitlines()
    ]
    if not actual_move:
        assert len(moves) == 1
        assert rows[-1]["kind"] == "SCENARIO_FAILED"
        assert not any(r["kind"] == "ACTUAL_POSTUPDATE_POSITION_CONFIRMED" for r in rows)
        return
    assert len(moves) == 2
    confirmed = [r for r in rows if r["kind"] == "ACTUAL_POSTUPDATE_POSITION_CONFIRMED"]
    assert [r["actual_xy"] for r in confirmed] == [[0.25, 0.25], [5, 0]]
    assert rows[-1]["kind"] == "PERTURBATION_COMPLETE_REQUIRES_NATIVE_AND_CREDIT_VALIDATION"
    assert all(r["physical_acceptance"] == "NOT_VERIFIED" for r in rows)
