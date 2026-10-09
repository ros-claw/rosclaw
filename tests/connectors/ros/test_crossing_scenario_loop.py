"""Run the actual scenario loop against synthetic original component callbacks."""

import importlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.connectors.ros.test_crossing_fixture import crossing_policy, swath_row
from tests.connectors.ros.test_dynamic_fixture_scenario import binding, policy, source_row
from tests.connectors.ros.test_physics_component_packets import packet


@pytest.mark.parametrize(
    "fault", [None, "ack_without_motion", "main_ends", "brush_off", "wrong_swath"]
)
def test_actual_loop_requires_small_independently_confirmed_moves_during_main(
    tmp_path, monkeypatch, fault
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    m = importlib.import_module("dynamic_scenario")
    crossing = importlib.import_module("crossing_fixture")
    b = {**binding(), "grid": {"frame_id": "map"}}
    s = {
        **policy(),
        "case": "D1",
        "target_xy": [0, -1],
        "dwell_sim_sec": 10,
        "introduce_after_cleaning_sim_sec": 1,
        "crossing_source": crossing_policy(),
    }
    fixture = {
        "schema_version": "rosclaw.sim_physics_fixture.v1",
        "binding": b,
        "obstacles": [
            {"name": "anonymous_blocker", "pose": [5, 0, 0.1, 0, 0, 0], "box_size": [0.7, 0.7, 0.2]}
        ],
    }
    (tmp_path / "physics.json").write_text(json.dumps(fixture))
    (tmp_path / "scenario.json").write_text(json.dumps(s))
    (tmp_path / "body.json").write_text(
        json.dumps({"effective_body_hash": b["body_snapshot_hash"], "physical_radius_m": 0.25})
    )
    (tmp_path / "plan-events-synthetic.jsonl").touch()
    actual_xy, moves, latest = [5, 0], [], {}
    tick = [0]
    swath = swath_row()
    if fault == "wrong_swath":
        swath["payload"]["frame_id"] = "odom"

    class Cursor:
        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            tick[0] += 1
            p = packet()
            p.update(
                sequence=tick[0],
                paused=False,
                sim_time_sec=tick[0] * 0.1,
                captured_at_unix_ns=time.time_ns(),
            )
            p["body"]["world_pose"][:2] = [-1, -1]
            p["obstacles"][0]["world_pose"][:2] = actual_xy
            row = source_row(p)
            latest.update(
                x=-1,
                y=-1,
                time_sec=p["sim_time_sec"],
                captured_at=datetime.now(UTC).isoformat(),
                cleaning_enabled=not (fault == "brush_off" and len(moves) > 4),
                collision_count=0,
                observation_complete=True,
                physics_packet_sha256=row["payload"]["packet_sha256"],
            )
            return [swath, row] if tick[0] == 1 else [row]

    class MainWindow:
        def __init__(self, *args):
            pass

        def poll(self):
            return (
                None
                if fault == "main_ends" and len(moves) > 4
                else {"action_id": "action", "nav_goal_id": "goal", "audit_event_sha256": "main"}
            )

    def service(name, kind, request):
        assert (name, kind) == ("set_pose", "gz.msgs.Pose")
        match = re.search(r"position \{ x: (\S+) y: (\S+) z:", request)
        target = [float(v) for v in match.groups()]
        moves.append(target)
        if fault != "ack_without_motion":
            actual_xy[:] = target
        return "data: true"

    monotonic = [0.0]

    def clock():
        monotonic[0] += 0.02
        return monotonic[0]

    monkeypatch.setattr(
        m, "time", SimpleNamespace(monotonic=clock, sleep=lambda _: None, time_ns=time.time_ns)
    )
    monkeypatch.setattr(m, "AuditCursor", Cursor)
    monkeypatch.setattr(m, "observed", lambda: latest)
    monkeypatch.setattr(m, "service", service)
    monkeypatch.setattr(crossing, "MainCoverageWindow", MainWindow)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scenario", "--directory", str(tmp_path), "--scenario", str(tmp_path / "scenario.json")],
    )
    if fault is None:
        m.main()
    else:
        with pytest.raises((ValueError, TimeoutError)):
            m.main()
    rows = [
        json.loads(line)
        for line in (tmp_path / "dynamic-scenario-events.jsonl").read_text().splitlines()
    ]
    confirmations = [row for row in rows if row["kind"] == "ACTUAL_POSTUPDATE_POSITION_CONFIRMED"]
    if fault is None:
        assert len(moves) == len(confirmations) == 42
        assert [row["crossing_index"] for row in confirmations[:-1]] == list(range(41))
        assert actual_xy == [5, 0]
        assert rows[-1]["kind"] == "PERTURBATION_COMPLETE_REQUIRES_NATIVE_AND_CREDIT_VALIDATION"
    else:
        assert rows[-1]["kind"] == "SCENARIO_FAILED"
        assert all(
            row["kind"] != "PERTURBATION_COMPLETE_REQUIRES_NATIVE_AND_CREDIT_VALIDATION"
            for row in rows
        )
