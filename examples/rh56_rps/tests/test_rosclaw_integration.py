"""Integration tests for the ROSClaw-native RH56 RPS demo."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rosclaw.body.resolver import BodyResolver

# Importing registers the runtime handler side effect.
from rosclaw_rps.rosclaw_integration import RosclawRpsSession
from rosclaw.skill_manager.loader import SkillLoader
from rosclaw.skill_manager.registry import SkillRegistry


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "dual"
MANIFEST_PATH = CONFIG_DIR / "skills" / "rh56_rps.skill.yaml"


def _rosclaw_config(tmp_path: Path) -> dict:
    return {
        "practice": {
            "robot_id": "rh56_rps_robot",
            "robot_type": "dual_rh56",
            "task_id": "rh56_rps",
            "task_name": "RH56 RPS test",
            "skill_id": "rh56_rps",
            "data_root": str(tmp_path / "practice"),
            "sources": {"agent": True, "runtime": True},
            "recorder": {"jsonl_enabled": True},
            "seekdb": {"enabled": False},
        },
        "memory": {"backend": "memory"},
        "body": {"workspace": str(CONFIG_DIR / "body")},
        "skill_manifest": str(MANIFEST_PATH.relative_to(CONFIG_DIR)),
    }


def _event_types(events_jsonl: Path) -> set[str]:
    types: set[str] = set()
    with open(events_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            types.add(data.get("event_type", ""))
    return types


@pytest.fixture
def session(tmp_path: Path) -> RosclawRpsSession:
    cfg = _rosclaw_config(tmp_path)
    sess = RosclawRpsSession(config_dir=CONFIG_DIR, rosclaw_config=cfg)
    sess.initialize()
    sess.start()
    yield sess
    sess.stop()


def test_rosclaw_session_lifecycle(session: RosclawRpsSession, tmp_path: Path) -> None:
    practice_id = session.active_practice_id
    assert practice_id is not None
    assert practice_id.startswith("prac_")

    data_root = tmp_path / "practice"
    session_dir = data_root / "sessions" / practice_id
    assert session_dir.exists()
    assert (session_dir / "raw" / "events.jsonl").exists()
    assert (session_dir / "manifest.yaml").exists()
    assert (data_root / "indexes" / "practice_catalog.sqlite").exists()


def test_demo_body_is_linked_and_compatible() -> None:
    resolver = BodyResolver(workspace=CONFIG_DIR)
    assert resolver.is_linked(), "Demo body workspace should be linked"
    report = resolver.check_skill_compatibility("rh56_rps", "1.0.0")
    assert report.status in {"compatible", "degraded"}, f"Unexpected status: {report.status}"


def test_skill_registry_loads_rh56_rps() -> None:
    registry = SkillRegistry()
    registry.initialize()
    SkillLoader(registry).load_skill_manifest(MANIFEST_PATH)

    entry = registry.get("rh56_rps")
    assert entry is not None
    assert entry.name == "rh56_rps"
    assert entry.version == "1.0.0"
    assert "dexterous_hand_control" in entry.requirements.get("capabilities", {}).get("all_of", [])


def test_mock_round_emits_events(session: RosclawRpsSession, tmp_path: Path) -> None:
    result = session.run_skill({"mode": "mock", "rounds": 1, "auto": True})
    assert result.get("handler_result", {}).get("status") == "success"

    practice_id = session.active_practice_id
    events_jsonl = tmp_path / "practice" / "sessions" / practice_id / "raw" / "events.jsonl"
    events_by_type: dict[str, dict] = {}
    with open(events_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            events_by_type[data.get("event_type", "")] = data

    expected = {
        "rps.run.started",
        "rps.round.started",
        "rps.robot_choice_committed",
        "rps.human.gesture_detected",
        "rps.gesture.executed",
        "rps.round.resolved",
        "rps.run.summary",
        "rps.run.completed",
        "skill.execution.start",
        "skill.execution.complete",
        "runtime.start",
    }
    missing = expected - set(events_by_type)
    assert not missing, f"Missing event types: {missing}"

    # Lifecycle events must carry meaningful payloads.
    start_ev = events_by_type["skill.execution.start"]
    assert start_ev["payload"].get("skill_name") == "rh56_rps"
    assert "parameters" in start_ev["payload"]

    complete_ev = events_by_type["skill.execution.complete"]
    assert complete_ev["payload"].get("skill_name") == "rh56_rps"
    assert "result" in complete_ev["payload"]

    practice_start = events_by_type["practice.session_started"]
    assert practice_start["payload"].get("practice_id") == practice_id


def test_practice_lifecycle_events_on_bus(tmp_path: Path) -> None:
    cfg = _rosclaw_config(tmp_path)
    sess = RosclawRpsSession(config_dir=CONFIG_DIR, rosclaw_config=cfg)
    try:
        sess.initialize()
        sess.start()
        # practice.start/stop are control-plane events consumed by the recorder;
        # they are visible on the RuntimeBus but are not duplicated to JSONL.
        start_history = sess.runtime_bus.get_history("practice.start", limit=5)
        assert len(start_history) == 1
        assert start_history[0].payload.get("practice_id") == sess.active_practice_id
        sess.run_skill({"mode": "mock", "rounds": 1, "auto": True})
    finally:
        sess.stop()

    stop_history = sess.runtime_bus.get_history("practice.stop", limit=5)
    assert len(stop_history) == 1
    assert stop_history[0].payload.get("outcome") in {"SUCCESS", "FAILED"}


def test_runtime_stop_written_after_stop(tmp_path: Path) -> None:
    cfg = _rosclaw_config(tmp_path)
    sess = RosclawRpsSession(config_dir=CONFIG_DIR, rosclaw_config=cfg)
    sess.initialize()
    sess.start()
    practice_id = sess.active_practice_id
    sess.run_skill({"mode": "mock", "rounds": 1, "auto": True})
    sess.stop()

    events_jsonl = tmp_path / "practice" / "sessions" / practice_id / "raw" / "events.jsonl"
    types = _event_types(events_jsonl)
    assert "runtime.stop" in types
    assert "runtime.start" in types


def test_practice_recorder_writes_jsonl(session: RosclawRpsSession, tmp_path: Path) -> None:
    session.run_skill({"mode": "mock", "rounds": 1, "auto": True})
    practice_id = session.active_practice_id
    events_jsonl = tmp_path / "practice" / "sessions" / practice_id / "raw" / "events.jsonl"

    resolved = []
    with open(events_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("event_type") == "rps.round.resolved":
                resolved.append(data)
    assert len(resolved) == 1
    assert resolved[0]["payload"]["round"]["result"] in {"robot_win", "robot_lose", "draw"}


def test_continuous_telemetry_is_recorded(session: RosclawRpsSession, tmp_path: Path) -> None:
    """Continuous joint-state telemetry should be written to the practice flywheel."""
    session.run_skill({"mode": "mock", "rounds": 1, "auto": True})
    practice_id = session.active_practice_id
    events_jsonl = tmp_path / "practice" / "sessions" / practice_id / "raw" / "events.jsonl"

    telemetry = []
    with open(events_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("event_type") == "rps.telemetry":
                telemetry.append(data)

    assert len(telemetry) >= 1, "expected at least one rps.telemetry event"
    sample = telemetry[0]
    assert "right" in sample["payload"] or "left" in sample["payload"]
    hand = sample["payload"].get("right") or sample["payload"].get("left")
    assert hand is not None
    assert "angle_actual" in hand
    assert "summary" in hand


def test_sqlite_seekdb_backend(tmp_path: Path) -> None:
    cfg = _rosclaw_config(tmp_path)
    cfg["memory"] = {"backend": "sqlite", "db_path": str(tmp_path / "seekdb.sqlite")}

    sess = RosclawRpsSession(config_dir=CONFIG_DIR, rosclaw_config=cfg)
    try:
        sess.initialize()
        sess.start()
        result = sess.run_skill({"mode": "mock", "rounds": 1, "auto": True})
        assert result.get("handler_result", {}).get("status") == "success"
    finally:
        sess.stop()

    db_path = tmp_path / "seekdb.sqlite"
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name, success_count, failure_count FROM skill_metadata WHERE skill_id = ?",
            ("rh56_rps",),
        ).fetchall()
        assert len(rows) == 1
        name, success_count, failure_count = rows[0]
        assert name == "rh56_rps"
        assert success_count + failure_count >= 1
    finally:
        conn.close()
