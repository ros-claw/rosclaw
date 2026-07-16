"""Verify that the runtime source always writes lifecycle events."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.cli import cmd_practice_start
from rosclaw.firstboot.workspace import init_workspace


def _start_args(practice_home: Path, **overrides) -> SimpleNamespace:
    defaults = {
        "robot": "d405_lab_01",
        "robot_type": None,
        "task": None,
        "skill": None,
        "provider": None,
        "capability": "vlm.risk_assessment",
        "sources": "runtime",
        "mock": False,
        "duration": "1s",
        "seekdb": False,
        "data_root": str(practice_home / "practice_output"),
        "sample_hz": 1.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def practice_home(tmp_path, monkeypatch):
    """Create an isolated ROSClaw home with a linked RealSense D405 body."""
    home = tmp_path / "rosclaw_home"
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    init_workspace(home)
    from rosclaw.body.service import BodyInstanceService

    service = BodyInstanceService(workspace=home)
    service.create_or_init(
        robot="realsense-d405",
        name="d405_lab_01",
        nickname="d405_lab_01",
        mode="single",
        update_registry=True,
        switch_active=True,
        render_agent_view=True,
        force=True,
    )
    return home


def _event_types(session_dir: Path) -> list[str]:
    events_jsonl = session_dir / "raw" / "events.jsonl"
    timeline_jsonl = session_dir / "timeline.jsonl"
    target = events_jsonl if events_jsonl.exists() else timeline_jsonl
    if not target.exists():
        return []
    types = []
    for line in target.read_text(encoding="utf-8").strip().splitlines():
        if line:
            types.append(json.loads(line)["event_type"])
    return types


def test_practice_runtime_source_writes_lifecycle_events(practice_home):
    """A runtime-only session must record runtime.start and runtime.stop."""
    args = _start_args(practice_home)
    assert cmd_practice_start(args) == 0

    session_dir = next((practice_home / "practice_output" / "sessions").iterdir())
    types = _event_types(session_dir)

    assert "runtime.start" in types
    assert "runtime.stop" in types
    assert len(types) >= 2

    episode = json.loads((session_dir / "episode.json").read_text())
    assert episode["event_count"] >= 2
    assert episode["outcome"] == "SUCCESS"


def test_practice_default_camera_skill_when_none_provided(practice_home, monkeypatch):
    """If camera source is enabled without --skill, default to realsense_capture_rgbd."""

    def _fake_execute(self, skill_name: str, parameters: dict | None = None):
        output_dir = Path(parameters.get("output_dir", "/tmp"))
        output_dir.mkdir(parents=True, exist_ok=True)
        color_path = output_dir / "color.png"
        color_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        return {
            "status": "success",
            "handler_result": {
                "artifacts": {"color": str(color_path)},
                "metrics": {"latency_ms": 10.0},
            },
        }

    monkeypatch.setattr(
        "rosclaw.skill_manager.executor.SkillExecutor.execute",
        _fake_execute,
    )

    args = _start_args(practice_home, skill=None, sources="camera,runtime", duration="1s")
    assert cmd_practice_start(args) == 0

    session_dir = next((practice_home / "practice_output" / "sessions").iterdir())
    types = _event_types(session_dir)
    assert "skill.start" in types
    assert "rgbd_frame" in types
    assert "decision" in types

    episode = json.loads((session_dir / "episode.json").read_text())
    assert episode["outcome"] == "SUCCESS"
    assert episode["event_count"] >= 4
