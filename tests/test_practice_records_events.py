"""Verify that practice start/run record real events when a skill is supplied."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.body.service import BodyInstanceService
from rosclaw.cli import cmd_practice_run, cmd_practice_start
from rosclaw.firstboot.workspace import init_workspace


def _png_header(width: int = 640, height: int = 480) -> bytes:
    """Minimal PNG IHDR chunk that satisfies the CLI's dimension parser."""
    magic = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    length = struct.pack(">I", len(ihdr_data))
    chunk_type = b"IHDR"
    # CRC isn't validated by the CLI fallback, but include a placeholder.
    crc = b"\x00\x00\x00\x00"
    return magic + length + chunk_type + ihdr_data + crc


def _make_color_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_header(640, 480))


@pytest.fixture
def practice_home(tmp_path, monkeypatch):
    """Create an isolated ROSClaw home with a linked RealSense D405 body."""
    home = tmp_path / "rosclaw_home"
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    init_workspace(home)
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


def _fake_execute(self, skill_name: str, parameters: dict | None = None):
    """SkillExecutor.execute replacement that writes a fake color frame."""
    output_dir = Path(parameters.get("output_dir", "/tmp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    color_path = output_dir / "color.png"
    _make_color_frame(color_path)
    return {
        "status": "success",
        "handler_result": {
            "artifacts": {"color": str(color_path)},
            "metrics": {"latency_ms": 12.3},
        },
    }


def _run_args(practice_home: Path, **overrides) -> SimpleNamespace:
    defaults = {
        "robot": "d405_lab_01",
        "skill": "realsense_capture_rgbd",
        "provider": None,
        "capability": "vlm.risk_assessment",
        "output_root": str(practice_home / "practice_output"),
        "data_root": None,
        "robot_type": None,
        "task": None,
        "workspace": str(practice_home),
        "json": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _start_args(practice_home: Path, **overrides) -> SimpleNamespace:
    defaults = {
        "robot": "d405_lab_01",
        "robot_type": None,
        "task": None,
        "skill": "realsense_capture_rgbd",
        "provider": None,
        "capability": "vlm.risk_assessment",
        "sources": "agent,runtime",
        "mock": False,
        "duration": "1s",
        "seekdb": False,
        "data_root": str(practice_home / "practice_output"),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _event_types(session_dir: Path) -> list[str]:
    timeline_jsonl = session_dir / "timeline.jsonl"
    if not timeline_jsonl.exists():
        return []
    types = []
    for line in timeline_jsonl.read_text(encoding="utf-8").strip().splitlines():
        if line:
            types.append(json.loads(line)["event_type"])
    return types


def test_practice_run_records_real_events(practice_home, monkeypatch):
    monkeypatch.setattr(
        "rosclaw.skill_manager.executor.SkillExecutor.execute",
        _fake_execute,
    )

    args = _run_args(practice_home)
    assert cmd_practice_run(args) == 0

    session_dir = next((practice_home / "practice_output" / "sessions").iterdir())
    types = _event_types(session_dir)

    assert "runtime.start" in types
    assert "skill.start" in types
    assert "skill.result" in types
    assert "rgbd_frame" in types
    assert "decision" in types
    assert "runtime.stop" in types
    assert len(types) >= 6

    # Frame artifact was copied into the episode.
    assert (session_dir / "artifacts" / "frames" / "color_000001.png").exists()


def test_practice_start_records_real_events(practice_home, monkeypatch):
    monkeypatch.setattr(
        "rosclaw.skill_manager.executor.SkillExecutor.execute",
        _fake_execute,
    )

    args = _start_args(practice_home)
    assert cmd_practice_start(args) == 0

    session_dir = next((practice_home / "practice_output" / "sessions").iterdir())
    types = _event_types(session_dir)

    assert "runtime.start" in types
    assert "skill.start" in types
    assert "skill.result" in types
    assert "rgbd_frame" in types
    assert "decision" in types
    # The start command keeps the session open; runtime.stop is emitted when
    # the session is finalized by the coordinator.
    assert len(types) >= 5

    summary = json.loads((session_dir / "episode.json").read_text())
    assert summary["event_count"] >= 5
    assert summary["outcome"] == "SUCCESS"


def test_practice_start_without_skill_records_zero_events(practice_home):
    args = _start_args(practice_home, skill=None, duration="1s")
    assert cmd_practice_start(args) == 0

    session_dir = next((practice_home / "practice_output" / "sessions").iterdir())
    types = _event_types(session_dir)
    assert types == []

    summary = json.loads((session_dir / "episode.json").read_text())
    assert summary["event_count"] == 0
    assert summary["outcome"] == "FAILED"
    assert "zero_events" in summary["failure_labels"]
