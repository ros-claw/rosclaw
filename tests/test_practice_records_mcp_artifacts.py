"""Tests that ``rosclaw practice run`` records MCP skill artifacts and events."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rosclaw.cli import cmd_practice_run, cmd_practice_validate


class TestPracticeRecordsMcpArtifacts:
    """End-to-end exercise of the Phase F practice run flow."""

    def test_run_records_camera_provider_and_sandbox_events(
        self,
        linked_realsense_workspace,
        fake_realsense_skill,
        tmp_path,
    ):
        output_root = tmp_path / "episode"
        args = SimpleNamespace(
            robot="d405_lab_01",
            robot_type=None,
            task="realsense_inspection",
            skill="realsense_capture_rgbd",
            provider="cosmos-reason2-lan",
            capability="vlm.risk_assessment",
            output_root=str(output_root),
            data_root=None,
            workspace=str(linked_realsense_workspace),
            json=False,
        )

        assert cmd_practice_run(args) == 0

        # Discover the session directory.
        sessions_dir = output_root / "sessions"
        assert sessions_dir.exists()
        session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
        assert len(session_dirs) == 1
        session_dir = session_dirs[0]

        # Episode metadata was produced.
        episode_path = session_dir / "episode.json"
        assert episode_path.exists()
        episode = json.loads(episode_path.read_text())
        assert episode["outcome"] == "SUCCESS"
        assert episode["event_count"] > 0
        assert episode["robot_id"] == "d405_lab_01"

        # Events JSONL has camera, provider, and sandbox events.
        events_path = session_dir / "raw" / "events.jsonl"
        assert events_path.exists()
        events = [json.loads(line) for line in events_path.read_text().strip().splitlines()]
        sources = {ev["source"] for ev in events}
        assert "camera" in sources
        assert "provider" in sources
        assert "sandbox" in sources

        camera_events = [ev for ev in events if ev["source"] == "camera"]
        assert camera_events[0]["event_type"] == "rgbd_frame"
        assert camera_events[0]["payload"]["rgb_ref"].endswith("color.png")

        # Provider artifact was written.
        provider_result = session_dir / "provider" / "provider_result.json"
        assert provider_result.exists()
        provider_data = json.loads(provider_result.read_text())
        assert provider_data["provider_id"] == "cosmos-reason2-lan"
        assert "normalized" in provider_data

        # Timeline mirrors raw events.
        timeline_path = session_dir / "timeline.jsonl"
        assert timeline_path.exists()
        timeline = [json.loads(line) for line in timeline_path.read_text().strip().splitlines()]
        assert len(timeline) == len(events)

    def test_run_without_provider_records_camera_and_sandbox(
        self,
        linked_realsense_workspace,
        fake_realsense_skill,
        tmp_path,
    ):
        output_root = tmp_path / "episode"
        args = SimpleNamespace(
            robot="d405_lab_01",
            robot_type=None,
            task=None,
            skill="realsense_capture_rgbd",
            provider=None,
            capability="vlm.risk_assessment",
            output_root=str(output_root),
            data_root=None,
            workspace=str(linked_realsense_workspace),
            json=False,
        )

        assert cmd_practice_run(args) == 0

        session_dir = next((output_root / "sessions").iterdir())
        events = json.loads((session_dir / "episode.json").read_text())
        assert events["outcome"] == "SUCCESS"

        sources = {ev["source"] for ev in [
            json.loads(line)
            for line in (session_dir / "raw" / "events.jsonl").read_text().strip().splitlines()
        ]}
        assert "camera" in sources
        assert "sandbox" in sources
        assert "provider" not in sources

        # Strict validation should fail without a provider event.
        validate_args = SimpleNamespace(
            episode_id=session_dir.name,
            data_root=str(output_root),
            strict=True,
            json=False,
        )
        assert cmd_practice_validate(validate_args) == 1
