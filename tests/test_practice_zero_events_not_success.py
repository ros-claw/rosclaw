"""Tests that a practice session with zero events does not report SUCCESS."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from rosclaw.cli import cmd_practice_run, cmd_practice_validate
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator


class TestPracticeZeroEventsNotSuccess:
    """Cover the no-events / skill-failure outcome rules."""

    def test_coordinator_zero_events_reports_failed(self, tmp_path):
        """A session that captures nothing must be FAILED, not SUCCESS."""
        config = PracticeConfig(
            robot_id="d405_lab_01",
            robot_type="realsense_d405",
            task_id="test",
            skill_id="realsense_capture_rgbd",
            data_root=str(tmp_path),
            sources=SourceConfig(camera=True, sandbox=True),
            publish_to_event_bus=False,
        )
        coordinator = PracticeCoordinator(config)
        coordinator.initialize()
        coordinator.start()
        coordinator.stop()

        summary = coordinator.summary
        assert summary is not None
        assert summary.outcome == "FAILED"
        assert summary.failure_labels == ["zero_events"]
        assert summary.event_count == 0

        # The on-disk manifest and episode.json must reflect the failure.
        session_dir = Path(summary.artifact_dir)
        episode = json.loads((session_dir / "episode.json").read_text())
        assert episode["outcome"] == "FAILED"
        assert episode["failure_labels"] == ["zero_events"]

        manifest_path = session_dir / "manifest.yaml"
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text())
        assert manifest["status"]["outcome"] == "FAILED"
        assert manifest["status"]["failure_labels"] == ["zero_events"]

    def test_cli_run_skill_failure_records_zero_events_and_fails(
        self,
        linked_realsense_workspace,
        tmp_path,
    ):
        """If the skill handler returns an error, the episode must fail with zero events."""
        from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry

        def _failing_load_builtins(registry=None):
            if registry is None:
                registry = SkillRegistry()
            registry.register(
                SkillEntry(
                    name="realsense_capture_rgbd",
                    description="Failing RealSense skill stub",
                    skill_type="programmed",
                    handler=lambda _params: {
                        "status": "error",
                        "reason": "No RealSense MCP server installed or healthy",
                    },
                    metadata={"builtin": True},
                    version="1.0.0",
                )
            )
            return registry, []

        import rosclaw.skill.builtins as builtins_module

        original_load_builtins = builtins_module.load_builtins
        builtins_module.load_builtins = _failing_load_builtins
        try:
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

            assert cmd_practice_run(args) == 1

            session_dir = next((output_root / "sessions").iterdir())
            episode = json.loads((session_dir / "episode.json").read_text())
            assert episode["outcome"] == "FAILED"
            assert episode["event_count"] == 0
            assert "zero_events" in episode["failure_labels"]

            validate_args = SimpleNamespace(
                episode_id=session_dir.name,
                data_root=str(output_root),
                strict=False,
                json=False,
            )
            assert cmd_practice_validate(validate_args) == 1
        finally:
            builtins_module.load_builtins = original_load_builtins
