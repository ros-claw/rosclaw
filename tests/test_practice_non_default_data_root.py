"""Tests for ``rosclaw practice run`` with a non-default output root."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.cli import cmd_practice_run, cmd_practice_validate


class TestPracticeNonDefaultDataRoot:
    """Ensure the ``--output-root`` flag fully overrides the default data root."""

    @pytest.fixture
    def fake_realsense_skill(self, monkeypatch, dummy_png):
        from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry

        def _fake_run(params):
            output_dir = Path(params.get("output_dir") or "./capture")
            output_dir.mkdir(parents=True, exist_ok=True)
            color = output_dir / "color.png"
            shutil.copy(dummy_png, color)
            return {
                "status": "success",
                "skill": "realsense_capture_rgbd",
                "artifacts": {"color": str(color)},
                "metrics": {"latency_ms": 12.0, "usb_mode": "USB3", "degraded": False},
            }

        def _fake_load_builtins(registry=None):
            if registry is None:
                registry = SkillRegistry()
            registry.register(
                SkillEntry(
                    name="realsense_capture_rgbd",
                    description="Fake RealSense RGB-D capture skill",
                    skill_type="programmed",
                    handler=_fake_run,
                    metadata={"builtin": True},
                    version="1.0.0",
                )
            )
            return registry, []

        monkeypatch.setattr("rosclaw.skill.builtins.load_builtins", _fake_load_builtins)
        monkeypatch.setattr("rosclaw.cli._image_dimensions", lambda _path: (640, 480))

    def test_run_uses_output_root_and_validate_finds_it(
        self,
        linked_realsense_workspace,
        fake_realsense_skill,
        tmp_path,
    ):
        custom_root = tmp_path / "custom_practice"
        args = SimpleNamespace(
            robot="d405_lab_01",
            robot_type=None,
            task="test_task",
            skill="realsense_capture_rgbd",
            provider=None,
            capability="vlm.risk_assessment",
            output_root=str(custom_root),
            data_root=None,
            workspace=str(linked_realsense_workspace),
            json=False,
        )

        assert cmd_practice_run(args) == 0

        assert custom_root.exists()
        sessions_dir = custom_root / "sessions"
        assert sessions_dir.exists()
        session_dir = next(sessions_dir.iterdir())
        assert (session_dir / "episode.json").exists()
        assert (session_dir / "raw" / "events.jsonl").exists()

        # Validate must locate the episode using the same custom root.
        validate_args = SimpleNamespace(
            episode_id=session_dir.name,
            data_root=str(custom_root),
            strict=False,
            json=False,
        )
        assert cmd_practice_validate(validate_args) == 0

    def test_run_ignores_default_root_when_output_root_given(
        self,
        linked_realsense_workspace,
        fake_realsense_skill,
        tmp_path,
    ):
        default_root = tmp_path / "default_root"
        custom_root = tmp_path / "custom_root"
        # Pre-create default root to prove it stays empty.
        default_root.mkdir(parents=True, exist_ok=True)

        args = SimpleNamespace(
            robot="d405_lab_01",
            robot_type=None,
            task=None,
            skill="realsense_capture_rgbd",
            provider=None,
            capability="vlm.risk_assessment",
            output_root=str(custom_root),
            data_root=str(default_root),
            workspace=str(linked_realsense_workspace),
            json=False,
        )

        assert cmd_practice_run(args) == 0
        assert not any(default_root.iterdir())
        assert any((custom_root / "sessions").iterdir())
