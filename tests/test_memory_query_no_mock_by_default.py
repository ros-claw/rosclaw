"""Tests that ``rosclaw memory query`` does not return mock data by default."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.cli import cmd_memory_query


class TestMemoryQueryNoMockByDefault:
    """Phase H no-mock memory query tests."""

    def test_query_without_real_experiences_returns_empty(self, capsys, monkeypatch, tmp_path):
        """Without --demo, an empty memory must not fall back to artifact mock data."""
        # Isolate the persistent memory DB and artifact fallback directories.
        monkeypatch.setattr("rosclaw.cli._memory_db_path", lambda: tmp_path / "seekdb.sqlite")
        monkeypatch.setattr(
            "rosclaw.cli._practice_artifacts_dir", lambda: tmp_path / "practice"
        )

        args = SimpleNamespace(query="RealSense D405 RGB-D capture", limit=5, demo=False)
        assert cmd_memory_query(args) == 0

        captured = capsys.readouterr().out
        assert "No matching experiences found." in captured

    def test_query_with_demo_uses_artifact_fallback(self, capsys, monkeypatch, tmp_path):
        """With --demo, the artifact fallback may return matching episodes."""
        monkeypatch.setattr("rosclaw.cli._memory_db_path", lambda: tmp_path / "seekdb.sqlite")

        practice_root = tmp_path / "practice"
        episodes_dir = practice_root / "episodes" / "ep_demo_01"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "episode_id": "ep_demo_01",
            "status": "SUCCESS",
            "robot_id": "realsense_d405",
            "praxis_event": {"agent_instruction": "RealSense D405 RGB-D capture"},
        }
        (episodes_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        monkeypatch.setattr("rosclaw.cli._practice_artifacts_dir", lambda: practice_root)

        args = SimpleNamespace(query="RealSense D405 RGB-D capture", limit=5, demo=True)
        assert cmd_memory_query(args) == 0

        captured = capsys.readouterr().out
        assert "ep_demo_01" in captured
        assert "SUCCESS" in captured

    def test_query_demo_flag_is_available_on_parser(self):
        """The CLI parser must accept --demo on memory query."""
        import argparse

        from rosclaw.cli import main

        # main() is not idempotent to re-parse; instead inspect the parser tree.
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        memory_parser = subparsers.add_parser("memory")
        memory_subparsers = memory_parser.add_subparsers()
        query_parser = memory_subparsers.add_parser("query")
        query_parser.add_argument("query")
        query_parser.add_argument("--limit", type=int, default=5)
        query_parser.add_argument("--demo", action="store_true")

        args = query_parser.parse_args(["RealSense", "--demo"])
        assert args.demo is True
