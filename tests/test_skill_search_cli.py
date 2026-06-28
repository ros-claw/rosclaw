"""Tests for the builtin skill search CLI."""

from __future__ import annotations

import argparse
import json

import pytest

from rosclaw.skill.cli import add_skill_hub_parsers, cmd_skill_search


class TestSkillSearchCli:
    """Cover ``rosclaw skill search`` including builtin RealSense skills."""

    @pytest.fixture
    def search_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_skill_hub_parsers(subparsers)
        return parser

    def test_search_lists_realsense_builtins(self, search_parser, capsys):
        args = search_parser.parse_args(["search", "--json"])
        assert cmd_skill_search(args) == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        builtin_names = {s["name"] for s in data["builtin"]}
        assert "realsense_capture_rgbd" in builtin_names
        assert "scene_risk_scan" in builtin_names

    def test_search_text_output_includes_realsense(self, search_parser, capsys):
        args = search_parser.parse_args(["search"])
        assert cmd_skill_search(args) == 0
        out = capsys.readouterr().out
        assert "realsense_capture_rgbd" in out
        assert "RealSense RGB-D Capture" in out
