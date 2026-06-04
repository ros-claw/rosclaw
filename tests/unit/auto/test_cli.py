"""L9: CLI tests."""
import subprocess
import sys
import pytest


class TestCLI:
    """AUTO-CLI-001/002/003: CLI commands availability."""

    def test_cli_help(self):
        """AUTO-CLI-000: CLI --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "--help"],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
        )
        assert result.returncode == 0
        assert "rosclaw-auto" in result.stdout

    def test_cli_init(self):
        """AUTO-CLI-001: rosclaw auto init creates task."""
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "init",
             "--task", "pick_cube", "--robot", "panda", "--skill", "pick_v1"],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
        )
        assert result.returncode == 0
        assert "pick_cube" in result.stdout

    def test_cli_run_dry(self):
        """AUTO-CLI-002: rosclaw auto run --dry-run works."""
        # First create the task
        subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "init",
             "--task", "pick_cube", "--robot", "panda", "--skill", "pick_v1"],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
        )
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "run",
             "--task", "pick_cube", "--rounds", "2", "--dry-run"],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
        )
        assert result.returncode == 0

    def test_cli_report(self):
        """AUTO-CLI-003: rosclaw auto report works."""
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "report",
             "--task", "pick_cube"],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
        )
        assert result.returncode == 0
        assert "Evolution Report" in result.stdout or "report" in result.stdout.lower()

    def test_cli_champion(self):
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "champion",
             "--task", "pick_cube"],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
        )
        assert result.returncode == 0

    def test_cli_deadends(self):
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "deadends",
             "--task", "pick_cube"],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
        )
        assert result.returncode == 0
