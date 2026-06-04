"""CLI report and demo tests."""
import os
import shutil
import subprocess
import sys
import tempfile
import pytest


class TestCLIReport:
    """AUTO-CLI-004/005: CLI report and dashboard export."""

    def test_cli_report_markdown_output(self):
        """AUTO-CLI-004: rosclaw auto report --format md writes markdown file."""
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "init",
             "--task", "demo_report", "--robot", "panda", "--skill", "pick_v1"],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
        )
        assert result.returncode == 0

        out_path = tempfile.mktemp(suffix=".md")
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "report",
             "--task", "demo_report", "--format", "md", "--output", out_path],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
        )
        assert result.returncode == 0
        assert os.path.exists(out_path)
        with open(out_path) as f:
            content = f.read()
        assert "Evolution Report" in content
        os.unlink(out_path)

    def test_cli_report_json_output(self):
        """AUTO-CLI-005: rosclaw auto report --format json writes dashboard JSON."""
        out_path = tempfile.mktemp(suffix=".json")
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "report",
             "--task", "demo_report", "--format", "json", "--output", out_path],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
        )
        assert result.returncode == 0
        assert os.path.exists(out_path)
        with open(out_path) as f:
            content = f.read()
        assert "summary" in content
        assert "champions" in content
        os.unlink(out_path)

    def test_demo_script_runs(self):
        """AUTO-DEMO-001: demo script runs without crash."""
        env = os.environ.copy()
        env["PYTHONPATH"] = "/home/ubuntu/rosclaw/rosclaw_auto"
        result = subprocess.run(
            [sys.executable, "demo/auto_demo.py"],
            capture_output=True, text=True, cwd="/home/ubuntu/rosclaw/rosclaw_auto",
            env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "=== ROSClaw-Auto v1.0 Demo ===" in result.stdout
        assert "Champion promoted" in result.stdout or "Registered DeadEnd" in result.stdout
