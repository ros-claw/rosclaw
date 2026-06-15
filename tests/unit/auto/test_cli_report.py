"""CLI report and demo tests."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# Resolve the rosclaw_auto checkout relative to this test file so the suite
# works regardless of where the repository is cloned.
def _find_rosclaw_auto() -> Path:
    start = Path(__file__).resolve().parent
    for parent in [start, *start.parents]:
        candidate = parent / "rosclaw_auto"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find rosclaw_auto directory. "
        "Make sure it exists as a sibling or ancestor of this test file."
    )


AUTO_REPO = _find_rosclaw_auto()


class TestCLIReport:
    """AUTO-CLI-004/005: CLI report and dashboard export."""

    def test_cli_report_markdown_output(self):
        """AUTO-CLI-004: rosclaw auto report --format md writes markdown file."""
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "init",
             "--task", "demo_report", "--robot", "panda", "--skill", "pick_v1"],
            capture_output=True, text=True, cwd=str(AUTO_REPO),
        )
        assert result.returncode == 0

        out_path = tempfile.mktemp(suffix=".md")
        result = subprocess.run(
            [sys.executable, "-m", "rosclaw_auto.cli", "report",
             "--task", "demo_report", "--format", "md", "--output", out_path],
            capture_output=True, text=True, cwd=str(AUTO_REPO),
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
            capture_output=True, text=True, cwd=str(AUTO_REPO),
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
        env["PYTHONPATH"] = str(AUTO_REPO)
        result = subprocess.run(
            [sys.executable, "demo/auto_demo.py"],
            capture_output=True, text=True, cwd=str(AUTO_REPO),
            env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "=== ROSClaw-Auto v1.0 Demo ===" in result.stdout
        assert "Champion promoted" in result.stdout or "Registered DeadEnd" in result.stdout
