"""CLI report and demo tests."""

import json


class TestCLIReport:
    """AUTO-CLI-004/005: CLI report and dashboard export."""

    def test_cli_report_markdown_output(self, run_auto_cli, tmp_path):
        """AUTO-CLI-004: rosclaw auto report --format md writes markdown file."""
        result = run_auto_cli(
            "init",
            "--task",
            "demo_report",
            "--robot",
            "panda",
            "--skill",
            "pick_v1",
        )
        assert result.returncode == 0

        out_path = tmp_path / "report.md"
        result = run_auto_cli(
            "report",
            "--task",
            "demo_report",
            "--format",
            "md",
            "--output",
            str(out_path),
        )
        assert result.returncode == 0
        content = out_path.read_text()
        assert "Evolution Report" in content

    def test_cli_report_json_output(self, run_auto_cli, tmp_path):
        """AUTO-CLI-005: rosclaw auto report --format json writes dashboard JSON."""
        init_result = run_auto_cli(
            "init",
            "--task",
            "demo_report",
            "--robot",
            "panda",
            "--skill",
            "pick_v1",
        )
        assert init_result.returncode == 0

        out_path = tmp_path / "report.json"
        result = run_auto_cli(
            "report",
            "--task",
            "demo_report",
            "--format",
            "json",
            "--output",
            str(out_path),
        )
        assert result.returncode == 0
        content = json.loads(out_path.read_text())
        assert "summary" in content
        assert "champions" in content

    def test_main_cli_routes_auto_workflow(self, run_auto_cli):
        """AUTO-CLI-006: the installed rosclaw CLI routes auto commands."""
        init_result = run_auto_cli(
            "auto",
            "init",
            "--task",
            "main_cli_demo",
            "--robot",
            "panda",
            "--skill",
            "pick_v1",
            module="rosclaw.cli",
        )
        assert init_result.returncode == 0, init_result.stderr
        assert "Created task" in init_result.stdout

        run_result = run_auto_cli(
            "auto",
            "run",
            "--task",
            "main_cli_demo",
            "--rounds",
            "1",
            "--dry-run",
            module="rosclaw.cli",
        )
        assert run_result.returncode == 0, run_result.stderr
        assert "Evolution complete" in run_result.stdout

    def test_main_cli_keeps_lerobot_runtime_lazy(self, run_isolated_python):
        """AUTO-CLI-007: unrelated commands do not load the optional LeRobot runtime."""
        result = run_isolated_python(
            "import sys; import rosclaw.cli; "
            "assert 'rosclaw.integrations.lerobot.cli' not in sys.modules; "
            "assert 'rosclaw.integrations.lerobot.runtime' not in sys.modules"
        )
        assert result.returncode == 0, result.stderr
