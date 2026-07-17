"""L9: CLI tests."""


class TestCLI:
    """AUTO-CLI-001/002/003: CLI commands availability."""

    def test_cli_help(self, run_auto_cli):
        """AUTO-CLI-000: CLI --help works."""
        result = run_auto_cli("--help")
        assert result.returncode == 0
        assert "rosclaw auto" in result.stdout

    def test_cli_init(self, run_auto_cli):
        """AUTO-CLI-001: rosclaw auto init creates task."""
        result = run_auto_cli(
            "init",
            "--task",
            "pick_cube",
            "--robot",
            "panda",
            "--skill",
            "pick_v1",
        )
        assert result.returncode == 0
        assert "pick_cube" in result.stdout

    def test_cli_run_dry(self, run_auto_cli):
        """AUTO-CLI-002: rosclaw auto run --dry-run works."""
        init_result = run_auto_cli(
            "init",
            "--task",
            "pick_cube",
            "--robot",
            "panda",
            "--skill",
            "pick_v1",
        )
        assert init_result.returncode == 0

        result = run_auto_cli(
            "run",
            "--task",
            "pick_cube",
            "--rounds",
            "2",
            "--dry-run",
        )
        assert result.returncode == 0
        assert "Evolution complete" in result.stdout

    def test_cli_report(self, run_auto_cli):
        """AUTO-CLI-003: rosclaw auto report works."""
        init_result = run_auto_cli(
            "init",
            "--task",
            "pick_cube",
            "--robot",
            "panda",
            "--skill",
            "pick_v1",
        )
        assert init_result.returncode == 0

        result = run_auto_cli("report", "--task", "pick_cube")
        assert result.returncode == 0
        assert "Evolution Report" in result.stdout

    def test_cli_champion(self, run_auto_cli):
        result = run_auto_cli("champion", "--task", "pick_cube")
        assert result.returncode == 0
        assert "No champion found" in result.stdout

    def test_cli_deadends(self, run_auto_cli):
        result = run_auto_cli("deadends", "--task", "pick_cube")
        assert result.returncode == 0
        assert "Dead ends: 0" in result.stdout
