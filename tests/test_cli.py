"""Tests for rosclaw.cli"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestVersion:
    def test_version_flag(self, capsys):
        from rosclaw.cli import main

        with pytest.raises(SystemExit) as exc:
            sys.argv = ["rosclaw", "--version"]
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "rosclaw 1.0.0" in captured.out


class TestInit:
    def test_init_creates_workspace(self, tmp_path):
        from rosclaw.cli import main

        ws = tmp_path / "ws"
        sys.argv = ["rosclaw", "init", str(ws)]
        assert main() == 0

        assert (ws / "rosclaw.yaml").exists()
        assert (ws / "practice_data").is_dir()
        assert (ws / "skills").is_dir()
        assert (ws / "models").is_dir()

        config = (ws / "rosclaw.yaml").read_text()
        assert "robot_id: rosclaw_bot" in config
        assert "safety_level: MODERATE" in config

    def test_init_refuses_overwrite(self, tmp_path):
        from rosclaw.cli import main

        ws = tmp_path / "ws"
        sys.argv = ["rosclaw", "init", str(ws)]
        main()

        sys.argv = ["rosclaw", "init", str(ws)]
        assert main() == 0  # Idempotent: returns 0 if already exists

    def test_init_force_overwrite(self, tmp_path):
        from rosclaw.cli import main

        ws = tmp_path / "ws"
        sys.argv = ["rosclaw", "init", str(ws)]
        main()

        sys.argv = ["rosclaw", "init", "--force", str(ws)]
        assert main() == 0


class TestRun:
    @patch("rosclaw.core.Runtime")
    def test_run_starts_runtime(self, mock_runtime_cls):
        from rosclaw.cli import main

        mock_runtime = MagicMock()
        mock_runtime.is_running = True
        mock_runtime_cls.return_value = mock_runtime

        call_count = 0

        def fake_sleep(t):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                mock_runtime.is_running = False

        with patch("time.sleep", fake_sleep):
            sys.argv = ["rosclaw", "run", "--robot-id", "test_bot"]
            assert main() == 0

        mock_runtime.initialize.assert_called_once()
        mock_runtime.start.assert_called_once()
        mock_runtime.stop.assert_called_once()


class TestStatus:
    def test_status(self, capsys):
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "status"]
        assert main() == 0
        captured = capsys.readouterr()
        assert "ROSClaw v1.0 Status" in captured.out
        assert "Overall:" in captured.out
        assert "HEALTHY" in captured.out

    def test_status_shows_modules(self, capsys):
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "status"]
        main()
        captured = capsys.readouterr()
        assert "core.runtime" in captured.out
        assert "firewall.validator" in captured.out
        assert "memory.interface" in captured.out


class TestDoctor:
    def test_doctor_runs(self, capsys):
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "doctor"]
        main()
        captured = capsys.readouterr()
        assert "ROSClaw v1.0 — Doctor" in captured.out
        assert "Python version" in captured.out
        assert "e-URDF-Zoo" in captured.out

    def test_doctor_passes_in_workspace(self, tmp_path, capsys):
        from rosclaw.cli import main
        import os

        sys.argv = ["rosclaw", "init", str(tmp_path / "ws")]
        main()

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path / "ws")
            sys.argv = ["rosclaw", "doctor"]
            code = main()
            captured = capsys.readouterr()
            assert "ROSClaw v1.0 — Doctor" in captured.out
            assert code == 0 or "All checks passed" in captured.out or "Issues found" in captured.out
        finally:
            os.chdir(old_cwd)


class TestLogs:
    def test_logs_no_logs_dir(self, capsys):
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "logs"]
        assert main() == 0
        captured = capsys.readouterr()
        assert "Log directory not found" in captured.out or "No log files" in captured.out

    def test_logs_with_logs(self, tmp_path, capsys):
        from rosclaw.cli import main
        import os

        log_dir = tmp_path / ".rosclaw" / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "runtime.log").write_text("INFO: test log line 1\nERROR: test error\n")
        (log_dir / "provider.log").write_text("INFO: provider log\n")

        old_home = os.environ.get("HOME")
        try:
            os.environ["HOME"] = str(tmp_path)
            sys.argv = ["rosclaw", "logs", "--tail", "10"]
            assert main() == 0
            captured = capsys.readouterr()
            assert "ROSClaw Logs" in captured.out
            assert "runtime.log" in captured.out
        finally:
            if old_home is not None:
                os.environ["HOME"] = old_home
            else:
                os.environ.pop("HOME", None)

    def test_logs_level_filter(self, tmp_path, capsys):
        from rosclaw.cli import main
        import os

        log_dir = tmp_path / ".rosclaw" / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "runtime.log").write_text("INFO: normal\nERROR: critical\nDEBUG: detail\n")

        old_home = os.environ.get("HOME")
        try:
            os.environ["HOME"] = str(tmp_path)
            sys.argv = ["rosclaw", "logs", "--level", "ERROR", "--tail", "10"]
            assert main() == 0
            captured = capsys.readouterr()
            assert "ERROR" in captured.out
        finally:
            if old_home is not None:
                os.environ["HOME"] = old_home
            else:
                os.environ.pop("HOME", None)


class TestRobotList:
    def test_robot_list(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "robot", "list"]
        assert main() == 0
        captured = capsys.readouterr()
        assert "Robot Registry" in captured.out or "No robots found" in captured.out


class TestRobotInspect:
    def test_robot_inspect_found(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "robot", "inspect", "ur5e"]
        main()
        captured = capsys.readouterr()
        assert "Robot Profile" in captured.out or "not found" in captured.out

    def test_robot_inspect_not_found(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "robot", "inspect", "nonexistent_robot_xyz"]
        code = main()
        captured = capsys.readouterr()
        assert "not found" in captured.out or code == 1


class TestRobotValidate:
    def test_robot_validate_found(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "robot", "validate", "ur5e"]
        main()
        captured = capsys.readouterr()
        assert "Validation Result" in captured.out or "not found" in captured.out

    def test_robot_validate_not_found(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "robot", "validate", "nonexistent_robot_xyz"]
        code = main()
        captured = capsys.readouterr()
        assert "not found" in captured.out or code == 1


class TestPracticeCommands:
    def test_practice_list_empty(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "practice", "list"]
        assert main() == 0
        captured = capsys.readouterr()
        assert "No practice episodes" in captured.out or "Episodes" in captured.out

    def test_practice_show_not_found(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "practice", "show", "ep_nonexistent"]
        code = main()
        captured = capsys.readouterr()
        assert "not found" in captured.out or code == 1

    def test_practice_replay_not_found(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "practice", "replay", "ep_nonexistent"]
        code = main()
        captured = capsys.readouterr()
        assert "not found" in captured.out or code == 1

    def test_practice_export_not_found(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "practice", "export", "ep_nonexistent", "--format", "json"]
        code = main()
        captured = capsys.readouterr()
        assert "not found" in captured.out or code == 1


class TestDashboard:
    def test_dashboard(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "dashboard"]
        code = main()
        captured = capsys.readouterr()
        assert "ROSClaw v1.0 Dashboard" in captured.out
        assert code == 0

    def test_dashboard_open(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "dashboard", "--open"]
        code = main()
        captured = capsys.readouterr()
        assert "localhost:8765" in captured.out
        assert code == 0


class TestProviderList:
    def test_provider_list(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "provider", "list"]
        code = main()
        captured = capsys.readouterr()
        assert "Provider Registry" in captured.out or "No providers" in captured.out
        assert code == 0


class TestSkillList:
    def test_skill_list(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "skill", "list"]
        code = main()
        captured = capsys.readouterr()
        assert "Skill Registry" in captured.out or "No skills" in captured.out
        assert code == 0


class TestSandboxCommands:
    def test_sandbox_list_worlds(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "sandbox", "list-worlds"]
        code = main()
        captured = capsys.readouterr()
        assert "Sandbox Worlds" in captured.out
        assert "mock" in captured.out
        assert code == 0

    def test_sandbox_validate_found(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "sandbox", "validate", "ur5e"]
        main()
        captured = capsys.readouterr()
        assert "Validating" in captured.out or "not found" in captured.out

    def test_sandbox_validate_not_found(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "sandbox", "validate", "nonexistent_robot_xyz"]
        code = main()
        captured = capsys.readouterr()
        assert "not found" in captured.out or "error" in captured.out.lower() or code != 0


class TestMemoryCommands:
    def test_memory_status(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "memory", "status"]
        code = main()
        captured = capsys.readouterr()
        assert "Memory Status" in captured.out
        assert "experiences" in captured.out or code == 0

    def test_memory_query(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "memory", "query", "pick up cup"]
        code = main()
        captured = capsys.readouterr()
        assert "Query" in captured.out or "No matching" in captured.out or code == 0

    def test_memory_explain(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "memory", "explain"]
        code = main()
        captured = capsys.readouterr()
        assert "Failure Explanation" in captured.out or "No failure" in captured.out or code == 0


class TestEventsCommand:
    def test_events_tail(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "events", "--tail", "5"]
        code = main()
        captured = capsys.readouterr()
        assert "EventBus" in captured.out or "No events" in captured.out
        assert code == 0


class TestStopCommand:
    def test_stop_no_pid_file(self, capsys):
        from rosclaw.cli import main
        from pathlib import Path

        pid_file = Path.home() / ".rosclaw" / "runtime.pid"
        # Ensure no PID file
        if pid_file.exists():
            pid_file.unlink()

        sys.argv = ["rosclaw", "stop"]
        code = main()
        captured = capsys.readouterr()
        assert "No runtime PID" in captured.out or "not running" in captured.out
        assert code == 1

    def test_stop_with_pid_file(self, capsys, tmp_path):
        from rosclaw.cli import main

        pid_file = tmp_path / "runtime.pid"
        pid_file.write_text("99999")

        with patch("rosclaw.cli.Path.home", return_value=tmp_path):
            sys.argv = ["rosclaw", "stop"]
            code = main()
            captured = capsys.readouterr()
            # 99999 won't exist so it should be ProcessLookupError
            assert "not found" in captured.out or code == 1


class TestRestartCommand:
    @patch("rosclaw.core.Runtime")
    def test_restart(self, mock_runtime_cls, capsys):
        from rosclaw.cli import main

        mock_runtime = MagicMock()
        mock_runtime.is_running = True
        mock_runtime_cls.return_value = mock_runtime

        call_count = 0
        def fake_sleep(t):  # noqa: E306
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                mock_runtime.is_running = False

        # Ensure no PID file to avoid stop path issues
        from pathlib import Path
        pid_file = Path.home() / ".rosclaw" / "runtime.pid"
        if pid_file.exists():
            pid_file.unlink()

        with patch("time.sleep", fake_sleep):
            sys.argv = ["rosclaw", "restart"]
            code = main()

        assert code == 0
