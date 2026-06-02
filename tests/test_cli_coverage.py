"""Coverage tests for rosclaw.cli edge cases and uncovered branches."""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ------------------------------------------------------------------
# main() help paths
# ------------------------------------------------------------------

class TestMainHelpPaths:
    def test_main_no_command_prints_help(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw"]
        code = main()
        captured = capsys.readouterr()
        assert "usage:" in captured.out or "rosclaw" in captured.out
        assert code == 1

    def test_robot_subcommand_no_action(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "robot"]
        code = main()
        captured = capsys.readouterr()
        assert "usage:" in captured.out or code == 1

    def test_provider_subcommand_no_action(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "provider"]
        code = main()
        capsys.readouterr()
        assert code == 1

    def test_skill_subcommand_no_action(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "skill"]
        code = main()
        capsys.readouterr()
        assert code == 1

    def test_sandbox_subcommand_no_action(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "sandbox"]
        code = main()
        capsys.readouterr()
        assert code == 1

    def test_memory_subcommand_no_action(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "memory"]
        code = main()
        capsys.readouterr()
        assert code == 1

    def test_practice_subcommand_no_action(self, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "practice"]
        code = main()
        capsys.readouterr()
        assert code == 1


# ------------------------------------------------------------------
# cmd_doctor edge cases
# ------------------------------------------------------------------

class TestDoctorEdgeCases:
    @patch("platform.python_version", return_value="2.7.0")
    def test_doctor_python_version_too_old(self, mock_pyver, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "doctor"]
        code = main()
        captured = capsys.readouterr()
        assert "Doctor" in captured.out
        assert code == 1

    @patch("importlib.import_module", side_effect=ImportError("no module"))
    def test_doctor_core_module_import_fails(self, mock_import, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "doctor"]
        code = main()
        captured = capsys.readouterr()
        assert "Doctor" in captured.out
        assert code == 1

    def test_doctor_no_zoo_no_config(self, capsys, tmp_path):
        from rosclaw.cli import main
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            sys.argv = ["rosclaw", "doctor"]
            code = main()
            captured = capsys.readouterr()
            assert "Doctor" in captured.out
            assert code == 1
        finally:
            os.chdir(old_cwd)


# ------------------------------------------------------------------
# cmd_logs edge cases
# ------------------------------------------------------------------

class TestLogsEdgeCases:
    def test_logs_unreadable_file(self, tmp_path, capsys):
        from rosclaw.cli import main
        import os

        log_dir = tmp_path / ".rosclaw" / "logs"
        log_dir.mkdir(parents=True)
        bad_file = log_dir / "broken.log"
        bad_file.write_text("some content")
        # Make file unreadable
        old_mode = bad_file.stat().st_mode
        bad_file.chmod(0o000)

        old_home = os.environ.get("HOME")
        try:
            os.environ["HOME"] = str(tmp_path)
            sys.argv = ["rosclaw", "logs", "--tail", "10"]
            code = main()
            captured = capsys.readouterr()
            assert "Logs" in captured.out or code == 0
        finally:
            bad_file.chmod(old_mode)
            if old_home is not None:
                os.environ["HOME"] = old_home
            else:
                os.environ.pop("HOME", None)

    def test_logs_module_filter(self, tmp_path, capsys):
        from rosclaw.cli import main
        import os

        log_dir = tmp_path / ".rosclaw" / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "runtime.log").write_text("INFO: test\n")
        (log_dir / "provider.log").write_text("INFO: provider\n")

        old_home = os.environ.get("HOME")
        try:
            os.environ["HOME"] = str(tmp_path)
            sys.argv = ["rosclaw", "logs", "--module", "runtime", "--tail", "10"]
            main()
            captured = capsys.readouterr()
            assert "runtime.log" in captured.out
            assert "provider.log" not in captured.out
        finally:
            if old_home is not None:
                os.environ["HOME"] = old_home
            else:
                os.environ.pop("HOME", None)


# ------------------------------------------------------------------
# cmd_status degraded modules
# ------------------------------------------------------------------

class TestStatusEdgeCases:
    @patch("importlib.import_module", side_effect=ImportError("no module"))
    def test_status_all_degraded(self, mock_import, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "status"]
        code = main()
        captured = capsys.readouterr()
        assert "DEGRADED" in captured.out
        assert code == 1


# ------------------------------------------------------------------
# cmd_stop edge cases
# ------------------------------------------------------------------

class TestStopEdgeCases:
    def test_stop_process_lookup_error(self, capsys, tmp_path):
        from rosclaw.cli import main
        import os

        pid_file = tmp_path / "runtime.pid"
        pid_file.write_text("99999")

        old_home = os.environ.get("HOME")
        try:
            os.environ["HOME"] = str(tmp_path)
            sys.argv = ["rosclaw", "stop"]
            code = main()
            captured = capsys.readouterr()
            assert "not found" in captured.out or "PID" in captured.out
            assert code == 1
        finally:
            if old_home is not None:
                os.environ["HOME"] = old_home
            else:
                os.environ.pop("HOME", None)


# ------------------------------------------------------------------
# cmd_events with history
# ------------------------------------------------------------------

class TestEventsEdgeCases:
    def test_events_with_history(self, capsys):
        from rosclaw.cli import main
        from rosclaw.core.event_bus import EventBus, Event

        bus = EventBus()
        bus.publish(Event(topic="test.event", payload={"x": 1}, source="test"))

        sys.argv = ["rosclaw", "events", "--tail", "5"]
        code = main()
        captured = capsys.readouterr()
        assert "test.event" in captured.out or "EventBus" in captured.out
        assert code == 0


# ------------------------------------------------------------------
# cmd_robot_install error path
# ------------------------------------------------------------------

class TestRobotInstallEdgeCases:
    @patch("rosclaw.runtime.RobotRegistry")
    def test_robot_install_not_found(self, mock_reg_cls, capsys):
        from rosclaw.cli import main
        mock_reg = MagicMock()
        mock_reg.install.side_effect = FileNotFoundError("robot not found")
        mock_reg.list_available.return_value = ["ur5e", "panda"]
        mock_reg_cls.return_value = mock_reg

        sys.argv = ["rosclaw", "robot", "install", "nonexistent"]
        code = main()
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or "not found" in captured.out.lower()
        assert code == 1


# ------------------------------------------------------------------
# cmd_practice export unknown format
# ------------------------------------------------------------------

class TestPracticeExportEdgeCases:
    @patch("rosclaw.practice.episode_recorder.EpisodeRecorder")
    def test_practice_export_unknown_format(self, mock_rec_cls, capsys):
        from rosclaw.cli import main
        mock_rec = MagicMock()
        mock_rec.get_episode.return_value = {"episode_id": "ep_1"}
        mock_rec_cls.return_value = mock_rec

        sys.argv = ["rosclaw", "practice", "export", "ep_1", "--format", "xml"]
        # Need to patch argparse choices — "xml" is not a valid choice
        # Instead test that the command is rejected
        with pytest.raises(SystemExit):
            main()


# ------------------------------------------------------------------
# cmd_memory_query with results
# ------------------------------------------------------------------

class TestMemoryQueryEdgeCases:
    @patch("rosclaw.memory.interface.MemoryInterface")
    def test_memory_query_with_results(self, mock_mem_cls, capsys):
        from rosclaw.cli import main
        mock_mem = MagicMock()
        mock_mem.find_similar_experiences.return_value = [
            {
                "id": "exp_1",
                "event_type": "success",
                "instruction": "pick the red cup",
                "outcome": "success",
                "tags": ["grasp", "cup"],
            }
        ]
        mock_mem_cls.return_value = mock_mem

        sys.argv = ["rosclaw", "memory", "query", "cup", "--limit", "3"]
        code = main()
        captured = capsys.readouterr()
        assert "pick the red cup" in captured.out or "Query" in captured.out
        assert code == 0

    @patch("rosclaw.memory.interface.MemoryInterface")
    def test_memory_explain_with_failure(self, mock_mem_cls, capsys):
        from rosclaw.cli import main
        mock_mem = MagicMock()
        mock_mem.explain_last_failure.return_value = {
            "id": "f1",
            "failure_type": "grasp_miss",
            "root_cause": "insufficient_force",
            "recovery_hint": "increase gripper force",
            "sandbox_intervened": True,
            "timestamp": "2024-01-01",
        }
        mock_mem_cls.return_value = mock_mem

        sys.argv = ["rosclaw", "memory", "explain", "--task-id", "task_1"]
        code = main()
        captured = capsys.readouterr()
        assert "grasp_miss" in captured.out or "Failure" in captured.out
        assert code == 0


# ------------------------------------------------------------------
# cmd_sandbox_validate exception
# ------------------------------------------------------------------

class TestSandboxValidateEdgeCases:
    @patch("rosclaw.runtime.RobotRegistry")
    def test_sandbox_validate_exception(self, mock_reg_cls, capsys):
        from rosclaw.cli import main
        mock_reg = MagicMock()
        mock_reg.validate.side_effect = RuntimeError("simulation crash")
        mock_reg_cls.return_value = mock_reg

        sys.argv = ["rosclaw", "sandbox", "validate", "ur5e"]
        code = main()
        captured = capsys.readouterr()
        assert "error" in captured.out.lower() or "crash" in captured.out.lower()
        assert code == 1


# ------------------------------------------------------------------
# cmd_dashboard with degraded modules
# ------------------------------------------------------------------

class TestDashboardEdgeCases:
    @patch("importlib.import_module", side_effect=ImportError("no module"))
    def test_dashboard_all_degraded(self, mock_import, capsys):
        from rosclaw.cli import main
        sys.argv = ["rosclaw", "dashboard"]
        code = main()
        captured = capsys.readouterr()
        assert "Dashboard" in captured.out
        assert code == 0
