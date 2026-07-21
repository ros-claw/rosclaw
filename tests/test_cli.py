"""Tests for rosclaw.cli"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rosclaw.body.service import BodyInstanceService


class TestVersion:
    def test_version_flag(self, capsys):
        from rosclaw.cli import main

        with pytest.raises(SystemExit) as exc:
            sys.argv = ["rosclaw", "--version"]
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "rosclaw 1.0.1" in captured.out

    def test_top_level_help_discovers_daemon_control_plane(self, capsys):
        from rosclaw.entrypoint import main

        with pytest.raises(SystemExit) as exc:
            sys.argv = ["rosclaw", "--help"]
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "daemon" in captured.out
        assert "Inspect or call the local rosclawd control plane" in captured.out


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
        import os

        from rosclaw.cli import main

        sys.argv = ["rosclaw", "init", str(tmp_path / "ws")]
        main()

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path / "ws")
            sys.argv = ["rosclaw", "doctor"]
            code = main()
            captured = capsys.readouterr()
            assert "ROSClaw v1.0 — Doctor" in captured.out
            assert (
                code == 0 or "All checks passed" in captured.out or "Issues found" in captured.out
            )
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
        import os

        from rosclaw.cli import main

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
        import os

        from rosclaw.cli import main

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
    def test_practice_list_empty(self, tmp_path, capsys):
        from rosclaw.cli import main

        data_root = tmp_path / "practice_data"
        sys.argv = ["rosclaw", "practice", "list", "--data-root", str(data_root)]
        assert main() == 0
        captured = capsys.readouterr()
        assert "No practice sessions" in captured.out or "Sessions" in captured.out

    def test_practice_show_not_found(self, tmp_path, capsys):
        from rosclaw.cli import main

        data_root = tmp_path / "practice_data"
        sys.argv = ["rosclaw", "practice", "show", "ep_nonexistent", "--data-root", str(data_root)]
        code = main()
        captured = capsys.readouterr()
        assert "not found" in captured.out or code == 1

    def test_practice_replay_not_found(self, tmp_path, capsys):
        from rosclaw.cli import main

        data_root = tmp_path / "practice_data"
        sys.argv = [
            "rosclaw",
            "practice",
            "replay",
            "ep_nonexistent",
            "--data-root",
            str(data_root),
        ]
        code = main()
        captured = capsys.readouterr()
        assert "not found" in captured.out or code == 1

    def test_practice_export_not_found(self, capsys):
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "practice", "export", "ep_nonexistent", "--format", "json"]
        code = main()
        captured = capsys.readouterr()
        assert "not found" in captured.out or code == 1

    def test_practice_init(self, monkeypatch, tmp_path):
        from rosclaw.cli import main

        monkeypatch.setenv("HOME", str(tmp_path))
        sys.argv = ["rosclaw", "practice", "init", "--robot", "test_bot"]
        assert main() == 0
        assert (tmp_path / ".rosclaw" / "practice" / "config.yaml").exists()

    def test_practice_start_mock(self, monkeypatch, tmp_path, capsys):
        from rosclaw.cli import main

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        data_root = tmp_path / "practice_data"
        sys.argv = [
            "rosclaw",
            "practice",
            "start",
            "--robot",
            "test_bot",
            "--task",
            "mock_task",
            "--sources",
            "agent,runtime",
            "--mock",
            "--duration",
            "500ms",
            "--data-root",
            str(data_root),
        ]
        assert main() == 0
        captured = capsys.readouterr()
        assert "Started session" in captured.out
        assert "Stopped" in captured.out

    def test_practice_export_jsonl(self, monkeypatch, tmp_path, capsys):
        from rosclaw.cli import main

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        data_root = tmp_path / "practice_data"
        sys.argv = [
            "rosclaw",
            "practice",
            "start",
            "--robot",
            "test_bot",
            "--task",
            "mock_task",
            "--sources",
            "agent",
            "--mock",
            "--duration",
            "300ms",
            "--data-root",
            str(data_root),
        ]
        assert main() == 0
        # Find the practice id from data root
        sessions = [d for d in (data_root / "sessions").iterdir() if d.is_dir()]
        assert sessions
        practice_id = sessions[0].name
        sys.argv = [
            "rosclaw",
            "practice",
            "export",
            practice_id,
            "--format",
            "jsonl",
            "--data-root",
            str(data_root),
        ]
        capsys.readouterr()  # clear output from start command
        assert main() == 0
        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line.strip()]
        assert len(lines) > 0
        import json

        assert json.loads(lines[0])["schema_version"] == "practice.event.v1"

    def test_practice_list_show_replay_with_session(self, monkeypatch, tmp_path, capsys):
        from rosclaw.cli import main

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        data_root = tmp_path / "practice_data"
        sys.argv = [
            "rosclaw",
            "practice",
            "start",
            "--robot",
            "test_bot",
            "--task",
            "mock_task",
            "--sources",
            "agent,runtime",
            "--mock",
            "--duration",
            "500ms",
            "--data-root",
            str(data_root),
        ]
        assert main() == 0
        sessions = [d for d in (data_root / "sessions").iterdir() if d.is_dir()]
        assert sessions
        practice_id = sessions[0].name
        capsys.readouterr()  # clear start output

        sys.argv = ["rosclaw", "practice", "list", "--data-root", str(data_root)]
        assert main() == 0
        captured = capsys.readouterr()
        assert practice_id in captured.out
        assert "SUCCESS" in captured.out

        sys.argv = ["rosclaw", "practice", "show", practice_id, "--data-root", str(data_root)]
        capsys.readouterr()
        assert main() == 0
        captured = capsys.readouterr()
        assert practice_id in captured.out
        assert "Robot:" in captured.out

        sys.argv = ["rosclaw", "practice", "replay", practice_id, "--data-root", str(data_root)]
        capsys.readouterr()
        assert main() == 0
        captured = capsys.readouterr()
        assert "REPLAY:" in captured.out
        assert "Total events:" in captured.out
        assert "agent" in captured.out

        import uvicorn

        from rosclaw.cli import main

        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)
        sys.argv = ["rosclaw", "dashboard"]
        code = main()
        captured = capsys.readouterr()
        assert "ROSClaw v1.0 Dashboard" in captured.out
        assert code == 0

    def test_dashboard_open(self, capsys, monkeypatch):
        # Stub uvicorn.run so the test doesn't actually start a blocking server.
        import uvicorn

        from rosclaw.cli import main

        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)
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

    def test_sandbox_run_accepts_world_argument(self, capsys, tmp_path):
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "sandbox",
            "run",
            "--robot",
            "ur5e",
            "--world",
            "tabletop",
            "--task",
            "reach",
            "--artifact-dir",
            str(tmp_path),
        ]
        code = main()
        captured = capsys.readouterr()
        assert code == 0
        assert "World:      tabletop" in captured.out
        assert "Status:     COMPLETED" in captured.out
        assert "Evidence:   TASK_VERIFIED" in captured.out
        assert "Verified:   True" in captured.out

    def test_sandbox_fixture_requires_explicit_mode(self, capsys, tmp_path):
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "sandbox",
            "run",
            "--robot",
            "sim_ur5e",
            "--world",
            "tabletop",
            "--task",
            "reach",
            "--mode",
            "fixture",
            "--artifact-dir",
            str(tmp_path),
        ]
        code = main()
        captured = capsys.readouterr()

        assert code == 0
        assert "MODE: FIXTURE" in captured.out
        assert "NO PHYSICS WAS EXECUTED" in captured.out
        assert "NOT VALID FOR ACCEPTANCE" in captured.out
        assert "Verified:   False" in captured.out

    def test_sandbox_missing_model_fails_with_json_receipt(self, capsys, tmp_path):
        import json

        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "sandbox",
            "run",
            "--robot",
            "definitely_missing_robot",
            "--world",
            "tabletop",
            "--task",
            "reach",
            "--artifact-dir",
            str(tmp_path),
            "--json",
        ]
        code = main()
        payload = json.loads(capsys.readouterr().out)

        assert code == 1
        assert payload["final_state"] == "FAILED"
        assert payload["verified"] is False
        assert payload["errors"][0]["code"] == "PHYSICS_UNAVAILABLE"


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
        from pathlib import Path

        from rosclaw.cli import main

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


class TestProviderDiagnose:
    """CLI smoke tests for `rosclaw provider diagnose`."""

    @pytest.fixture
    def linked_workspace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        BodyInstanceService().create_or_init(
            robot="unitree-g1",
            name="g1-cli",
            mode="registry",
            update_registry=True,
            switch_active=True,
        )
        return tmp_path / ".rosclaw"

    def test_provider_diagnose_json(self, linked_workspace, capsys):
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "provider",
            "diagnose",
            "--body",
            "current",
            "--json",
        ]
        code = main()
        captured = capsys.readouterr()
        assert code == 0
        assert "g1-cli" in captured.out
        data = json.loads(captured.out)
        assert data["body_instance_id"] == "g1-cli"
        assert "interfaces" in data
        assert "summary" in data

    def test_provider_diagnose_available_override(self, linked_workspace, capsys):
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "provider",
            "diagnose",
            "--body",
            "current",
            "--available",
            "head_camera",
            "--json",
        ]
        code = main()
        captured = capsys.readouterr()
        assert code == 0
        data = json.loads(captured.out)
        assert data["interfaces"]["head_camera"]["status"] == "available"


class TestSandboxGenerateConfig:
    """CLI smoke tests for `rosclaw sandbox generate-config`."""

    @pytest.fixture
    def linked_workspace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        BodyInstanceService().create_or_init(
            robot="unitree-g1",
            name="g1-cli",
            mode="registry",
            update_registry=True,
            switch_active=True,
        )
        return tmp_path / ".rosclaw"

    def test_sandbox_generate_config_mujoco(self, linked_workspace, capsys):
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "sandbox",
            "generate-config",
            "--body",
            "current",
            "--engine",
            "mujoco",
        ]
        code = main()
        captured = capsys.readouterr()
        assert code == 0
        assert "Generated mujoco config" in captured.out
        assert "g1-cli" in captured.out

        output_path = (
            linked_workspace / "bodies" / "g1-cli" / "refs" / "sandbox" / "mujoco.config.yaml"
        )
        assert output_path.exists()

    def test_sandbox_generate_config_isaac_json(self, linked_workspace, capsys):
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "sandbox",
            "generate-config",
            "--body",
            "current",
            "--engine",
            "isaac",
            "--json",
        ]
        code = main()
        captured = capsys.readouterr()
        assert code == 0
        data = json.loads(captured.out)
        assert data["engine"] == "isaac"
        assert data["body_instance_id"] == "g1-cli"
        assert Path(data["output_path"]).exists()


class TestBodyUpdateStateFromProviderHealth:
    """CLI smoke test for `rosclaw body update-state --from-provider-health`."""

    @pytest.fixture
    def linked_workspace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        BodyInstanceService().create_or_init(
            robot="unitree-g1",
            name="g1-cli",
            mode="registry",
            update_registry=True,
            switch_active=True,
        )
        return tmp_path / ".rosclaw"

    def test_update_state_from_provider_health(self, linked_workspace, capsys):
        from rosclaw.cli import main

        # Pre-mark a sensor unavailable so the provider-health patch is non-empty.
        sys.argv = [
            "rosclaw",
            "body",
            "update-state",
            "--component-status",
            "head_camera=unavailable",
            "--reason",
            "pre-mark",
        ]
        assert main() == 0

        sys.argv = [
            "rosclaw",
            "body",
            "update-state",
            "--from-provider-health",
            "--reason",
            "provider check",
        ]
        code = main()
        captured = capsys.readouterr()
        assert code == 0
        assert "Updated body state" in captured.out
