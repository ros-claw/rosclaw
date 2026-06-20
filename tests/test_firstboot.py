"""Tests for ROSClaw firstboot command."""

from __future__ import annotations

import json
import sys

import yaml


class TestFirstbootNonInteractive:
    def test_firstboot_offline_default(self, tmp_path, monkeypatch):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes", "--profile", "offline", "--no-telemetry"]
        code = main()
        assert code in (0, 2)

        cfg_path = home / "config" / "rosclaw.yaml"
        assert cfg_path.exists()
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg["cloud"]["enabled"] is False
        assert cfg["telemetry"]["enabled"] is False
        assert cfg["sandbox"]["enabled"] is True
        assert cfg["security"]["require_firewall_for_real_robot"] is True
        assert cfg["runtime"]["robot_id"] == "sim_ur5e"

    def test_firstboot_cloud_opt_in(self, tmp_path, monkeypatch):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes", "--profile", "cloud", "--telemetry"]
        code = main()
        assert code in (0, 2)

        cfg = yaml.safe_load((home / "config" / "rosclaw.yaml").read_text(encoding="utf-8"))
        assert cfg["cloud"]["enabled"] is True
        assert cfg["cloud"]["sync"]["configs"] is True
        assert cfg["telemetry"]["enabled"] is True
        assert cfg["telemetry"]["anonymous_install_ping"] is True

    def test_firstboot_idempotent_preserves_custom_robot(self, tmp_path, monkeypatch):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes", "--profile", "offline", "--no-telemetry"]
        main()

        cfg_path = home / "config" / "rosclaw.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg["runtime"]["robot_id"] = "custom_bot"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        sys.argv = ["rosclaw", "firstboot", "--yes", "--profile", "offline", "--no-telemetry"]
        main()

        cfg2 = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg2["runtime"]["robot_id"] == "custom_bot"
        backups = list((home / "backups").glob("rosclaw.yaml.*.bak"))
        assert len(backups) >= 1

    def test_firstboot_creates_mcp_json(self, tmp_path, monkeypatch):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes", "--enable-mcp"]
        main()

        mcp_path = home / "config" / "mcp.json"
        assert mcp_path.exists()
        mcp = json.loads(mcp_path.read_text(encoding="utf-8"))
        assert mcp["mcpServers"]["rosclaw"]["command"] == "rosclaw-mcp"
        assert "ROSCLAW_HOME" in mcp["mcpServers"]["rosclaw"]["env"]

    def test_firstboot_custom_workspace_path(self, tmp_path, monkeypatch):
        home = tmp_path / "custom-rc"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes"]
        main()

        assert (home / "config" / "rosclaw.yaml").exists()
        assert (home / "state" / "install.json").exists()

    def test_firstboot_state_install_json(self, tmp_path, monkeypatch):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes", "--profile", "offline"]
        main()

        state = json.loads((home / "state" / "install.json").read_text(encoding="utf-8"))
        assert state["firstboot_completed"] is True
        assert state["firstboot_profile"] == "offline"

    def test_firstboot_disables_mcp(self, tmp_path, monkeypatch):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes", "--disable-mcp"]
        main()

        assert not (home / "config" / "mcp.json").exists()


class TestFirstbootInteractive:
    def test_summary_helper_prints_expected_lines(self, tmp_path, capsys):
        from rosclaw.firstboot.wizard import _print_summary

        home = tmp_path / ".rosclaw"
        _print_summary(
            home=home,
            profile="cloud",
            robot="turtlebot",
            safety="strict",
            telemetry_enabled=True,
            enable_mcp=False,
            use_cases={"sandbox": True, "mcp": False},
        )
        captured = capsys.readouterr()
        for label in ("Workspace:", "Profile:", "Robot:", "Safety:", "Telemetry:", "MCP config:", "sandbox:"):
            assert label in captured.out

    def test_interactive_summary_and_cancellation(self, tmp_path, monkeypatch):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        import argparse

        from rosclaw.firstboot import wizard

        calls = []

        def fake_ask_yes_no(prompt: str, default: bool) -> bool:
            calls.append(prompt)
            if "Apply these settings" in prompt:
                return False
            return default

        monkeypatch.setattr(wizard, "ask_yes_no", fake_ask_yes_no)

        args = argparse.Namespace(
            workspace=str(home),
            profile="offline",
            robot=None,
            safety=None,
            force=False,
            json=False,
            dev=False,
        )
        code = wizard.run_firstboot_interactive(args)

        assert code == 0
        assert not (home / "config" / "rosclaw.yaml").exists()
        assert any("Apply these settings" in c for c in calls)


class TestFirstbootDoctorIntegration:
    def test_firstboot_then_doctor_bootstrap(self, tmp_path, monkeypatch):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes", "--profile", "offline", "--no-telemetry"]
        main()

        sys.argv = ["rosclaw", "doctor", "--bootstrap"]
        code = main()
        assert code in (0, 2)
