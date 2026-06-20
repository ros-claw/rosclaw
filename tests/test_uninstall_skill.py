"""Tests for ROSClaw uninstall and skill check commands."""

from __future__ import annotations

import json
import sys


class TestUninstall:
    def test_uninstall_keep_data_preserves_workspace(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        home.mkdir()
        (home / "config").mkdir()
        (home / "config" / "rosclaw.yaml").write_text("runtime:\n  robot_id: test\n", encoding="utf-8")
        (home / "state").mkdir()
        (home / "state" / "install.json").write_text('{"firstboot_completed": true}', encoding="utf-8")

        from rosclaw.cli import main

        sys.argv = ["rosclaw", "uninstall", "--keep-data"]
        code = main()

        captured = capsys.readouterr()
        assert code == 0
        assert home.exists()
        assert (home / "config" / "rosclaw.yaml").exists()
        assert "Workspace kept" in captured.out

    def test_uninstall_purge_removes_workspace(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        home.mkdir()
        (home / "config").mkdir()
        (home / "config" / "rosclaw.yaml").write_text("runtime:\n  robot_id: test\n", encoding="utf-8")

        from rosclaw.cli import main

        sys.argv = ["rosclaw", "uninstall", "--purge"]
        code = main()

        captured = capsys.readouterr()
        assert code == 0
        assert not home.exists()
        assert "Removed workspace" in captured.out

    def test_uninstall_no_flag_requires_choice(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))

        from rosclaw.cli import main

        sys.argv = ["rosclaw", "uninstall"]
        code = main()

        captured = capsys.readouterr()
        assert code == 1
        assert "--keep-data" in captured.out
        assert "--purge" in captured.out


class TestSkillCheck:
    def test_skill_check_builtin_found(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))

        from rosclaw.cli import main

        sys.argv = ["rosclaw", "skill", "check", "reach"]
        code = main()

        captured = capsys.readouterr()
        assert code == 0
        assert "Checking skill: reach" in captured.out
        assert "available" in captured.out.lower()

    def test_skill_check_missing_skill(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))

        from rosclaw.cli import main

        sys.argv = ["rosclaw", "skill", "check", "not_a_real_skill"]
        code = main()

        captured = capsys.readouterr()
        assert code == 1
        assert "not found" in captured.out.lower()
        assert "reach" in captured.out or "grasp" in captured.out or "navigate" in captured.out

    def test_skill_check_json_output(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))

        from rosclaw.cli import main

        sys.argv = ["rosclaw", "skill", "check", "grasp", "--json"]
        code = main()

        captured = capsys.readouterr()
        out = captured.out
        payload = out[out.index("{"): out.rindex("}") + 1]
        data = json.loads(payload)

        assert code == 0
        assert data["skill_id"] == "grasp"
        assert data["found"] is True
        assert data["status"] == "ok"
        assert data["skill_type"] == "manipulation"
