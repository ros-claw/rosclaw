"""CLI-level tests for ``rosclaw mcp install/list/health``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


class TestMcpInstall:
    def test_cli_install_dry_run_text(
        self,
        fake_home: Path,
        project_root: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "mcp",
            "install",
            "unitree-g1",
            "--dry-run",
            "--offline",
            "--project-root",
            str(project_root),
        ]
        assert main() == 0
        captured = capsys.readouterr()
        assert "Hardware MCP Install Plan" in captured.out
        assert "io.rosclaw.hardware.unitree-g1" in captured.out
        assert "unitree-g1" in captured.out

    def test_cli_install_dry_run_json(
        self,
        fake_home: Path,
        project_root: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "mcp",
            "install",
            "unitree-g1",
            "--dry-run",
            "--offline",
            "--json",
            "--project-root",
            str(project_root),
        ]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["solved"]["manifest_id"] == "io.rosclaw.hardware.unitree-g1"
        assert data["solved"]["version"] == "1.0.0"
        assert "mcp_bindings.unitree_g1" in data["body_patch"]


class TestMcpList:
    def test_cli_list_offline_text(
        self,
        fake_home: Path,
        project_root: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "mcp",
            "list",
            "--offline",
            "--project-root",
            str(project_root),
        ]
        assert main() == 0
        captured = capsys.readouterr()
        assert "Hardware MCP Servers" in captured.out
        assert "unitree-g1" in captured.out
        assert "realsense-d455" in captured.out

    def test_cli_list_offline_json(
        self,
        fake_home: Path,
        project_root: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "mcp",
            "list",
            "--offline",
            "--json",
            "--project-root",
            str(project_root),
        ]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        available = {entry["manifest_id"] for entry in data["available"]}
        assert "io.rosclaw.hardware.unitree-g1" in available
        assert "io.rosclaw.hardware.realsense-d455" in available


class TestMcpHealth:
    @pytest.fixture
    def _patch_protocol_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import rosclaw.mcp.onboarding.health as health_module

        monkeypatch.setattr(
            health_module,
            "_command_resolvable",
            lambda cmd, env=None: (True, "/fake/python"),
        )

    @pytest.fixture
    def _managed_mcp_json(self, project_root: Path) -> None:
        mcp_json = project_root / ".mcp.json"
        mcp_json.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "rosclaw-unitree-g1": {
                            "type": "stdio",
                            "command": "rosclaw-mcp-run",
                            "args": ["unitree-g1"],
                            "x-rosclaw.managed": {},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_cli_health_unitree_ok(
        self,
        fake_home: Path,
        project_root: Path,
        installed_unitree: None,
        body_yaml_unitree: Path,
        monkeypatch_registry: Any,
        monkeypatch: pytest.MonkeyPatch,
        _patch_protocol_resolution: None,
        _managed_mcp_json: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from rosclaw.cli import main

        monkeypatch_registry.install("unitree-g1")
        monkeypatch.chdir(project_root)

        sys.argv = [
            "rosclaw",
            "mcp",
            "health",
            "unitree-g1",
            "--project-root",
            str(project_root),
        ]
        assert main() == 0
        captured = capsys.readouterr()
        assert "unitree-g1" in captured.out
        assert "PASS" in captured.out
        assert "FAIL" not in captured.out
