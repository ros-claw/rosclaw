"""Tests for the ``realsense_capture_rgbd`` builtin skill handler.

These tests mock the RealSense MCP server so they can run without hardware.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rosclaw.mcp.onboarding.installed import InstalledRecord, InstalledRegistry
from rosclaw.skill.builtins.realsense_capture_rgbd import runner


@pytest.fixture
def fake_mcp_script(tmp_path: Path) -> Path:
    """Create a tiny MCP stdio server that pretends to be librealsense-mcp."""
    script = tmp_path / "fake_realsense_mcp.py"
    script.write_text(
        '\n'.join([
            "import json, sys, pathlib",
            "",
            "def send(msg):",
            "    print(json.dumps(msg), flush=True)",
            "",
            "for line in sys.stdin:",
            "    line = line.strip()",
            "    if not line or not line.startswith('{'): continue",
            "    req = json.loads(line)",
            "    req_id = req.get('id')",
            "    method = req.get('method', '')",
            "    if method == 'initialize':",
            "        send({'jsonrpc': '2.0', 'id': req_id, 'result': {'protocolVersion': '2024-11-05'}})",
            "    elif method == 'notifications/initialized':",
            "        continue",
            "    elif method == 'tools/list':",
            "        send({'jsonrpc': '2.0', 'id': req_id, 'result': {'tools': [{'name': 'capture_aligned_rgbd'}, {'name': 'list_devices'}]}})",
            "    elif method == 'tools/call':",
            "        args = req['params']['arguments']",
            "        name = req['params']['name']",
            "        if name == 'capture_aligned_rgbd':",
            "            pathlib.Path(args['color_path']).write_bytes(b'PNG')",
            "            pathlib.Path(args['depth_path']).write_bytes(b'PNG')",
            "            text = json.dumps({'color_path': args['color_path'], 'depth_path': args['depth_path'], 'usb_mode': 'USB3', 'serial': args.get('serial')})",
            "            send({'jsonrpc': '2.0', 'id': req_id, 'result': {'content': [{'type': 'text', 'text': text}]}})",
            "        elif name == 'list_devices':",
            "            text = json.dumps({'devices': [{'serial': '123456789'}]})",
            "            send({'jsonrpc': '2.0', 'id': req_id, 'result': {'content': [{'type': 'text', 'text': text}]}})",
        ]),
        encoding="utf-8",
    )
    return script


@pytest.fixture
def workspace_with_realsense_body(tmp_path: Path, fake_mcp_script: Path) -> Path:
    """Create a temporary workspace with a perception-only body and fake MCP."""
    workspace = tmp_path / "ws"
    body_dir = workspace / "body"
    body_dir.mkdir(parents=True)

    body_yaml = body_dir / "body.yaml"
    body_yaml.write_text(
        """
schema_version: rosclaw.body.v1
body_instance:
  id: d405_lab_01
  robot_model: realsense_d405
  serial_number: "123456789"
model_ref:
  profile_id: realsense_d405
metadata:
  perception_only: true
  no_actuation: true
safety:
  environment:
    perception_only: true
    no_actuation: true
""",
        encoding="utf-8",
    )

    registry = InstalledRegistry(home=workspace)
    record = InstalledRecord(
        server_name="librealsense-mcp",
        manifest_id="librealsense-mcp",
        name="librealsense-mcp",
        version="1.0.0",
        installed_at="2026-01-01T00:00:00Z",
        artifact_type="git",
        server_dir=str(workspace / "mcp" / "installed" / "librealsense-mcp"),
        runtime_config_path=str(workspace / "mcp" / "runtime" / "librealsense-mcp.yaml"),
    )
    registry.add(record)

    runtime_config = workspace / "mcp" / "runtime" / "librealsense-mcp.yaml"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(
        json.dumps(
            {
                "transport": {
                    "command": sys.executable,
                    "args": [str(fake_mcp_script)],
                    "env": {},
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return workspace


class TestRealsenseCaptureRgbdSkill:
    """Cover the RealSense capture skill runner with a mocked MCP."""

    def test_run_creates_color_and_depth_artifacts(self, workspace_with_realsense_body, tmp_path):
        workspace = workspace_with_realsense_body
        output_dir = tmp_path / "capture"

        result = runner.run(
            {
                "workspace": str(workspace),
                "body_id": "d405_lab_01",
                "output_dir": str(output_dir),
                "serial": "123456789",
            }
        )

        assert result["status"] == "success"
        assert result["skill"] == "realsense_capture_rgbd"
        assert result["server_name"] == "librealsense-mcp"
        assert result["tool"] == "capture_aligned_rgbd"
        assert output_dir.exists()
        assert (output_dir / "color.png").exists()
        assert (output_dir / "depth.png").exists()
        assert (output_dir / "capture_result.json").exists()
        assert result["metrics"]["usb_mode"] == "USB3"
        assert result["metrics"]["degraded"] is False

    def test_run_fails_when_body_is_not_perception_only(self, tmp_path):
        workspace = tmp_path / "ws"
        body_dir = workspace / "body"
        body_dir.mkdir(parents=True)
        (body_dir / "body.yaml").write_text(
            "schema_version: rosclaw.body.v1\nbody_instance:\n  id: manip_01\nmodel_ref:\n  profile_id: panda\n",
            encoding="utf-8",
        )

        result = runner.run(
            {
                "workspace": str(workspace),
                "body_id": "manip_01",
                "output_dir": str(tmp_path / "capture"),
            }
        )

        assert result["status"] == "blocked"
        assert "perception-only" in result["reason"].lower()

    def test_run_reports_error_when_no_realsense_mcp_installed(self, tmp_path):
        workspace = tmp_path / "ws"
        body_dir = workspace / "body"
        body_dir.mkdir(parents=True)
        (body_dir / "body.yaml").write_text(
            "schema_version: rosclaw.body.v1\nbody_instance:\n  id: d405_lab_01\nmetadata:\n  perception_only: true\n",
            encoding="utf-8",
        )

        result = runner.run(
            {
                "workspace": str(workspace),
                "body_id": "d405_lab_01",
                "output_dir": str(tmp_path / "capture"),
            }
        )

        assert result["status"] == "error"
        assert "mcp" in result["reason"].lower()

    def test_skill_executor_runs_builtin_handler(
        self, workspace_with_realsense_body, tmp_path, monkeypatch
    ):
        """End-to-end through SkillExecutor with the builtin registry."""
        from rosclaw.core.event_bus import EventBus
        from rosclaw.skill.builtins import load_builtins
        from rosclaw.skill_manager.executor import SkillExecutor
        from rosclaw.skill_manager.registry import SkillRegistry

        # Isolate the default BodyResolver home so no real body is linked.
        monkeypatch.setenv("HOME", str(tmp_path))

        workspace = workspace_with_realsense_body
        output_dir = tmp_path / "capture_executor"

        registry = SkillRegistry()
        load_builtins(registry)
        executor = SkillExecutor(event_bus=EventBus(), registry=registry, body_resolver=None)

        result = executor.execute(
            "realsense_capture_rgbd",
            parameters={
                "workspace": str(workspace),
                "body_id": "d405_lab_01",
                "output_dir": str(output_dir),
                "serial": "123456789",
            },
        )

        handler_result = result.get("handler_result", {})
        assert handler_result.get("status") == "success", handler_result
        assert (output_dir / "color.png").exists()
        assert (output_dir / "depth.png").exists()
