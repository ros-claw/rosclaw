"""Explicit file arguments never authorize fallback ROS network discovery."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.cli import ros_cli


@pytest.mark.parametrize(
    "content", [None, "{", "[]", "{}", '{"topics":"wrong","ros_version":"ros2"}']
)
def test_compile_offline_file_invalid_never_constructs_transport(
    tmp_path, monkeypatch, capsys, content
):
    path = tmp_path / "graph.json"
    if content is not None:
        path.write_text(content)
    opened = []
    monkeypatch.setattr(ros_cli, "RosbridgeTransport", lambda **kwargs: opened.append(kwargs))
    args = SimpleNamespace(
        robot_id="probe", graph=str(path), endpoint="ws://127.0.0.1:9090", output=None, json=True
    )
    rc = ros_cli.cmd_ros_compile(args)
    result = json.loads(capsys.readouterr().out)
    assert not opened
    assert rc == 1 and result["ok"] is False
    assert "error" in result


@pytest.mark.parametrize(
    "content", [None, "{", "[]", "{}", '{"robot_id":"probe","capabilities":"wrong"}']
)
@pytest.mark.parametrize(
    "command",
    [
        ros_cli.cmd_ros_list_capabilities,
        ros_cli.cmd_ros_inspect_capability,
        ros_cli.cmd_ros_validate_capability,
    ],
)
def test_manifest_invalid_never_constructs_transport(
    tmp_path, monkeypatch, capsys, content, command
):
    path = tmp_path / "manifest.json"
    if content is not None:
        path.write_text(content)
    opened = []
    monkeypatch.setattr(ros_cli, "RosbridgeTransport", lambda **kwargs: opened.append(kwargs))
    args = SimpleNamespace(
        robot_id="probe",
        manifest=str(path),
        endpoint="ws://127.0.0.1:9090",
        json=True,
        capability_id="probe.command.cmd_vel",
        args="{}",
    )
    rc = command(args)
    result = json.loads(capsys.readouterr().out)
    assert not opened
    assert rc == 1 and result["ok"] is False
    assert "Offline manifest" in result["error"]


def test_no_file_retains_explicit_live_discovery(monkeypatch):
    opened = []

    def fail_transport(**kwargs):
        opened.append(kwargs)
        raise RuntimeError("guarded mock network")

    monkeypatch.setattr(ros_cli, "RosbridgeTransport", fail_transport)
    assert (
        ros_cli._get_provider_manifest(
            SimpleNamespace(manifest=None, endpoint="ws://127.0.0.1:9090", robot_id="probe")
        )
        is None
    )
    assert len(opened) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["ros", "compile", "--graph", "/missing/graph.json", "--json"],
        ["ros", "list-capabilities", "--manifest", "/missing/manifest.json", "--json"],
        [
            "ros",
            "inspect-capability",
            "probe.observation",
            "--manifest",
            "/missing/manifest.json",
            "--json",
        ],
        [
            "ros",
            "validate-capability",
            "probe.observation",
            "--manifest",
            "/missing/manifest.json",
            "--json",
        ],
    ],
)
def test_actual_entrypoint_parsing_offline_missing(argv, monkeypatch, capsys):
    import sys

    from rosclaw.entrypoint import main

    opened = []
    monkeypatch.setattr(ros_cli, "RosbridgeTransport", lambda **kwargs: opened.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["rosclaw", *argv])
    assert main() == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False and result["error"]
    assert not opened
