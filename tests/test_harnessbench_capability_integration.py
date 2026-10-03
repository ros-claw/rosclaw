"""Offline native-interface grader tests, explicitly not live-agent trials."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarks.harnessbench.capability_integration import (
    GRAPH_CASES,
    INTEGRATION_TASKS,
    judge_integration,
)
from rosclaw.connectors.ros.cli import ros_cli


def _stage(root: Path, task_id: str):
    task = INTEGRATION_TASKS[task_id]
    for name, text in task.staged_files.items():
        (root / name).write_text(text)
    (root / "answer.json").write_text('{"scope":"FIXTURE_ONLY","reason":"local fixture test"}')


def _transcript(root, commands):
    sessions = root / "rh/agent/sessions"
    sessions.mkdir(parents=True)
    entries = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "name": "bash", "arguments": {"command": command}}
                ],
            }
        }
        for command in commands
    ]
    (sessions / "unit-test-not-real-agent.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries)
    )


@pytest.mark.parametrize("task_id", [f"CI{i:02d}" for i in range(1, 7)])
def test_real_offline_cli_outputs_checked_then_wrong_field_rejected(
    tmp_path, task_id, monkeypatch, capsys
):
    from types import SimpleNamespace

    _stage(tmp_path, task_id)

    def reject_network(**kwargs):
        raise AssertionError("unit test opened ROS transport")

    monkeypatch.setattr(ros_cli, "RosbridgeTransport", reject_network)
    args = SimpleNamespace(
        robot_id="matrix",
        graph=str(tmp_path / "graph.json"),
        endpoint="fixture://offline",
        output=str(tmp_path / "manifest.json"),
        json=True,
    )
    assert ros_cli.cmd_ros_compile(args) == 0
    (tmp_path / "compile.json").write_text(capsys.readouterr().out)
    args.manifest = str(tmp_path / "manifest.json")
    assert ros_cli.cmd_ros_list_capabilities(args) == 0
    (tmp_path / "list.json").write_text(capsys.readouterr().out)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    inspection_dir = tmp_path / "inspections"
    inspection_dir.mkdir()
    commands = [
        "rosclaw ros compile --graph graph.json",
        "rosclaw ros list-capabilities --manifest manifest.json",
    ]
    for cap in manifest["capabilities"]:
        args.capability_id = cap["id"]
        assert ros_cli.cmd_ros_inspect_capability(args) == 0
        (inspection_dir / f"{cap['id']}.json").write_text(capsys.readouterr().out)
        commands.append(f"rosclaw ros inspect-capability {cap['id']} --manifest manifest.json")
    if task_id in {"CI01", "CI05"}:
        args.args = '{"linear":{"x":999}}'
        ros_cli.cmd_ros_validate_capability(args)
        (tmp_path / "validate.json").write_text(capsys.readouterr().out)
        commands.append("rosclaw ros validate-capability cap --manifest manifest.json")
    _transcript(tmp_path, commands)
    good = judge_integration(task_id, tmp_path)
    assert good["verified_success"], good
    manifest["capabilities"][0]["risk"]["read_only"] = "false"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert not judge_integration(task_id, tmp_path)["verified_success"]


@pytest.mark.parametrize("task_id", ["CI07", "CI08"])
def test_negative_real_entrypoint(tmp_path, task_id, monkeypatch, capsys):
    from rosclaw.entrypoint import main

    _stage(tmp_path, task_id)
    monkeypatch.chdir(tmp_path)
    args = (
        ["ros", "compile", "--graph", "graph.json", "--json"]
        if task_id == "CI07"
        else ["ros", "list-capabilities", "--manifest", "missing.json", "--json"]
    )
    monkeypatch.setattr(sys, "argv", ["rosclaw", *args])
    monkeypatch.setattr(
        ros_cli, "RosbridgeTransport", lambda **kwargs: pytest.fail("network opened")
    )
    assert main() == 1
    (tmp_path / "failure.json").write_text(capsys.readouterr().out)
    (tmp_path / "exitcode.txt").write_text("1")
    _transcript(tmp_path, ["rosclaw " + " ".join(args)])
    assert judge_integration(task_id, tmp_path)["verified_success"]
    (tmp_path / "exitcode.txt").write_text("0")
    assert not judge_integration(task_id, tmp_path)["verified_success"]


@pytest.mark.parametrize("task_id", ["CI09", "CI10", "CI11", "CI12"])
def test_actual_practice_fixture_pipeline(tmp_path, task_id, monkeypatch, capsys):
    from rosclaw.entrypoint import main

    _stage(tmp_path, task_id)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / "temporary-home"))
    record = ["practice", "record", "--fixture", "fixture.json", "--out", "practice-data", "--json"]
    monkeypatch.setattr(sys, "argv", ["rosclaw", *record])
    rc = main()
    (tmp_path / "record.json").write_text(capsys.readouterr().out)
    (tmp_path / "record_exitcode.txt").write_text(str(rc))
    commands = ["rosclaw " + " ".join(record)]
    if task_id != "CI10":
        raw = tmp_path / "practice-data/sessions/practice_rh56_minimal_loop/raw/events.jsonl"
        if task_id == "CI11":
            events = [json.loads(line) for line in raw.read_text().splitlines()]
            for event in events:
                if event.get("event_type") == "physical_feedback_event":
                    del event["timestamp_ns"]
                    break
            raw.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        verify = [
            "practice",
            "verify",
            "practice_rh56_minimal_loop",
            "--data-root",
            "practice-data",
            "--strict",
            "--json",
        ]
        monkeypatch.setattr(sys, "argv", ["rosclaw", *verify])
        rc = main()
        (tmp_path / "verify.json").write_text(capsys.readouterr().out)
        (tmp_path / "verify_exitcode.txt").write_text(str(rc))
        commands.append("rosclaw " + " ".join(verify))
    _transcript(tmp_path, commands)
    result = judge_integration(task_id, tmp_path)
    assert result["verified_success"], result
    # A written claim without actual session calls is never integration evidence.
    (tmp_path / "rh/agent/sessions/unit-test-not-real-agent.jsonl").unlink()
    assert not judge_integration(task_id, tmp_path)["verified_success"]


def test_graph_cases_do_not_stage_private_projections():
    assert len(GRAPH_CASES) == 6
    assert len(INTEGRATION_TASKS) == 12
    assert all(
        set(t.staged_files) <= {"graph.json", "fixture.json", "input.json"}
        for t in INTEGRATION_TASKS.values()
    )
