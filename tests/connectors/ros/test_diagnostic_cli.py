"""Actual CLI dispatch for bounded offline diagnostic artifacts."""

import argparse
import json
from datetime import timedelta

from rosclaw.connectors.ros.cli.ros_cli import add_ros_subparser, dispatch_ros_command
from tests.connectors.ros.test_diagnostic_experiments import NOW, capture


def parser():
    root = argparse.ArgumentParser()
    add_ros_subparser(root.add_subparsers(dest="command"))
    return root


def test_propose_then_readonly_falsification_cli_never_discovers_ros(tmp_path, monkeypatch, capsys):
    from rosclaw.connectors.ros import expert

    monkeypatch.setattr(
        expert,
        "inspect_system",
        lambda **_: (_ for _ in ()).throw(AssertionError("offline must not discover ROS")),
    )
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    plan = tmp_path / "experiment.json"
    before.write_text(json.dumps(capture().model_dump(mode="json")))
    later = NOW + timedelta(seconds=1)
    after.write_text(json.dumps(capture(stamp=later, qos=False).model_dump(mode="json")))
    args = parser().parse_args(
        [
            "ros",
            "diagnostic-plan",
            "--operational-graph",
            str(before),
            "--issue-id",
            "F-SCAN-001",
            "--output",
            str(plan),
            "--json",
        ]
    )
    assert dispatch_ros_command(args) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["experiment"]["execution_status"] == "PROPOSED_NOT_EXECUTED"
    assert json.loads(plan.read_text()) == value
    args = parser().parse_args(
        [
            "ros",
            "diagnostic-assess",
            "--operational-graph",
            str(before),
            "--experiment",
            str(plan),
            "--followup-graph",
            str(after),
            "--reviewed-at",
            later.isoformat(),
            "--json",
        ]
    )
    assert dispatch_ros_command(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["assessment"]["withdrawn_hypothesis_ids"]
    assert result["assessment"]["probe_execution"] == "NOT_ATTESTED_BY_GRAPH"


def test_cli_rejects_symlink_and_overwrite(tmp_path, capsys):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(capture().model_dump(mode="json")))
    original = path.read_bytes()
    args = parser().parse_args(
        [
            "ros",
            "diagnostic-plan",
            "--operational-graph",
            str(path),
            "--output",
            str(path),
            "--json",
        ]
    )
    assert dispatch_ros_command(args) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]
    assert path.read_bytes() == original
    link = tmp_path / "link.json"
    link.symlink_to(path)
    args.operational_graph = str(link)
    args.output = None
    assert dispatch_ros_command(args) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]
