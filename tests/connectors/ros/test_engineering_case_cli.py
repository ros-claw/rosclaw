"""Offline CLI contracts; fixtures do not prove actual navigation or hardware."""

import argparse
import hashlib
import json
from datetime import timedelta

import pytest

from rosclaw.connectors.ros.cli.ros_cli import (
    add_ros_subparser,
    cmd_ros_engineering_case,
    dispatch_ros_command,
)
from tests.connectors.ros.test_engineering_case_compiler import NOW, setup


def arguments(tmp_path, monkeypatch, *, tf_missing=False):
    from rosclaw.connectors.ros import expert

    def forbidden(*args, **kwargs):
        pytest.fail("offline engineering CLI must never inspect ROS")

    monkeypatch.setattr(expert, "inspect_system", forbidden)
    model, urdf, requirements = setup()
    if tf_missing:
        model.transforms = [edge for edge in model.transforms if edge.parent != "world"]
        model.observations["node_parameters"][requirements["lifecycle_nodes"]["localization"]][
            "tf_broadcast"
        ] = False
        model.seal()
    (tmp_path / "snapshot.json").write_text(json.dumps(model.to_dict()))
    (tmp_path / "robot.urdf").write_bytes(urdf)
    (tmp_path / "requirements.json").write_text(json.dumps(requirements))
    return argparse.Namespace(
        ros_command="engineering-case",
        snapshot=str(tmp_path / "snapshot.json"),
        urdf=str(tmp_path / "robot.urdf"),
        requirements=str(tmp_path / "requirements.json"),
        reviewed_at=NOW.isoformat(),
        source_review=None,
        expected_run_id=None,
        expected_body_hash=None,
        summary_only=False,
        output=None,
        json=True,
    )


def test_known_constraint_full_and_compact_replay_hashes_match(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path, monkeypatch, tf_missing=True)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    assert dispatch_ros_command(args) == 0
    full = json.loads(capsys.readouterr().out)
    assert (
        next(c["status"] for c in full["case"]["constraints"] if c["id"] == "F-TF-001")
        == "VIOLATED"
    )
    assert full["current_authority"] == "NONE" and full["current_l1"] == "NOT_EVALUATED"
    assert full["case"]["evidence_domain"] == "SEALED_REPLAY"
    args.summary_only = True
    assert cmd_ros_engineering_case(args) == 0
    compact = json.loads(capsys.readouterr().out)
    assert "case" not in compact and compact["case_hash"] == full["case_hash"]
    assert len(json.dumps(compact["summary"])) < 2500
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_old_replay_clock_does_not_make_stale_facts_current(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path, monkeypatch)
    args.reviewed_at = (NOW + timedelta(days=1)).isoformat()
    assert cmd_ros_engineering_case(args) == 0
    response = json.loads(capsys.readouterr().out)
    assert all(c["status"] == "UNKNOWN" for c in response["case"]["constraints"])
    assert "NOT_CURRENT_STATE" in response["scope"]


@pytest.mark.parametrize("reviewed_at", [None, "2026-10-16T10:00:00", "not-a-clock"])
def test_explicit_aware_review_clock_required(tmp_path, monkeypatch, capsys, reviewed_at):
    args = arguments(tmp_path, monkeypatch)
    args.reviewed_at = reviewed_at
    assert cmd_ros_engineering_case(args) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_tampered_snapshot_rejected_before_case_compilation(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path, monkeypatch)
    path = tmp_path / "snapshot.json"
    data = json.loads(path.read_text())
    data["observations"]["tampered"] = True
    path.write_text(json.dumps(data))
    assert cmd_ros_engineering_case(args) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_output_is_new_artifact_and_never_overwrites_input(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path, monkeypatch)
    raw = (tmp_path / "snapshot.json").read_bytes()
    args.output = args.snapshot
    assert cmd_ros_engineering_case(args) == 1
    capsys.readouterr()
    assert (tmp_path / "snapshot.json").read_bytes() == raw
    args.output = str(tmp_path / "derived.json")
    assert cmd_ros_engineering_case(args) == 0
    response = json.loads(capsys.readouterr().out)
    assert json.loads((tmp_path / "derived.json").read_text()) == response


def test_source_request_mismatch_remains_mismatch_without_admission(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path, monkeypatch)
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"result": "MATCHED", "sim_run_id": "actual-original-run"}))
    args.source_review = str(path)
    args.expected_run_id = "different-request"
    assert cmd_ros_engineering_case(args) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["case"]["source_status"] == "MISMATCH"
    assert response["current_authority"] == "NONE"
    assert (
        response["input_file_sha256"]["source_review"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_parser_requires_snapshot_and_cannot_select_live_graph_transport():
    parser = argparse.ArgumentParser()
    add_ros_subparser(parser.add_subparsers(dest="command"))
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "ros",
                "engineering-case",
                "--urdf",
                "x",
                "--requirements",
                "y",
                "--reviewed-at",
                NOW.isoformat(),
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "ros",
                "engineering-case",
                "--snapshot",
                "x",
                "--urdf",
                "x",
                "--requirements",
                "y",
                "--reviewed-at",
                NOW.isoformat(),
                "--endpoint",
                "ws://live",
            ]
        )
