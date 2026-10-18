"""Real parser/dispatch with offline originals, no transport or implicit execution."""

import argparse
import json

import yaml

from rosclaw.connectors.ros.cli.ros_cli import add_ros_subparser, dispatch_ros_command
from tests.connectors.ros.test_intervention_plan import inputs, request


def parser():
    root = argparse.ArgumentParser()
    add_ros_subparser(root.add_subparsers(dest="command"))
    return root


def prepare(tmp_path):
    graph, model, staging = inputs()
    data = {
        "graph": graph.model_dump(mode="json"),
        "snapshot": model.to_dict(),
        "proposal": request(),
        "bringup": staging["bringup.json"],
    }
    files = {}
    for key, value in data.items():
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(value))
        files[key] = path
    files["nav2"] = tmp_path / "nav2.yaml"
    files["nav2"].write_text(yaml.safe_dump(staging["nav2.yaml"]))
    args = [
        "ros",
        "intervention-plan",
        "--operational-graph",
        str(files["graph"]),
        "--snapshot",
        str(files["snapshot"]),
        "--proposal",
        str(files["proposal"]),
        "--staging-nav2",
        str(files["nav2"]),
        "--staging-bringup",
        str(files["bringup"]),
        "--json",
    ]
    return args, files


def test_cli_proposes_without_discovery_and_never_overwrites_input(tmp_path, monkeypatch, capsys):
    from rosclaw.connectors.ros import expert

    monkeypatch.setattr(
        expert,
        "inspect_system",
        lambda **_: (_ for _ in ()).throw(AssertionError("offline must not discover")),
    )
    args, files = prepare(tmp_path)
    original = {key: value.read_bytes() for key, value in files.items()}
    output = tmp_path / "derived.json"
    assert dispatch_ros_command(parser().parse_args(args + ["--output", str(output)])) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["intervention_plan"]["execution_status"] == "PROPOSED_NOT_EXECUTED"
    assert response["current_authority"] == "NONE" and json.loads(output.read_text()) == response
    assert {key: value.read_bytes() for key, value in files.items()} == original
    assert dispatch_ros_command(parser().parse_args(args + ["--output", str(files["nav2"])])) != 0
    assert not json.loads(capsys.readouterr().out)["ok"]
    assert files["nav2"].read_bytes() == original["nav2"]


def test_cli_rejects_symlink_and_yaml_aliases(tmp_path, capsys):
    args, files = prepare(tmp_path)
    files["nav2"].write_text(
        "node: &shared {ros__parameters: {transform_tolerance: 30.0}}\nother: *shared\n"
    )
    assert dispatch_ros_command(parser().parse_args(args)) != 0
    assert not json.loads(capsys.readouterr().out)["ok"]
    files["nav2"].write_text(yaml.safe_dump(inputs()[2]["nav2.yaml"]))
    target = tmp_path / "linked.json"
    target.symlink_to(files["proposal"])
    args[args.index("--proposal") + 1] = str(target)
    assert dispatch_ros_command(parser().parse_args(args)) != 0
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_result_cli_is_reference_binding_not_a_physical_success(tmp_path, capsys):
    from rosclaw.connectors.ros.engineering.experiment_result import ExperimentPhaseV1
    from tests.connectors.ros.test_experiment_result import fixtures

    plan, phases = fixtures()
    original = tmp_path / "plan.json"
    original.write_text(json.dumps({"intervention_plan": plan.model_dump(mode="json")}))
    records = tmp_path / "phases.json"
    records.write_text(
        json.dumps([ExperimentPhaseV1.model_validate(p).model_dump(mode="json") for p in phases])
    )
    args = parser().parse_args(
        ["ros", "experiment-result", "--plan", str(original), "--phases", str(records), "--json"]
    )
    assert dispatch_ros_command(args) == 0
    result = json.loads(capsys.readouterr().out)["experiment_result"]
    assert result["status"] == "NEEDS_INDEPENDENT_ORIGINAL_VERIFICATION"
    assert (
        result["original_files_read"] is False
        and result["causal_verification"] == "NOT_ESTABLISHED"
    )
