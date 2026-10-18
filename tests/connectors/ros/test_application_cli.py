"""Real offline CLI boundary; no discovery, execution or implicit file writes."""

import json

from rosclaw.connectors.ros.cli.ros_cli import dispatch_ros_command
from tests.connectors.ros.test_application_plan import artifacts, inputs
from tests.connectors.ros.test_intervention_cli import parser


def prepare(tmp_path):
    graph, model, urdf, requirements, manifests, request = inputs()
    files = {}
    for flag, value in [
        ("operational-graph", graph.model_dump(mode="json")),
        ("snapshot", model.to_dict()),
        ("requirements", requirements),
        ("component-manifests", manifests),
        ("request", request),
    ]:
        p = tmp_path / (flag + ".json")
        p.write_text(json.dumps(value))
        files[flag] = p
    files["urdf"] = tmp_path / "original.urdf"
    files["urdf"].write_bytes(urdf)
    args = ["ros", "application-plan", "--json"]
    for flag, path in files.items():
        args += ["--" + flag, str(path)]
    return args, files


def test_offline_plan_and_bundle_cli_grant_no_authority_and_reject_overwrite(
    tmp_path, monkeypatch, capsys
):
    from rosclaw.connectors.ros import expert
    from rosclaw.connectors.ros.engineering.application_plan import RosApplicationPlanV1

    monkeypatch.setattr(
        expert, "inspect_system", lambda **_: (_ for _ in ()).throw(AssertionError("no discovery"))
    )
    args, files = prepare(tmp_path)
    before = {k: p.read_bytes() for k, p in files.items()}
    out = tmp_path / "plan.json"
    assert dispatch_ros_command(parser().parse_args(args + ["--output", str(out)])) == 0
    response = json.loads(capsys.readouterr().out)
    plan = RosApplicationPlanV1.from_dict(response["application_plan"])
    assert plan.status == "CONDITIONAL_REUSE_STAGING" and response["current_authority"] == "NONE"
    raw = tmp_path / "artifacts.json"
    raw.write_text(json.dumps(artifacts(plan)))
    stage = list(args)
    stage[1] = "application-staging"
    stage += ["--plan", str(out), "--artifacts", str(raw)]
    assert dispatch_ros_command(parser().parse_args(stage)) == 0
    bundle = json.loads(capsys.readouterr().out)["application_staging"]
    assert bundle["physical_acceptance"] == "NOT_RUN" and bundle["driver_generated"] is False
    assert dispatch_ros_command(parser().parse_args(args + ["--output", str(out)])) != 0
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert {k: p.read_bytes() for k, p in files.items()} == before


def test_symlink_original_is_refused(tmp_path, capsys):
    args, files = prepare(tmp_path)
    link = tmp_path / "linked.json"
    link.symlink_to(files["request"])
    args[args.index("--request") + 1] = str(link)
    assert dispatch_ros_command(parser().parse_args(args)) != 0
    assert json.loads(capsys.readouterr().out)["ok"] is False
