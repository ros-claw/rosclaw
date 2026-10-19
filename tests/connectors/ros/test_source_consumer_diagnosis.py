"""Synthetic static contracts, never claims of SDK execution or physical success."""

import argparse
import hashlib
import json
from datetime import timedelta

import pytest

from rosclaw.connectors.ros.cli.ros_cli import add_ros_subparser, dispatch_ros_command
from rosclaw.connectors.ros.engineering.hypothesis import evaluate_hypothesis, propose_hypotheses
from rosclaw.connectors.ros.engineering.probe_planner import plan_diagnostic_probes
from rosclaw.connectors.ros.engineering.source_consumer import (
    INVARIANT,
    RosLiteralConsumerContractV1,
    inspect_source_consumer,
)
from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1
from rosclaw.contracts.common import content_hash
from tests.connectors.ros.test_diagnostic_experiments import capture

SDK = "sha256:" + "1" * 64
LIBRARY = "2" * 64
ENTITY = "unfamiliar_platform"
URI = "http://ignitionrobotics.org/schema"


def sealed_graph(payload):
    payload.pop("graph_hash", None)
    return RosOperationalGraphV1.from_dict(
        {**payload, "graph_hash": content_hash("rosopgraph", payload)}
    )


def originals(prefix="ns0"):
    ode = f'<ode><mu>1</mu><mu2>0</mu2><fdir1 {prefix}:expressed_in="base">1 -1 0</fdir1></ode>'
    surface = f"<surface><friction>{ode}</friction></surface>"
    urdf = (
        f'<robot name="declared" xmlns:{prefix}="{URI}"><link name="base"/>'
        f'<link name="wheel"/><gazebo reference="wheel"><collision>{surface}'
        "</collision></gazebo></robot>"
    ).encode()
    world = (
        f'<sdf xmlns:{prefix}="{URI}"><world name="offline"><model name="{ENTITY}">'
        f'<link name="base"/><link name="wheel"><collision name="wheel">{surface}'
        "</collision></link></model></world></sdf>"
    ).encode()
    # Explicit synthetic graph context; not an authenticated Case or SDK capture.
    data = capture().model_dump(mode="json")
    data["source_urdf_sha256"] = hashlib.sha256(urdf).hexdigest()
    graph = sealed_graph(data)
    parser_evidence = b'{"fixture":"operator-declared synthetic parser knowledge"}'
    contract = RosLiteralConsumerContractV1(
        contract_hash="",
        sdk_image=SDK,
        library_sha256=LIBRARY,
        parser_evidence_ref="fixture:declared-parser-contract",
        parser_evidence_sha256=hashlib.sha256(parser_evidence).hexdigest(),
    ).model_dump(mode="json")
    contract.pop("contract_hash")
    contract["contract_hash"] = content_hash("rosconsumercontract", contract)
    return [
        graph,
        urdf,
        world,
        contract,
        json.dumps({"Image": SDK}).encode(),
        json.dumps({"library_sha256": LIBRARY, "status": "UNTRUSTED_LABEL"}).encode(),
        parser_evidence,
    ]


def inspect(values):
    return inspect_source_consumer(*values, robot_entity=ENTITY)


def consumer_finding(graph):
    return next(f for f in graph.findings if f.invariant == INVARIANT)


@pytest.mark.parametrize("prefix", ["ns0", "ignition", "another_alias"])
def test_source_mismatch_enters_hypothesis_and_readonly_plan_without_rewriting(prefix):
    values = originals(prefix)
    before = values[0].model_dump(mode="json")
    raw = values[1:3].copy()
    review, graph = inspect(values)
    assert review["status"] == "VIOLATED"
    assert review["attribute_spellings"] == [prefix + ":expressed_in"]
    assert values[0].model_dump(mode="json") == before and values[1:3] == raw
    assert [f.model_dump(mode="json") for f in graph.findings[:-1]] == before["findings"]
    finding = consumer_finding(graph)
    assert graph.provenance[finding.evidence_refs[0]].freshness == "UNKNOWN"
    plan = plan_diagnostic_probes(graph, issue_ids=[finding.id], budget_sec=10)
    assert {h.predicate for h in plan.hypotheses} == {"SOURCE_CONSUMER_CONFLICT", "UNRESOLVED_PATH"}
    hypothesis = next(h for h in plan.hypotheses if h.predicate == "SOURCE_CONSUMER_CONFLICT")
    assert hypothesis.status == "SUPPORTED_OBSERVATION"
    assert hypothesis.causal_verification == "NOT_ESTABLISHED"
    assert any(p.request.operation == "SOURCE_CONSUMER_CONTRACT" for p in plan.probes)
    assert all(p.request.execution_status == "PROPOSED_NOT_EXECUTED" for p in plan.probes)
    assert review["physical_stop"] == "NOT_EVALUATED" and graph.current_authority == "NONE"


def test_compatible_literal_is_only_static_satisfaction():
    review, graph = inspect(originals("gz"))
    assert review["status"] == "SATISFIED"
    assert review["runtime_consumption"] == "NOT_ATTESTED_BY_OFFLINE_BYTES"
    plan = plan_diagnostic_probes(graph, issue_ids=[consumer_finding(graph).id])
    assert plan.status == "NO_TARGET_ISSUE" and not plan.hypotheses
    assert graph.current_l1 == "NOT_EVALUATED"


@pytest.mark.parametrize("which", ["sdk", "library", "missing_library"])
def test_pass_like_labels_do_not_override_unknown_version(which):
    values = originals()
    if which == "sdk":
        values[4] = json.dumps({"Image": "sha256:" + "3" * 64, "passed": True}).encode()
    else:
        values[5] = json.dumps(
            {"library_sha256": "3" * 64 if which == "library" else None, "status": "PASS"}
        ).encode()
    review, graph = inspect(values)
    assert review["status"] == "UNKNOWN"
    assert all(
        h.status == "UNKNOWN"
        for h in propose_hypotheses(graph, issue_ids=[consumer_finding(graph).id])
    )


@pytest.mark.parametrize("old,new", [(b"1 -1 0", b"1 1 0"), (b"<mu2>0", b"<mu2>0.2")])
def test_other_changed_physics_is_not_misdiagnosed_as_alias_only(old, new):
    values = originals()
    values[2] = values[2].replace(old, new)
    assert inspect(values)[0]["status"] == "UNKNOWN"


def test_parser_original_hash_and_graph_urdf_binding_are_enforced():
    values = originals()
    values[6] = b"{}"
    with pytest.raises(ValueError, match="parser evidence"):
        inspect(values)
    values = originals()
    values[1] += b" "
    with pytest.raises(ValueError, match="URDF differs"):
        inspect(values)
    values = originals()
    values[3]["literal_attribute"] = "ns0:expressed_in"
    with pytest.raises(ValueError):
        inspect(values)


@pytest.mark.parametrize(
    "change", [{"evidence_domain": "OBSERVATION"}, {"source_status": "MISMATCH"}]
)
def test_live_or_conflicting_context_is_not_upgraded(change):
    values = originals()
    values[0] = sealed_graph({**values[0].model_dump(mode="json"), **change})
    with pytest.raises(ValueError, match="sealed replay"):
        inspect(values)


def test_static_finding_cannot_attest_newer_probe_or_fake_dynamic_freshness():
    _, graph = inspect(originals())
    finding = consumer_finding(graph)
    hypothesis = next(
        h
        for h in propose_hypotheses(graph, issue_ids=[finding.id])
        if h.predicate == "SOURCE_CONSUMER_CONFLICT"
    )
    assert (
        evaluate_hypothesis(
            hypothesis, graph, not_before=graph.reviewed_at - timedelta(milliseconds=1)
        ).status
        == "UNKNOWN"
    )
    data = graph.model_dump(mode="json")
    data["provenance"][finding.evidence_refs[0]]["freshness"] = "FRESH"
    assert evaluate_hypothesis(hypothesis, sealed_graph(data)).status == "UNKNOWN"
    data = graph.model_dump(mode="json")
    data["findings"][-1]["status"] = "SATISFIED"
    assert evaluate_hypothesis(hypothesis, sealed_graph(data)).status == "UNKNOWN"


def test_known_run_contradiction_fails_closed_and_duplicate_attachment_rejected():
    values = originals()
    data = values[0].model_dump(mode="json")
    data["entities"].append(
        {
            "id": "fixture-run",
            "kind": "run",
            "name": "1" * 32,
            "interface_type": None,
            "evidence_refs": [next(iter(data["provenance"]))],
        }
    )
    values[0] = sealed_graph(data)
    with pytest.raises(ValueError, match="run identity"):
        inspect(values)
    witness = b'<plugin name="rosclaw::SourceWitness"><run_id>' + b"1" * 32 + b"</run_id></plugin>"
    unrelated = b'<plugin name="unrelated"><run_id>' + b"2" * 32 + b"</run_id></plugin>"
    values[2] = values[2].replace(b"</world>", witness + unrelated + b"</world>")
    review, after = inspect(values)
    assert review["run_binding"] == "ORIGINAL_GRAPH_RUN_MATCHED"
    values[0] = after
    with pytest.raises(ValueError, match="already attached"):
        inspect(values)


def test_duplicate_declared_run_witness_is_not_resolved_by_matching_one_id():
    values = originals()
    data = values[0].model_dump(mode="json")
    data["entities"].append(
        {
            "id": "fixture-run",
            "kind": "run",
            "name": "1" * 32,
            "interface_type": None,
            "evidence_refs": [next(iter(data["provenance"]))],
        }
    )
    values[0] = sealed_graph(data)
    witness = b'<plugin name="rosclaw::SourceWitness"><run_id>' + b"1" * 32 + b"</run_id></plugin>"
    values[2] = values[2].replace(b"</world>", witness + witness + b"</world>")
    with pytest.raises(ValueError, match="run identity"):
        inspect(values)


def test_nested_namespace_scope_and_unrelated_model_do_not_change_target_verdict():
    values = originals("gz")
    values[2] = values[2].replace(f'xmlns:gz="{URI}"'.encode(), b"")
    values[2] = values[2].replace(b"<fdir1 ", f'<fdir1 xmlns:gz="{URI}" '.encode())
    values[2] = values[2].replace(b"</world>", b'<model name="unrelated"/></world>')
    assert inspect(values)[0]["status"] == "SATISFIED"


@pytest.mark.parametrize("bad", [b"<invalid", b"<!DOCTYPE sdf><sdf/>", b"", "not bytes"])
def test_invalid_or_unbounded_xml_is_rejected(bad):
    values = originals()
    values[2] = bad
    with pytest.raises(ValueError):
        inspect(values)


def test_cli_originals_to_graph_hypothesis_and_plan_no_discovery_or_overwrite(
    tmp_path, monkeypatch, capsys
):
    from rosclaw.connectors.ros import expert

    monkeypatch.setattr(expert, "inspect_system", lambda **_: pytest.fail("offline opened ROS"))
    values = originals()
    inputs = dict(
        zip(
            [
                "operational-graph",
                "urdf",
                "world-sdf",
                "consumer-contract",
                "container-inspection",
                "backend-observation",
                "parser-evidence",
            ],
            [
                json.dumps(values[0].model_dump(mode="json")).encode(),
                values[1],
                values[2],
                json.dumps(values[3]).encode(),
                *values[4:],
            ],
            strict=True,
        )
    )
    root = argparse.ArgumentParser()
    add_ros_subparser(root.add_subparsers(dest="command"))
    argv = ["ros", "source-consumer-review", "--robot-entity", ENTITY, "--json"]
    for name, raw in inputs.items():
        path = tmp_path / name
        path.write_bytes(raw)
        argv += ["--" + name, str(path)]
    output = tmp_path / "result.json"
    argv += ["--output", str(output)]
    args = root.parse_args(argv)
    assert dispatch_ros_command(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source_consumer_review"]["status"] == "VIOLATED"
    assert result["experiment"]["execution_status"] == "PROPOSED_NOT_EXECUTED"
    original = output.read_bytes()
    assert dispatch_ros_command(args) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]
    assert output.read_bytes() == original
    link = tmp_path / "alias"
    link.symlink_to(tmp_path / "world-sdf")
    args.output = None
    args.world_sdf = str(link)
    assert dispatch_ros_command(args) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]
