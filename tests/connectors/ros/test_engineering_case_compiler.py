"""Pure counterfactual engineering cases; no ROS, model or physical execution."""

import json
from copy import deepcopy
from datetime import timedelta

import pytest

from rosclaw.connectors.ros.engineering import compile_engineering_case, summarize_case
from rosclaw.contracts.common import content_hash
from tests.connectors.ros.test_navigation_observation_review import NOW, navigation_fixture


def setup():
    model, urdf, requirements = navigation_fixture()
    requirements["interface_names"].update(
        {
            "sensing.lidar": "/sample/scan2",
            "state.odometry": "/sample/position_feedback",
            "navigation.navigate_to_pose": "/sample/go",
        }
    )
    for topic in model.graph["topics"]:
        if topic["name"] in requirements["costmap_topics"]:
            node = topic["publishers"][0]
            model.observations["node_parameters"][node] = {
                "robot_radius": 0.4,
                "footprint": "[]",
                "robot_base_frame": "chassis",
            }
            model.observations["parameter_captured_at"][node] = NOW.isoformat()
    loc = requirements["lifecycle_nodes"]["localization"]
    model.observations["node_parameters"][loc] = {"tf_broadcast": True}
    model.observations["parameter_captured_at"][loc] = NOW.isoformat()
    return model.seal(), urdf, requirements


def compile_case(model, urdf, requirements, **kw):
    return compile_engineering_case(
        model.seal(),
        urdf,
        requirements,
        source_ref="fixture:original-system",
        now=kw.pop("now", NOW),
        evidence_domain="SEALED_REPLAY",
        **kw,
    )


def statuses(case):
    return {c.id: c.status for c in case.constraints}


def test_healthy_snapshot_does_not_request_repair_or_admit_navigation():
    m, u, r = setup()
    before = deepcopy(m.to_dict())
    case = compile_case(m, u, r)
    assert all(s == "SATISFIED" for k, s in statuses(case).items() if k != "F-SOURCE-001")
    assert not case.repair_candidates
    assert case.current_authority == "NONE" and case.current_l1 == "NOT_EVALUATED"
    assert m.to_dict() == before


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("tf", "F-TF-001"),
        ("scan", "F-SCAN-001"),
        ("lifecycle", "F-LIFE-001"),
        ("footprint", "F-GEOM-001"),
        ("duplicate", "F-ROUTE-001"),
        ("new_fault_combination", "F-SCAN-001"),
    ],
)
def test_observed_faults_without_case_id_dispatch(fault, expected):
    m, u, r = setup()
    if fault == "tf":
        m.transforms = [e for e in m.transforms if e.parent != "world"]
        m.observations["node_parameters"][r["lifecycle_nodes"]["localization"]]["tf_broadcast"] = (
            False
        )
    if fault in {"scan", "new_fault_combination"}:
        next(s for s in m.signals if s.topic == "/sample/scan2").last_message_age_ms = 700
    if fault in {"lifecycle", "new_fault_combination"}:
        m.lifecycle[0].state = "INACTIVE"
    if fault == "footprint":
        m.observations["node_parameters"][r["lifecycle_nodes"]["planner"]]["robot_radius"] = 0.1
    if fault == "duplicate":
        next(t for t in m.graph["topics"] if t["name"] == r["velocity_output_topic"])[
            "publishers"
        ].append("/unknown/competing")
    case = compile_case(m, u, r)
    assert statuses(case)[expected] == "VIOLATED"
    assert all(c.stage_only and not c.applicable_now for c in case.repair_candidates)
    if fault == "scan":
        assert not any(
            c.semantic_field == "sensor.scan_bridge_state" for c in case.repair_candidates
        )
    if fault == "duplicate":
        assert (
            next(
                c for c in case.repair_candidates if c.semantic_field == "control.route_review"
            ).proposed_value
            is None
        )


@pytest.mark.parametrize("seconds", [-1, 1])
def test_future_and_stale_facts_cannot_generate_remedies(seconds):
    m, u, r = setup()
    m.observations["node_parameters"][r["lifecycle_nodes"]["planner"]]["robot_radius"] = 0.1
    case = compile_case(m, u, r, now=NOW + timedelta(seconds=seconds))
    assert not case.repair_candidates
    assert set(statuses(case).values()) == {"UNKNOWN"}


def test_wrong_source_or_run_suppresses_repair_candidates():
    m, u, r = setup()
    m.observations["node_parameters"][r["lifecycle_nodes"]["planner"]]["robot_radius"] = 0.1
    for source, extra in [
        ({"result": "MISMATCH"}, {}),
        ({"result": "MATCHED", "sim_run_id": "old"}, {"expected_run_id": "new"}),
        ({"result": "MATCHED", "ros_runtime_snapshot_hash": "wrong"}, {}),
    ]:
        case = compile_case(m, u, r, source_review=source, **extra)
        assert statuses(case)["F-SOURCE-001"] == "VIOLATED" and not case.repair_candidates
        assert all(c.status == "UNKNOWN" for c in case.constraints if c.id != "F-SOURCE-001")


def test_unsubstantiated_matched_flag_does_not_become_verified_source():
    m, u, r = setup()
    case = compile_case(m, u, r, source_review={"result": "MATCHED"})
    assert case.source_status == "UNKNOWN"


def test_competing_tf_and_contradictory_footprint_remain_unknown():
    m, u, r = setup()
    m.transforms.append(m.transforms[0].model_copy())
    m.observations["node_parameters"][r["lifecycle_nodes"]["planner"]]["footprint"] = (
        "[[0,0],[1,0],[0,1]]"
    )
    case = compile_case(m, u, r)
    assert statuses(case)["F-TF-001"] == "UNKNOWN"
    assert statuses(case)["F-GEOM-001"] == "UNKNOWN"
    assert not case.repair_candidates


def test_content_hash_replay_and_all_observed_reference_values_are_resolvable():
    m, u, r = setup()
    case = compile_case(m, u, r)
    again = compile_case(m, u, r)
    assert case.model_dump() == again.model_dump()
    docs = {
        "fixture:original-system": m.to_dict(),
        **{content_hash("derived", v): v for v in case.derived_documents.values()},
    }
    for ref in case.evidence_index.values():
        if ref.classification == "UNKNOWN":
            continue
        value = docs[ref.source_ref]
        for key in ref.pointer.split("/")[1:] if ref.pointer else []:
            key = key.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        assert content_hash("factvalue", value) == ref.value_hash
    assert len(json.dumps(summarize_case(case), ensure_ascii=False)) < 2500
    next(s for s in m.signals if s.topic == "/sample/scan2").last_message_age_ms = 800
    assert compile_case(m, u, r).case_hash != case.case_hash


def test_unsealed_mutation_and_unspecified_replay_clock_rejected():
    m, u, r = setup()
    m.errors.append("changed")
    with pytest.raises(ValueError, match="hash"):
        compile_engineering_case(m, u, r, source_ref="original", now=NOW)
    m.seal()
    with pytest.raises(ValueError, match="clock"):
        compile_engineering_case(m, u, r, source_ref="original", evidence_domain="SEALED_REPLAY")


def test_serialized_case_rejects_tampered_evidence_or_authority():
    from rosclaw.connectors.ros.engineering import RosEngineeringCaseV1

    model, urdf, requirements = setup()
    case = compile_case(model, urdf, requirements)
    data = case.model_dump(mode="json")
    assert RosEngineeringCaseV1.from_dict(data) == case
    data["constraints"][0]["observed"] = "altered"
    with pytest.raises(ValueError, match="hash"):
        RosEngineeringCaseV1.from_dict(data)
    data = case.model_dump(mode="json")
    data["current_authority"] = "REAL"
    with pytest.raises(ValueError):
        RosEngineeringCaseV1.from_dict(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("localization.tf_broadcast", "true"),
        ("costmap.radius", -1),
        ("control.route_review", "disable_unknown_node"),
        ("unsafe.velocity_limit", 10),
    ],
)
def test_unsafe_or_untyped_repair_intents_are_rejected(field, value):
    from rosclaw.connectors.ros.engineering import RepairIntentV1

    with pytest.raises(ValueError):
        RepairIntentV1(
            id="candidate",
            semantic_field=field,
            proposed_value=value,
            evidence_refs=[],
            preconditions=[],
        )


def test_architecture_is_pure_and_has_no_execution_imports():
    import ast
    from pathlib import Path

    directory = Path(__file__).resolve().parents[3] / "src/rosclaw/connectors/ros/engineering"
    forbidden = {
        "subprocess",
        "socket",
        "rclpy",
        "rclcpp",
        "roslibpy",
        "Runtime",
        "open",
        "exec",
        "eval",
    }
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] in forbidden for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not any(a.name in forbidden for a in node.names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden
