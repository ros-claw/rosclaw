"""Anonymous source-backed interface compilation, never held-out physical proof."""

from copy import deepcopy
from datetime import timedelta

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.context.sim_binding import propose_sim_fixture_binding
from rosclaw.connectors.ros.context.sim_execution_interfaces import propose_sim_execution_interfaces
from rosclaw.connectors.ros.context.sim_workspace import compile_sim_body_workspace
from rosclaw.connectors.ros.intelligence.system_model import Lifecycle, Signal
from tests.connectors.ros.test_sim_binding_proposal import NOW, example


def execution_example(namespace="/sample"):
    model, data, attachment, policy = example(namespace)
    endpoints = {
        "navigate_to_pose": namespace + "/go",
        "navigate_through_poses": namespace + "/through",
        "navigate_complete_coverage": namespace + "/cover",
        "set_initial_pose": namespace + "/init",
        "lease": namespace + "/lease",
        "cleaning": namespace + "/brush_switch",
        "hold": namespace + "/hold",
    }
    node, observation = namespace + "/coverage_node", namespace + "/independent"
    model.graph["actions"] += [
        {
            "name": endpoints["navigate_through_poses"],
            "action_type": "nav2_msgs/action/NavigateThroughPoses",
        },
        {
            "name": endpoints["navigate_complete_coverage"],
            "action_type": "opennav_coverage_msgs/action/NavigateCompleteCoverage",
        },
    ]
    model.graph["services"] += [
        {"name": endpoints["set_initial_pose"], "srv_type": "nav2_msgs/srv/SetInitialPose"},
        {"name": endpoints["lease"], "srv_type": "std_srvs/srv/SetBool"},
        {"name": endpoints["hold"], "srv_type": "std_srvs/srv/SetBool"},
    ]
    model.graph["topics"].append({"name": observation, "msg_type": "std_msgs/msg/String"})
    model.signals.append(
        Signal(
            topic=observation,
            source="native",
            captured_at=NOW,
            publisher_count=1,
            last_message_age_ms=10,
        )
    )
    model.lifecycle.append(
        Lifecycle(name=node, source=node + "/get_state", captured_at=NOW, state="ACTIVE")
    )
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    policy["execution_interfaces"] = {
        "endpoints": endpoints,
        "observation_topic": observation,
        "coverage_lifecycle_node": node,
    }
    return model, data, attachment, policy


@pytest.mark.parametrize("namespace", ["/sample", "/other/renamed"])
def test_fresh_generic_coverage_binding_compiles_exact_declared_endpoints(tmp_path, namespace):
    model, data, attachment, policy = execution_example(namespace)
    result = compile_sim_body_workspace(
        tmp_path / "body", model, data, attachment=attachment, policy=policy, now=NOW
    )
    proposal = result["execution_interface_proposal"]
    assert proposal["status"] == "READY_FOR_SIM_INTERFACE_COMPILATION"
    assert proposal["endpoints"] == policy["execution_interfaces"]["endpoints"]
    assert proposal["requires_independent_source_and_actual_map_admission"] is True
    assert (
        proposal["physical_acceptance_level"] == "NOT_RUN"
        and proposal["capabilities_granted"] == []
    )
    body = BodyResolver(workspace=tmp_path / "body").get_effective_body(recompile_if_stale=False)
    binding = body.provider_interfaces["ros_capability_bindings"]["coverage.execute"]
    assert binding["name"] == namespace + "/cover"
    assert binding["lifecycle_nodes"] == {"coverage_server": namespace + "/coverage_node"}
    assert (
        body.provider_interfaces["sim_fixture_evidence"]["execution_interface_proposal"] == proposal
    )
    assert not result["direct_actions_dispatched"] and not result["usable_for_real_execution"]


@pytest.mark.parametrize(
    "fault",
    [
        "coverage_type",
        "coverage_duplicate",
        "initial_type",
        "missing_lease",
        "inactive",
        "old_lifecycle",
        "wrong_lifecycle_source",
        "observer_type",
        "observer_publishers",
        "observer_age",
        "observer_capture",
        "body_mismatch",
        "extra",
        "relative",
    ],
)
def test_missing_stale_ambiguous_or_mismatched_interfaces_refuse_compilation(tmp_path, fault):
    model, data, attachment, policy = execution_example()
    specification = propose_sim_fixture_binding(
        model, data, attachment=attachment, policy=policy, now=NOW
    )["specification"]
    if fault == "coverage_type":
        model.graph["actions"][-1]["action_type"] = "nav2_msgs/action/NavigateToPose"
    elif fault == "coverage_duplicate":
        model.graph["actions"].append(deepcopy(model.graph["actions"][-1]))
    elif fault == "initial_type":
        model.graph["services"][-3]["srv_type"] = "std_srvs/srv/SetBool"
    elif fault == "missing_lease":
        model.graph["services"].pop(-2)
    elif fault == "inactive":
        model.lifecycle[-1].state = "INACTIVE"
    elif fault == "old_lifecycle":
        model.lifecycle[-1].captured_at = NOW - timedelta(seconds=6)
    elif fault == "wrong_lifecycle_source":
        model.lifecycle[-1].source = "unobserved"
    elif fault == "observer_type":
        model.graph["topics"][-1]["msg_type"] = "std_msgs/msg/Bool"
    elif fault == "observer_publishers":
        model.signals[-1].publisher_count = 2
    elif fault == "observer_age":
        model.signals[-1].last_message_age_ms = 300
    elif fault == "observer_capture":
        model.signals[-1].captured_at = NOW - timedelta(seconds=6)
    elif fault == "body_mismatch":
        policy["execution_interfaces"]["endpoints"]["cleaning"] = "/another/clean"
    elif fault == "extra":
        policy["execution_interfaces"]["endpoints"]["cmd_vel"] = "/actuator"
    else:
        policy["execution_interfaces"]["observation_topic"] = "relative"
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    result = propose_sim_execution_interfaces(model, policy, specification, now=NOW)
    assert result["status"] == "UNKNOWN" and result["unknown_fields"]
    target = tmp_path / "rejected"
    with pytest.raises(ValueError):
        compile_sim_body_workspace(
            target, model, data, attachment=attachment, policy=policy, now=NOW
        )
    assert not target.exists()


def test_modified_unsealed_snapshot_refuses_interface_proposal():
    model, data, attachment, policy = execution_example()
    specification = propose_sim_fixture_binding(
        model, data, attachment=attachment, policy=policy, now=NOW
    )["specification"]
    model.graph["actions"][-1]["name"] = "/different"
    with pytest.raises(ValueError, match="snapshot hash"):
        propose_sim_execution_interfaces(model, policy, specification, now=NOW)


@pytest.mark.parametrize("fault", ["undeclared", "missing", "wrong_type", "duplicate"])
def test_generic_dynamic_hold_must_be_explicit_and_observed(fault):
    model, data, attachment, policy = execution_example("/renamed/robot")
    specification = propose_sim_fixture_binding(
        model, data, attachment=attachment, policy=policy, now=NOW
    )["specification"]
    hold = policy["execution_interfaces"]["endpoints"]["hold"]
    if fault == "undeclared":
        policy["execution_interfaces"]["endpoints"].pop("hold")
    elif fault == "missing":
        model.graph["services"] = [s for s in model.graph["services"] if s["name"] != hold]
    elif fault == "wrong_type":
        next(s for s in model.graph["services"] if s["name"] == hold)["srv_type"] = (
            "std_srvs/srv/Trigger"
        )
    else:
        model.graph["services"].append({"name": hold, "srv_type": "std_srvs/srv/SetBool"})
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    result = propose_sim_execution_interfaces(model, policy, specification, now=NOW)
    assert result["status"] == "UNKNOWN" and result["capabilities_granted"] == []
