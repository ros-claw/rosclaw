"""Fresh configuration assembly on synthetic sources; no actual daemon dispatch."""

from copy import deepcopy
from datetime import timedelta

import pytest
import yaml

from rosclaw.connectors.ros.context.sim_execution_config import prepare_sim_execution_config
from rosclaw.connectors.ros.context.sim_workspace import compile_sim_body_workspace
from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor
from tests.connectors.ros.test_sim_attachment_region import example as region_example
from tests.connectors.ros.test_sim_execution_interfaces import NOW, execution_example


@pytest.fixture
def sources(tmp_path):
    model, data, attachment, policy = execution_example("/different/robot")
    workspace = tmp_path / "workspace"
    compiled = compile_sim_body_workspace(
        workspace, model, data, attachment=attachment, policy=policy, now=NOW
    )
    measured, region = region_example()
    measured.update(
        frame_id="world",
        source="/different/robot/floor",
        captured_at=NOW.isoformat(),
        observation_complete=True,
    )
    spawn = region["start_pose"]
    spawn.update(
        frame_id="world",
        yaw=0.4,
        body_snapshot_hash=compiled["body_snapshot_hash"],
        run_id="fresh_run",
    )
    kwargs = {
        "policy": policy,
        "attachment": attachment,
        "measured_map": measured,
        "start_pose": spawn,
        "allowed_polygon": region["allowed_polygon"],
        "mission_id": "mission",
        "now": NOW,
    }
    return workspace, model, data, kwargs


def test_generic_shifted_region_and_namespaced_configuration_constructs_daemon_contract(
    sources, tmp_path
):
    workspace, model, data, kwargs = sources
    before = deepcopy(kwargs)
    result = prepare_sim_execution_config(workspace, model, data, **kwargs)
    assert kwargs == before
    assert result["status"] == "READY_FOR_SIM_DAEMON_SOURCE_ADMISSION"
    assert result["physical_acceptance_level"] == "NOT_RUN" and not result["actions_dispatched"]
    assert result["requires_independent_source_admission"] is True
    config = result["configuration"]
    assert config["grid"]["frame_id"] == "world" and config["grid"]["origin"] == [4, -3]
    assert config["configured_spawn"] == [4.65, -2.35, 0.4]
    assert config["endpoints"]["navigate_complete_coverage"] == "/different/robot/cover"
    assert config["observation_topic"] == "/different/robot/independent"
    assert (
        config["grid"]["accessible_cells"] == result["region_preflight"]["grid"]["accessible_cells"]
    )
    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=None,
        output=tmp_path,
        body_id=config["body_id"],
        body_snapshot_hash=config["body_snapshot_hash"],
        grid=config["grid"],
        recovery_centers=config["recovery_centers"],
        endpoints=config["endpoints"],
        configured_spawn=config["configured_spawn"],
        mission_polygon=config["mission_polygon"],
    )
    goal = executor._coverage_goal(
        {"polygons": [{"points": [{"x": x, "y": y, "z": 0} for x, y in config["mission_polygon"]]}]}
    )
    assert goal["frame_id"] == "world"
    assert len(goal["polygons"][0]["points"]) == 7


@pytest.mark.parametrize(
    "fault",
    [
        "old_map",
        "future_map",
        "map_source",
        "map_incomplete",
        "body",
        "run",
        "frame",
        "yaw",
        "manifest",
        "compiled_urdf",
        "source_change",
    ],
)
def test_mismatched_or_stale_runtime_config_rejected(sources, fault):
    workspace, model, data, kwargs = sources
    if fault == "old_map":
        kwargs["measured_map"]["captured_at"] = (NOW - timedelta(seconds=1)).isoformat()
    elif fault == "future_map":
        kwargs["measured_map"]["captured_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif fault == "map_source":
        kwargs["measured_map"]["source"] = "/unobserved/map"
    elif fault == "map_incomplete":
        kwargs["measured_map"]["observation_complete"] = False
    elif fault == "body":
        kwargs["start_pose"]["body_snapshot_hash"] = "different"
    elif fault == "run":
        kwargs["start_pose"]["run_id"] = ""
    elif fault == "frame":
        kwargs["measured_map"]["frame_id"] = "map"
    elif fault == "yaw":
        kwargs["start_pose"]["yaw"] = float("nan")
    elif fault == "manifest":
        path = workspace / "sim-body-workspace.yaml"
        value = yaml.safe_load(path.read_text())
        value["body_snapshot_hash"] = "changed"
        path.write_text(yaml.safe_dump(value))
    elif fault == "compiled_urdf":
        from rosclaw.body.resolver import BodyResolver

        BodyResolver(workspace=workspace).eurdf_profile_path.with_name("robot.urdf").write_bytes(
            b"changed"
        )
    else:
        model.graph["actions"][-1]["name"] = "/changed"
    with pytest.raises(ValueError):
        prepare_sim_execution_config(workspace, model, data, **kwargs)
