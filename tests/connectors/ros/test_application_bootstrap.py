"""Driver-only synthetic sources: future Nav2 absence is not a base fault."""

import copy
import hashlib
from datetime import timedelta

import pytest
import yaml

from rosclaw.connectors.ros.engineering.application_bootstrap import (
    REQUIRED_COMPONENTS,
    SERVERS,
    compile_driver_bootstrap,
)
from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph
from tests.connectors.ros.test_navigation_body_workspace import NOW, source


def inputs(namespace="/anonymous/base", width=0.4):
    model, raw, requirements, policy = source(namespace, "moving_frame", "range_frame", width)
    # A real base with sensors and driver, deliberately no Nav2 nodes or map TF.
    model.graph["topics"] = [
        t
        for t in model.graph["topics"]
        if t["msg_type"] in {"nav_msgs/msg/Odometry", "sensor_msgs/msg/LaserScan"}
    ]
    model.graph["actions"] = []
    model.lifecycle = []
    topics = {t["name"] for t in model.graph["topics"]}
    model.signals = [s for s in model.signals if s.topic in topics]
    model.transforms = [e for e in model.transforms if e.parent != "world"]
    driver = namespace + "/wheel_driver"
    command = namespace + "/base_command"
    model.graph["topics"].append(
        {
            "name": command,
            "msg_type": "geometry_msgs/msg/TwistStamped",
            "publishers": [],
            "subscribers": [driver],
        }
    )
    model.qos["endpoints"] = [{"topic": command, "node": driver, "kind": "subscriber"}]
    model.environment["distro"] = "jazzy"
    model.observations["package_inventory"] = list(REQUIRED_COMPONENTS)
    model.observations["node_parameters"] = {
        namespace + "/manager": {
            "update_rate": 40,
            "wheel_driver.type": "diff_drive_controller/DiffDriveController",
        },
        driver: {"linear.x.max_velocity": 0.5, "angular.z.max_velocity": 1.0},
    }
    model.observations["parameter_captured_at"] = {
        n: NOW.isoformat() for n in model.observations["node_parameters"]
    }
    model.seal()
    graph = compile_operational_graph(model, source_ref="fixture://base-only", now=NOW)
    manifests = {}
    policies = []
    for name in REQUIRED_COMPONENTS:
        xml = f"<package><name>{name}</name><version>1.2.3</version></package>"
        manifests[name] = {"xml": xml, "sha256": hashlib.sha256(xml.encode()).hexdigest()}
        policies.append(
            {
                "package": name,
                "supported_ros": ["ros2"],
                "supported_distros": ["jazzy"],
                "required_version": "1.2.3",
                "policy_source_sha256": "a" * 64,
            }
        )
    request = {
        "source_revision": "b" * 40,
        "source_urdf_sha256": hashlib.sha256(raw).hexdigest(),
        "control": {
            "command_topic": command,
            "consumer_node": driver,
            "message_type": "geometry_msgs/msg/TwistStamped",
            "driver_package": "diff_drive_controller",
        },
        "controller_manager": namespace + "/manager",
        "namespace": namespace,
        "map_frame": "target_map",
        "body_id": "new-observed-body",
        "navigation_limits": policy["navigation_limits"],
        "component_policies": policies,
    }
    template = {s: {"ros__parameters": {"use_sim_time": False}} for s in SERVERS}
    for cm in ("local_costmap", "global_costmap"):
        template[cm] = {
            cm: {
                "ros__parameters": {
                    "obstacle_layer": {"scan": {"topic": "/old_robot/scan"}},
                    "inflation_layer": {"inflation_radius": 100.0},
                }
            }
        }
    bringup = {
        "schema_version": "rosclaw.prepared_source_bringup.v1",
        "autostart": True,
        "tf_topics_relative": True,
        "scan_ros_topic": namespace + "/scan2",
        "namespace": namespace,
        "controllers": {"manager": namespace + "/manager", "names": ["wheel_driver"]},
    }
    return graph, model, raw, bringup, template, manifests, request


def reseal(args):
    args[1].seal()
    args[0] = compile_operational_graph(args[1], source_ref="fixture://base-only", now=NOW)


@pytest.mark.parametrize(
    "namespace,width", [("/anonymous/base", 0.4), ("/renamed/new_platform", 0.8)]
)
def test_driver_only_can_propose_future_stack_without_reusing_body_or_authority(namespace, width):
    args = inputs(namespace, width)
    before = copy.deepcopy(args[1].to_dict())
    result = compile_driver_bootstrap(*args)
    assert result["status"] == "CONDITIONAL_DRIVER_BOOTSTRAP_STAGING"
    assert result["execution_authority"] == "NONE" and result["current_l1"] == "NOT_EVALUATED"
    assert result["target_status"] == "PLANNED_NOT_OBSERVED_OR_ADMITTED"
    assert result["driver_generated"] is False
    assert result["derived_frames"] == {
        "base": "moving_frame",
        "odom": "local",
        "lidar": "range_frame",
        "target_map": "target_map",
    }
    assert args[1].to_dict() == before
    params = yaml.safe_load(result["files"]["nav2.yaml"])[namespace]
    assert params["amcl"]["ros__parameters"]["set_initial_pose"] is False
    assert params["global_costmap"]["global_costmap"]["ros__parameters"]["robot_radius"] > width / 2
    assert all(params[s]["ros__parameters"]["use_sim_time"] is True for s in SERVERS)
    assert "/old_robot/" not in result["files"]["nav2.yaml"]


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("stale_manager", "NEEDS_READ_ONLY_PROBE"),
        ("nan_driver", "NEEDS_READ_ONLY_PROBE"),
        ("overspeed", "UNSUPPORTED_POLICY"),
        ("wrong_class", "NEEDS_ADAPTER"),
        ("missing_component", "NEEDS_DEPENDENCIES"),
        ("version", "INCOMPATIBLE_COMPONENTS"),
        ("stale_laser", "NEEDS_READ_ONLY_PROBE"),
        ("missing_driver", "NEEDS_DRIVER"),
    ],
)
def test_missing_or_wrong_prerequisites_emit_no_partial_files(fault, expected):
    args = list(inputs())
    node = args[6]["control"]["consumer_node"]
    if fault == "stale_manager":
        args[1].observations["parameter_captured_at"][args[6]["controller_manager"]] = (
            NOW - timedelta(seconds=1)
        ).isoformat()
    elif fault == "nan_driver":
        args[1].observations["node_parameters"][node]["linear.x.max_velocity"] = "not-finite"
    elif fault == "overspeed":
        args[6]["navigation_limits"]["max_speed_mps"] = 0.6
    elif fault == "wrong_class":
        args[1].observations["node_parameters"][args[6]["controller_manager"]][
            "wheel_driver.type"
        ] = "other/Class"
    elif fault == "missing_component":
        args[1].observations["package_inventory"].remove("nav2_amcl")
    elif fault == "version":
        args[6]["component_policies"][0]["required_version"] = "2.0.0"
    elif fault == "stale_laser":
        for s in args[1].signals:
            s.last_message_age_ms = 501
    else:
        args[1].graph["topics"] = [
            t for t in args[1].graph["topics"] if t["name"] != args[6]["control"]["command_topic"]
        ]
    reseal(args)
    result = compile_driver_bootstrap(*args)
    assert result["status"] == expected
    assert result["files"] is None


def test_predecessor_absolute_reference_cannot_leak_and_source_hash_cannot_be_forged():
    args = list(inputs())
    args[4]["planner_server"]["ros__parameters"]["unknown_hook"] = "/old_robot/hidden"
    with pytest.raises(ValueError, match="unbound predecessor"):
        compile_driver_bootstrap(*args)
    args = list(inputs())
    args[6]["source_urdf_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="URDF bytes"):
        compile_driver_bootstrap(*args)


def test_cli_driver_only_uses_bounded_originals_and_never_overwrites(tmp_path, capsys):
    import json

    from rosclaw.connectors.ros.cli.ros_cli import dispatch_ros_command
    from tests.connectors.ros.test_intervention_cli import parser

    args = inputs()
    values = {
        "operational-graph": args[0].model_dump(mode="json"),
        "snapshot": args[1].to_dict(),
        "driver-bringup": args[3],
        "software-template": args[4],
        "component-manifests": args[5],
        "request": args[6],
    }
    argv = ["ros", "application-bootstrap", "--json"]
    paths = []
    for name, value in values.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        paths.append(path)
        argv += ["--" + name, str(path)]
    raw = tmp_path / "robot.urdf"
    raw.write_bytes(args[2])
    argv += ["--urdf", str(raw)]
    before = {p: p.read_bytes() for p in paths + [raw]}
    output = tmp_path / "new-proposal.json"
    assert dispatch_ros_command(parser().parse_args(argv + ["--output", str(output)])) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["application_bootstrap"]["status"] == "CONDITIONAL_DRIVER_BOOTSTRAP_STAGING"
    assert result["current_authority"] == "NONE"
    saved = output.read_bytes()
    assert dispatch_ros_command(parser().parse_args(argv + ["--output", str(output)])) != 0
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert output.read_bytes() == saved
    assert all(p.read_bytes() == original for p, original in before.items())
    link = tmp_path / "alias.urdf"
    link.symlink_to(raw)
    argv[argv.index("--urdf") + 1] = str(link)
    assert dispatch_ros_command(parser().parse_args(argv)) != 0
    assert json.loads(capsys.readouterr().out)["ok"] is False


@pytest.mark.parametrize("fault", ["description", "tf_receive", "tf_stamp", "stale_envelope"])
def test_driver_only_does_not_renew_original_description_tf_or_envelope(fault):
    args = list(inputs())
    if fault == "description":
        for row in args[1].observations["urdf_descriptions"].values():
            row["captured_at"] = (NOW - timedelta(milliseconds=501)).isoformat()
    elif fault == "stale_envelope":
        args[1].captured_at -= timedelta(milliseconds=501)
    else:
        for edge in args[1].transforms:
            if not edge.static:
                if fault == "tf_receive":
                    edge.captured_at -= timedelta(milliseconds=501)
                else:
                    edge.age_ms = 501
    reseal(args)
    result = compile_driver_bootstrap(*args)
    assert result["status"] in {"NEEDS_READ_ONLY_PROBE", "ABSTAIN"}
    assert result["files"] is None


def test_installed_default_without_map_server_generates_explicit_target_and_excludes_unused_nodes():
    args = list(inputs())
    args[4].pop("map_server")
    args[4]["docking_server"] = {"ros__parameters": {"base_frame": "/predecessor/unused"}}
    result = compile_driver_bootstrap(*args)
    params = yaml.safe_load(result["files"]["nav2.yaml"])[args[6]["namespace"]]
    assert params["map_server"]["ros__parameters"]["yaml_filename"] == "/assets/map.yaml"
    assert "docking_server" not in params
