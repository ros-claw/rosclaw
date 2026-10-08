"""Installed navigation templates/ROS YAML parser, synthetic source, no World."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

from rosclaw.connectors.ros.context.sim_controller_source import prepare_sim_controller_source
from rosclaw.connectors.ros.context.sim_navigation_source import (
    NODE_ROLES,
    prepare_sim_navigation_source,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--sdk-source-parser", required=True, type=Path)
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    source = args.directory / "original-executed-source"
    source.mkdir()
    fixture = repo / "tests/connectors/ros/test_sim_controller_source.py"
    paths = [
        fixture,
        Path(__file__),
        args.sdk_source_parser,
        repo / "src/rosclaw/connectors/ros/context/sim_controller_source.py",
        repo / "src/rosclaw/connectors/ros/context/sim_navigation_source.py",
        repo / "src/rosclaw/connectors/ros/context/geometry.py",
        repo / "src/rosclaw/connectors/ros/context/sim_attachment.py",
        *Path(__file__).with_name("generic_controller_parser").iterdir(),
        Path("/opt/ros/jazzy/share/nav2_bringup/params/nav2_params.yaml"),
        Path("/ws/src/opennav_coverage/opennav_coverage_demo/params/demo_params.yaml"),
    ]
    originals = {}
    for index, path in enumerate(paths):
        raw = path.read_bytes()
        (source / (str(index) + "_" + path.name)).write_bytes(raw)
        originals[str(path)] = hashlib.sha256(raw).hexdigest()
    (args.directory / "before-execution-source-hashes.json").write_text(
        json.dumps(originals, indent=2)
    )
    values = {}
    for node in ast.parse(fixture.read_bytes()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"URDF", "SDF", "POLICY"}
        ):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    urdf, sdf = ET.fromstring(values["URDF"]), ET.fromstring(values["SDF"])
    for model, is_urdf in ((urdf, True), (sdf.find("model"), False)):
        link = model.find("link")
        collision = ET.SubElement(link, "collision", name="body_envelope")
        geometry = ET.SubElement(collision, "geometry")
        if is_urdf:
            ET.SubElement(geometry, "box", size=".4 .3 .2")
        else:
            ET.SubElement(ET.SubElement(geometry, "box"), "size").text = ".4 .3 .2"
        ET.SubElement(model, "link", name="range_frame")
        joint = ET.SubElement(model, "joint", name="range_mount", type="fixed")
        if is_urdf:
            ET.SubElement(joint, "parent", link="platform")
            ET.SubElement(joint, "child", link="range_frame")
        else:
            ET.SubElement(joint, "parent").text = "platform"
            ET.SubElement(joint, "child").text = "range_frame"
    controller = prepare_sim_controller_source(
        ET.tostring(urdf), ET.tostring(sdf), values["POLICY"]
    )
    declaration = {
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "frames": {
            "base": "platform",
            "odom": "local_motion",
            "map": "declared_map",
            "lidar": "range_frame",
        },
        "nodes": {role: "/explicit_nav/" + role for role in NODE_ROLES},
        "topics": {
            role: "/source_topics/" + role
            for role in ["lidar", "odom", "map", "cmd_vel_raw", "cmd_vel_smoothed", "cmd_vel_safe"]
        },
        "endpoints": {
            role: "/explicit_actions/" + role
            for role in [
                "navigate_to_pose",
                "navigate_through_poses",
                "navigate_complete_coverage",
                "follow_path",
                "compute_path_to_pose",
                "compute_coverage_path",
                "set_initial_pose",
            ]
        },
        "spawn_xyyaw": [0.1, -0.2, 0.3],
        "operation_width_m": 0.30,
        "map_resolution": 0.05,
        "map_yaml_path": "/evidence/map.yaml",
        "coverage_bt_xml_path": "/ws/install/opennav_coverage_bt/share/opennav_coverage_bt/behavior_trees/navigate_w_basic_complete_coverage_nav_to_start.xml",
    }
    attachment = {
        "schema_version": "rosclaw.sim_cleaning_attachment.v1",
        "evidence_domain": "SIMULATION",
        "kind": "SIMULATED_CLEANING",
        "declaration_source": "explicit_synthetic_source_contract",
        "cleaning_polygon": [[-0.16, -0.16], [0.16, -0.16], [0.16, 0.16], [-0.16, 0.16]],
    }
    result = prepare_sim_navigation_source(
        controller["source_files"]["robot.urdf"],
        paths[-2].read_bytes(),
        paths[-1].read_bytes(),
        attachment=attachment,
        declaration=declaration,
        controller_report=controller["report"],
    )
    (args.directory / "robot.urdf").write_bytes(controller["source_files"]["robot.urdf"])
    (args.directory / "controller-source-report.json").write_text(
        json.dumps(controller["report"], indent=2)
    )
    (args.directory / "navigation-source-report.json").write_text(
        json.dumps(result["report"], indent=2)
    )
    (args.directory / "navigation-launch-source.json").write_text(
        json.dumps(result["launch_nodes"], indent=2)
    )
    parameters = args.directory / "nav2.yaml"
    parameters.write_text(yaml.safe_dump(result["parameters"]))
    actual = subprocess.run(
        [str(args.sdk_source_parser), "--yaml-source-only", str(parameters)],
        capture_output=True,
        timeout=15,
    )
    (args.directory / "original-sdk-stdout.txt").write_bytes(actual.stdout)
    (args.directory / "original-sdk-stderr.txt").write_bytes(actual.stderr)
    if actual.returncode:
        raise ValueError("installed ROS parameter parser rejected generic navigation source")
    parsed = json.loads(actual.stdout)
    if parsed["nodes_parsed"] != len(result["parameters"]) or parsed["Node_started"] is not False:
        raise ValueError("actual parameter parser lost a declared navigation source node")
    for path, sha in originals.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("executed source changed during installed navigation source check")
    report = {
        "status": "PASS_INSTALLED_NAVIGATION_TEMPLATE_AND_ROS_PARAMETER_SOURCE_PARSER",
        "nodes_parsed": parsed["nodes_parsed"],
        "source_unchanged": True,
        "Source_Node_World_or_action_started": False,
        "heldout_asset": "NOT_SELECTED",
        "physical_acceptance": "NOT_RUN",
        "authorization": False,
    }
    (args.directory / "source-contract-review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
