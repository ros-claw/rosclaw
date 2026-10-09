"""Installed SDK validation of generic source join; synthetic robot, no Node."""

import argparse
import ast
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import yaml
from generic_stack_source import prepare_generic_stack_source, read_prepared_generic_stack

from rosclaw.connectors.ros.context.sim_navigation_source import NODE_ROLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sdk-map-decoder", type=Path, required=True)
    parser.add_argument("--sdk-controller-parser", type=Path, required=True)
    parser.add_argument("--initialization-prior", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    controller_test = repo / "tests/connectors/ros/test_sim_controller_source.py"
    navigation_test = repo / "tests/connectors/ros/test_sim_navigation_source.py"
    stack_test = repo / "tests/connectors/ros/test_generic_stack_source.py"
    nav_template = Path("/opt/ros/jazzy/share/nav2_bringup/params/nav2_params.yaml")
    coverage_template = Path(
        "/ws/src/opennav_coverage/opennav_coverage_demo/params/demo_params.yaml"
    )
    originals = args.directory / "original-executed-source"
    originals.mkdir()
    hashes = {}
    paths = [
        Path(__file__),
        Path(__file__).with_name("generic_stack_source.py"),
        Path(__file__).with_name("generic_contact_fixture.py"),
        Path(__file__).with_name("backend_world_bundle.py"),
        args.sdk_map_decoder,
        args.sdk_controller_parser,
        repo / "src/rosclaw/connectors/ros/context/sim_controller_source.py",
        repo / "src/rosclaw/connectors/ros/context/sim_navigation_source.py",
        *[
            repo / "tests/connectors/ros" / p
            for p in (
                "test_generic_stack_source.py",
                "test_sim_controller_source.py",
                "test_sim_navigation_source.py",
            )
        ],
        *Path(__file__).with_name("generic_map_parser").iterdir(),
        Path("/opt/ros/jazzy/include/nav2_map_server/map_io.hpp"),
        Path("/opt/ros/jazzy/lib/libmap_io.so"),
        Path("/opt/ros/jazzy/share/nav2_bringup/params/nav2_params.yaml"),
        Path("/ws/src/opennav_coverage/opennav_coverage_demo/params/demo_params.yaml"),
        repo / "src/rosclaw/connectors/ros/context/sim_localization_source.py",
        repo / "src/rosclaw/connectors/ros/diagnosis/coverage_audit.py",
    ]
    for i, path in enumerate(paths):
        raw = path.read_bytes()
        (originals / (str(i) + "_" + path.name)).write_bytes(raw)
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    (args.directory / "before-execution-source-hashes.json").write_text(
        json.dumps(hashes, indent=2)
    )
    values = {}
    for node in ast.parse(controller_test.read_bytes()).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"URDF", "SDF", "POLICY"}
        ):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    navglobals = {"NODE_ROLES": NODE_ROLES}
    assignments = [
        n
        for n in ast.parse(navigation_test.read_bytes()).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in {"POLICY", "ATTACHMENT"}
    ]
    exec(
        compile(ast.Module(body=assignments, type_ignores=[]), str(navigation_test), "exec"),
        navglobals,
    )
    navigation = SimpleNamespace(
        POLICY=navglobals["POLICY"],
        ATTACHMENT=navglobals["ATTACHMENT"],
        ROOT=repo / "tests/fixtures/ros/navigation_source",
    )
    env = {
        "ET": ET,
        "deepcopy": deepcopy,
        "yaml": yaml,
        "controller": SimpleNamespace(**values),
        "navigation": navigation,
    }
    fn = next(
        n
        for n in ast.parse(stack_test.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == "synthetic_stack_inputs"
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(stack_test), "exec"), env)
    inputs = env["synthetic_stack_inputs"]()
    if (
        inputs["nav2_bytes"] != nav_template.read_bytes()
        or inputs["coverage_bytes"] != coverage_template.read_bytes()
    ):
        raise ValueError("repository fixture differs from actual installed template bytes")
    root = args.directory / "synthetic-joined-source"
    if args.initialization_prior:
        inputs["localization_initialization_declaration"] = {
            "schema_version": "rosclaw.sim_localization_initial_prior.v1",
            "source": "simulator_operator_fixture_policy",
            "approved": True,
            "evidence_domain": "SIMULATION",
            "source_pose_kind": "OPERATOR_FROZEN_SPAWN_PRIOR",
            "world_name": inputs["contact_declaration"]["world_name"],
            "map_frame": inputs["navigation_declaration"]["frames"]["map"],
            "world_to_map_xyyaw": [0, 0, 0],
        }
    manifest = prepare_generic_stack_source(root, **inputs)
    handoff = read_prepared_generic_stack(root)
    if handoff["manifest"] != manifest or handoff["authorization"] is not False:
        raise ValueError("prepared workspace handoff integrity failed")

    def actual(label, argv, expect=True):
        result = subprocess.run(argv, capture_output=True, timeout=20)
        (args.directory / (label + ".original-stdout")).write_bytes(result.stdout)
        (args.directory / (label + ".original-stderr")).write_bytes(result.stderr)
        if (result.returncode == 0) != expect:
            raise ValueError("installed SDK source check failed: " + label)
        return result

    actual("world", ["gz", "sdf", "-k", str(root / "world.sdf")])
    robot = ET.fromstring((root / "robot.sdf").read_bytes())
    controls = robot.findall("model/ros2_control")
    if len(controls) != 1:
        raise ValueError("one actual preserved control source required")
    (args.directory / "original-control-interface.xml").write_bytes(ET.tostring(controls[0]))
    robot.find("model").remove(controls[0])
    projection = args.directory / "standard-SDF-only-projection.sdf"
    projection.write_bytes(ET.tostring(robot))
    actual("standard_robot_projection", ["gz", "sdf", "-k", str(projection)])
    controller = json.loads((root / "controller-source-report.json").read_bytes())
    wheels = controller["source_wheel_names"]
    actual(
        "URDF_control_resources",
        [
            "" + str(args.sdk_controller_parser),
            str(root / "robot.urdf"),
            str(root / "controller_params.yaml"),
            inputs["controller_declaration"]["drive_controller"],
            ",".join(wheels["left"] + wheels["right"]),
        ],
    )
    actual(
        "navigation_parameters",
        [str(args.sdk_controller_parser), "--yaml-source-only", str(root / "nav2.yaml")],
    )
    decoded = actual("map_original", [str(args.sdk_map_decoder), str(root / "map.yaml")])
    row = json.loads(decoded.stdout.splitlines()[-1])
    if (row["width"], row["height"], row["data"], row["Node_started"]) != (
        2,
        2,
        [0, 0, 0, 0],
        False,
    ):
        raise ValueError("actual SDK map decode differs from original synthetic pixels")
    bad = args.directory / "invalid_map_source"
    bad.mkdir()
    (bad / "map.yaml").write_bytes(inputs["map_yaml_bytes"])
    (bad / "source_map.pgm").write_bytes(b"invalid image original source")
    actual("invalid_image_refusal", [str(args.sdk_map_decoder), str(bad / "map.yaml")], False)
    for path, sha in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("executed SDK source changed")
    for name, sha in manifest["output_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError("generated source changed during SDK validation")
    report = {
        "status": "PASS_INSTALLED_GENERIC_JOINED_SOURCE_SDK_CHECKS",
        "actual_map_dimensions": [row["width"], row["height"]],
        "actual_source_control_joints": len(wheels["left"] + wheels["right"]),
        "robot_SDF_validation_scope": "STANDARD_PROJECTION_ONLY_NONSTANDARD_ros2_control_SEPARATELY_PARSED_BY_HARDWARE_INTERFACE",
        "source_unchanged": True,
        "prepared_source_handoff_verified": True,
        "prepared_source_manifest_sha256": handoff["manifest_sha256"],
        "World_Node_action_or_model_started": False,
        "heldout_asset": "NOT_SELECTED",
        "physical_acceptance": "NOT_RUN",
        "authorization": False,
    }
    (args.directory / "source-contract-review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
