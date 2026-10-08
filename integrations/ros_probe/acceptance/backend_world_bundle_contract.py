"""Known vendor Body/source assembly and installed SDF parser, not physics.

This invokes source-only fixture renderers and the actual Body compiler. It
starts no Node, Gazebo world, scene service, actuator or Native task. It does
not select a held-out robot or admit a runtime source.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import stack
from backend_world_bundle import prepare_backend_world
from dynamic_bootstrap import prepare_bindings
from physics_fixture import prepare_physics
from profiles import PROFILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--physics-plugin", required=True, type=Path)
    parser.add_argument("--contact-plugin", required=True, type=Path)
    parser.add_argument("--installed-source-parser", type=Path)
    parser.add_argument("--instrument-service-binary", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=False)
    reports = []
    for name in ("waffle", "burger"):
        profile = PROFILES[name]
        root = args.directory / name
        root.mkdir()
        scene = root / "scene"
        stack.OUTPUT = scene
        stack.prepare(profile_name=name, coverage_preset="perimeter_stateless")
        bootstrap = root / "bootstrap"
        spec = prepare_bindings(
            bootstrap,
            urdf_path=scene / "robot.urdf",
            library_path=args.physics_plugin,
            mission_id="offline_source_bundle_" + name,
            profile_name=name,
        )
        brush = json.loads((bootstrap / "brush.json").read_text())
        (scene / "brush_binding.json").write_text(json.dumps(brush) + "\n")
        prepare_physics(
            scene,
            config_path=bootstrap / "physics.json",
            library_path=args.physics_plugin,
            brush_binding=brush,
            profile=profile,
        )
        # These are declared candidate support streams; only an actual live
        # source can prove support contact. Do not claim all touch the ground.
        support = json.loads((scene / "contact_topics.json").read_text())
        declaration = {
            "schema_version": "rosclaw.backend_probe_fixture_declaration.v1",
            "source": "simulator_operator_fixture_policy",
            "approved": True,
            "evidence_domain": "SIMULATION",
            "run_id": spec["binding"]["run_id"],
            "world_name": spec["binding"]["world_name"],
            "robot_model_name": profile.simulation_model,
            "probe_model_name": "owned_backend_instrument",
            "probe_xy": [6, 0],
            "sphere_radius_m": 0.05,
            "ground_z_m": 0,
            "lift_z_m": 10,
            "ground_collision_name": "floor::ground::ground",
            "cleaning_polygon": profile.cleaning_polygon,
            "maximum_robot_radius_m": 0.3,
            "pose_topic": "/instrument/pose",
            "component_topic": "/instrument/components",
            "contact_topic": "/instrument/contact",
        }
        original = {p.name: p.read_bytes() for p in scene.iterdir() if p.is_file()}
        result = prepare_backend_world(
            root / "bundle",
            scene_directory=scene,
            declaration=declaration,
            contact_library=args.contact_plugin,
            robot_support_topics=support,
            robot_ground_collisions=["floor::ground::ground"],
            robot_pose_topic="/rosclaw_sim/ground_truth",
            robot_pose_frame="ros_expert",
            probe_pose_frame="ros_expert",
            instrument_service_binary=args.instrument_service_binary,
            instrument_service_binary_sha256=hashlib.sha256(
                args.instrument_service_binary.read_bytes()
            ).hexdigest()
            if args.instrument_service_binary is not None
            else None,
        )
        if original != {p.name: p.read_bytes() for p in scene.iterdir() if p.is_file()}:
            raise ValueError("original known vendor fixture source changed during assembly")
        robot_source = root / "bundle/robot-source/robot.sdf"
        robot_tree = ET.fromstring(robot_source.read_bytes())
        projection = deepcopy(robot_tree)
        extensions = projection.find("model").findall("ros2_control")
        if len(extensions) != 1:
            raise ValueError("one explicit known ros2_control extension required")
        for extension in extensions:
            projection.find("model").remove(extension)
        projection_path = root / "robot-standard-SDF-projection-only.sdf"
        projection_path.write_bytes(ET.tostring(projection, encoding="utf-8"))
        # gz sdf -k's generic uniqueness check treats command/state interfaces
        # with the same name as duplicates inside a custom ros2_control tag.
        # Preserve the actual full robot source and those legitimate interfaces.
        full_parse = subprocess.run(
            ["gz", "sdf", "-p", str(robot_source)], capture_output=True, timeout=15
        )
        (root / "robot-full-source-parser-stdout.sdf").write_bytes(full_parse.stdout)
        (root / "robot-full-source-parser-stderr.txt").write_bytes(full_parse.stderr)
        if full_parse.returncode:
            raise ValueError("actual full robot source rejected by installed SDF parse/print")
        actual_control = robot_tree.find("model/ros2_control")
        parsed_control = ET.fromstring(full_parse.stdout).find("model/ros2_control")

        def structure(element):
            if element is None:
                raise ValueError("full SDF parser lost required ros2_control extension")
            return (
                element.tag,
                sorted(element.attrib.items()),
                (element.text or "").strip(),
                [structure(child) for child in element],
            )

        if structure(actual_control) != structure(parsed_control):
            raise ValueError("actual full SDF parser changed command/state interface source")
        source_parser_result = "NOT_RUN"
        controller_plan = "NOT_RUN"
        if args.instrument_service_binary is not None:
            plan_directory = root / "controller-source-plan"
            plan_directory.mkdir(mode=0o700)
            planned = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("owned_probe_controller.py")),
                    "--directory",
                    str(plan_directory),
                    "--bundle-directory",
                    str(root / "bundle"),
                    "--observer-directory",
                    str(root / "bundle/observer-source"),
                    "--physics-plugin",
                    str(scene / "librosclaw_passive_physics.so"),
                    "--physics-plugin-sha256",
                    hashlib.sha256(args.physics_plugin.read_bytes()).hexdigest(),
                    "--world-pid",
                    str(os.getpid()),
                    "--world-uid",
                    str(os.getuid()),
                    "--partition",
                    "rosclaw_backend_" + hashlib.sha256(brush["run_id"].encode()).hexdigest()[:32],
                    "--seed",
                    "100821",
                    "--duration",
                    "60",
                    "--prepare-only",
                ],
                capture_output=True,
                timeout=15,
            )
            (root / "controller-prepare-only-stdout.json").write_bytes(planned.stdout)
            (root / "controller-prepare-only-stderr.txt").write_bytes(planned.stderr)
            if planned.returncode:
                raise ValueError("known source-only controller plan failed")
            controller_plan = json.loads(planned.stdout)
            if (
                controller_plan.get("world_source_admission") is not False
                or controller_plan.get("authorization") is not False
            ):
                raise ValueError("source-only controller plan falsely admitted World or authority")
        if args.installed_source_parser is not None:
            control_projection = ET.fromstring((scene / "robot.urdf").read_bytes())
            original_controls = control_projection.findall("ros2_control")
            if original_controls:
                if len(original_controls) != 1 or structure(original_controls[0]) != structure(
                    actual_control
                ):
                    raise ValueError("original URDF/SDF control resource declarations differ")
            else:
                control_projection.append(deepcopy(actual_control))
            control_path = root / "actual-vendor-URDF-and-exact-control-extension-parser-input.urdf"
            control_path.write_bytes(ET.tostring(control_projection, encoding="utf-8"))
            checked = subprocess.run(
                [
                    str(args.installed_source_parser),
                    str(control_path),
                    str(scene / "bridge.yaml"),
                    str(scene / "truth_bridge.yaml"),
                    str(root / "bundle/robot-source/bridge.yaml"),
                    str(root / "bundle/world-source/bridge.yaml"),
                ],
                capture_output=True,
                timeout=15,
            )
            (root / "installed-source-parser-stdout.json").write_bytes(checked.stdout)
            (root / "installed-source-parser-stderr.txt").write_bytes(checked.stderr)
            if checked.returncode:
                raise ValueError("actual installed control/bridge source parser contract failed")
            source_parser_result = json.loads(checked.stdout)
        for path in (root / "bundle/world-source/world.sdf", projection_path):
            checked = subprocess.run(
                ["gz", "sdf", "-k", str(path)], capture_output=True, timeout=15
            )
            prefix = root / (
                "world-parser" if path.name == "world.sdf" else "robot-standard-projection-parser"
            )
            Path(str(prefix) + "-stdout.txt").write_bytes(checked.stdout)
            Path(str(prefix) + "-stderr.txt").write_bytes(checked.stderr)
            if checked.returncode or b"Valid." not in checked.stdout:
                raise ValueError("assembled known Body/world source rejected by actual SDF parser")
        reports.append(
            {
                "profile": name,
                "compiled_body_hash": brush["body_snapshot_hash"],
                "vendor_urdf_sha256": hashlib.sha256(
                    (scene / "robot.urdf").read_bytes()
                ).hexdigest(),
                "assembly_result": result,
                "source_scene_unchanged": True,
                "world_SDF_full_schema_check": "PASS",
                "robot_SDF_full_parse_print_and_control_extension_preservation": "PASS",
                "robot_standard_SDF_projection_schema_check": "PASS_EXCLUDES_ROS2_CONTROL_EXTENSION",
                "robot_original_full_generic_unique_name_check": "NOT_A_VALID_ROS2_CONTROL_SCHEMA_CHECK_INITIAL_FAILURE_RETAINED",
                "ros2_control_runtime_or_hardware_plugin_loading": "NOT_RUN",
                "installed_control_and_bridge_source_parser": source_parser_result,
                "owned_instrument_controller_prepare_only": controller_plan,
            }
        )
    summary = {
        "status": "PASS_KNOWN_VENDOR_BODY_SOURCE_ASSEMBLY_AND_SDF_PARSER",
        "reports": reports,
        "heldout_asset": "NOT_SELECTED",
        "node_or_DDS_started": False,
        "Gazebo_world_or_plugin_loaded": False,
        "scene_service_executed": False,
        "actuator_or_Native_task_started": False,
        "backend_health_admitted": False,
        "physical_acceptance": "NOT_RUN",
        "authorization": False,
    }
    (args.directory / "source-bundle-contract-review.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "known_bodies": len(reports),
                "physical_acceptance": "NOT_RUN",
            }
        )
    )


if __name__ == "__main__":
    main()
