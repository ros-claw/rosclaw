"""Actual installed SDK source parsers on synthetic XML, not held-out physics."""

import argparse
import ast
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

from rosclaw.connectors.ros.context.sim_controller_source import prepare_sim_controller_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--sdk-source-parser", required=True, type=Path)
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    fixture = repo / "tests/connectors/ros/test_sim_controller_source.py"
    paths = [
        fixture,
        Path(__file__),
        repo / "src/rosclaw/connectors/ros/context/sim_controller_source.py",
        *Path(__file__).with_name("generic_controller_parser").iterdir(),
        args.sdk_source_parser,
    ]
    originals = {}
    source = args.directory / "original-executed-source"
    source.mkdir()
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
    results = []
    for wheel_count in (2, 4):
        urdf, sdf = ET.fromstring(values["URDF"]), ET.fromstring(values["SDF"])
        if wheel_count == 4:
            for side in ("left", "right"):
                name, link = side + "_rear_axis", side + "_rear"
                ET.SubElement(urdf, "link", name=link)
                joint = ET.SubElement(urdf, "joint", name=name, type="continuous")
                ET.SubElement(joint, "parent", link="platform")
                ET.SubElement(joint, "child", link=link)
                ET.SubElement(joint, "axis", xyz="0 1 0")
                model = sdf.find("model")
                ET.SubElement(model, "link", name=link)
                joint = ET.SubElement(model, "joint", name=name, type="revolute")
                ET.SubElement(joint, "parent").text = "platform"
                ET.SubElement(joint, "child").text = link
                ET.SubElement(ET.SubElement(joint, "axis"), "xyz").text = "0 1 0"
                ET.SubElement(model.find("plugin"), side + "_joint").text = name
        generated = prepare_sim_controller_source(
            ET.tostring(urdf), ET.tostring(sdf), values["POLICY"]
        )
        root = args.directory / (str(wheel_count) + "-wheel-synthetic-source")
        root.mkdir()
        for name, raw in generated["source_files"].items():
            (root / name).write_bytes(raw)
        params = generated["controller_parameters"]
        drive = values["POLICY"]["drive_controller"]
        names = generated["report"]["source_wheel_names"]
        expected = ",".join(names["left"] + names["right"])
        cases = [("valid", deepcopy(params), (root / "robot.urdf").read_bytes(), True)]
        for fault in ("timeout", "prefix", "wheel_alias", "joint_mismatch", "bad_interface"):
            bad, body = deepcopy(params), (root / "robot.urdf").read_bytes()
            row = bad[drive]["ros__parameters"]
            if fault == "timeout":
                row["cmd_vel_timeout"] = 2.0
            elif fault == "prefix":
                row["tf_frame_prefix_enable"] = True
            elif fault == "wheel_alias":
                row["right_wheel_names"] = row["left_wheel_names"]
            elif fault == "joint_mismatch":
                row["right_wheel_names"] = ["invented_joint"]
            else:
                parsed = ET.fromstring(body)
                parsed.find("ros2_control/joint/command_interface").set("name", "position")
                body = ET.tostring(parsed)
            cases.append((fault, bad, body, False))
        for name, parameters, body, accepted in cases:
            case = root / name
            case.mkdir()
            (case / "robot.urdf").write_bytes(body)
            (case / "controllers.yaml").write_text(yaml.safe_dump(parameters))
            actual = subprocess.run(
                [
                    str(args.sdk_source_parser),
                    str(case / "robot.urdf"),
                    str(case / "controllers.yaml"),
                    drive,
                    expected,
                ],
                capture_output=True,
                timeout=15,
            )
            (case / "original-sdk-stdout.txt").write_bytes(actual.stdout)
            (case / "original-sdk-stderr.txt").write_bytes(actual.stderr)
            if (actual.returncode == 0) != accepted:
                raise ValueError(
                    "installed controller source parser result differs: "
                    + str(wheel_count)
                    + "/"
                    + name
                )
            if accepted:
                decoded = json.loads(actual.stdout)
                assert (
                    decoded["declared_control_joints"] == wheel_count
                    and decoded["nodes_parsed"] == 3
                )
                assert decoded["Node_started"] is False and decoded["hardware_loaded"] is False
            results.append(
                {
                    "wheels": wheel_count,
                    "case": name,
                    "returncode": actual.returncode,
                    "source_contract": "PASS",
                    "physical_acceptance": "NOT_RUN",
                }
            )
    for path, sha in originals.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("executed source bytes changed during installed SDK source check")
    report = {
        "status": "PASS_INSTALLED_GENERIC_CONTROLLER_SOURCE_PARSERS",
        "cases": results,
        "cases_passed": len(results),
        "source_unchanged": True,
        "heldout_asset": "NOT_SELECTED",
        "Node_started": False,
        "World_started": False,
        "hardware_loaded": False,
        "physical_acceptance": "NOT_RUN",
        "authorization": False,
    }
    (args.directory / "source-contract-review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
