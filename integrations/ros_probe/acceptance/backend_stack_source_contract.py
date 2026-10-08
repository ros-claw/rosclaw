"""Whole known-SIM source preparation, actual Body compiler, no Node or World."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import stack
from backend_stack import prepare_backend_stack
from dynamic_bootstrap import prepare_bindings
from probe_scene_geometry import decode_scene_json
from profiles import PROFILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--physics-plugin", type=Path, required=True)
    parser.add_argument("--contact-plugin", type=Path, required=True)
    parser.add_argument("--instrument-service-binary", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=False)
    # Save the executed source and real ELF bytes before preparation.
    source = args.directory / "original-executed-source"
    source.mkdir()
    original = {}
    repo = Path(__file__).resolve().parents[3]
    inputs = list(Path(__file__).parent.glob("*.py")) + list((repo / "src").rglob("*.py"))
    for path in inputs:
        relative = path.relative_to(repo)
        raw = path.read_bytes()
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        original[str(path)] = hashlib.sha256(raw).hexdigest()
    for name in ("physics_plugin", "contact_plugin", "instrument_service_binary"):
        path = getattr(args, name)
        raw = path.read_bytes()
        (source / (name + ".executed-elf")).write_bytes(raw)
        original[str(path)] = hashlib.sha256(raw).hexdigest()
    (args.directory / "before-execution-source-hashes.json").write_text(
        json.dumps(original, indent=2)
    )
    reports = []
    for name, profile in PROFILES.items():
        root = args.directory / name
        root.mkdir()
        original_scene = root / "original-vendor-scene"
        stack.OUTPUT = original_scene
        stack.prepare(profile_name=name, coverage_preset="perimeter_stateless", seed=100831)
        bootstrap = root / "bootstrap"
        spec = prepare_bindings(
            bootstrap,
            urdf_path=original_scene / "robot.urdf",
            library_path=args.physics_plugin,
            mission_id="whole_source_stack_" + name,
            profile_name=name,
        )
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
        declaration_path = root / "explicit-probe-policy.json"
        declaration_path.write_text(json.dumps(declaration))
        output = root / "whole-stack"
        output.mkdir()
        plan = prepare_backend_stack(
            SimpleNamespace(
                directory=output,
                profile=name,
                coverage_preset="perimeter_stateless",
                seed=100831,
                brush_binding=bootstrap / "brush.json",
                physics_fixture=bootstrap / "physics.json",
                physics_plugin=args.physics_plugin,
                contact_plugin=args.contact_plugin,
                probe_declaration=declaration_path,
                instrument_service_binary=args.instrument_service_binary,
                instrument_service_binary_sha256=original[str(args.instrument_service_binary)],
            )
        )
        bundle = output / "backend-bundle"
        assert (output / "world.sdf").read_bytes() == (
            bundle / "world-source/world.sdf"
        ).read_bytes()
        binding = decode_scene_json((output / "physics_binding.json").read_bytes())
        assert (
            binding == decode_scene_json((output / "physics_fixture.json").read_bytes())["binding"]
        )
        assert binding == decode_scene_json((output / "physics.json").read_bytes())["binding"]
        assert set(binding["scene_model_names"]) == {
            model.get("name")
            for model in ET.parse(output / "world.sdf").getroot().findall("world/model")
        } | {profile.simulation_model}
        assert (output / "robot.urdf").read_bytes() == (original_scene / "robot.urdf").read_bytes()
        assert (output / "backend_actor_constraint.json").read_bytes() == (
            output / "backend-observer/backend_actor_constraint.json"
        ).read_bytes()
        assert (
            decode_scene_json((output / "backend_actor_constraint.json").read_bytes())[
                "body_snapshot_hash"
            ]
            == spec["binding"]["body_snapshot_hash"]
        )
        parsed = subprocess.run(
            ["gz", "sdf", "-k", str(output / "world.sdf")], capture_output=True, timeout=15
        )
        (root / "world-source-parser-stdout.txt").write_bytes(parsed.stdout)
        (root / "world-source-parser-stderr.txt").write_bytes(parsed.stderr)
        if parsed.returncode:
            raise ValueError("whole source stack world rejected by installed SDF parser")
        assert plan["actual_world_source_admission"] is False
        reports.append(
            {
                "profile": name,
                "body_snapshot_hash": spec["binding"]["body_snapshot_hash"],
                "whole_source_preparation": "PASS",
                "physical_acceptance": "NOT_RUN",
            }
        )
    for path, sha in original.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("executed source changed during source-only validation")
    report = {
        "status": "PASS_KNOWN_WHOLE_SOURCE_STACK",
        "reports": reports,
        "source_unchanged": True,
        "node_started": False,
        "world_started": False,
        "service_executed": False,
        "physical_acceptance": "NOT_RUN",
        "authorization": False,
    }
    (args.directory / "source-contract-review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
