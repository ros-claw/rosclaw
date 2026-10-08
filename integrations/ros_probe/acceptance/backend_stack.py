"""Owned known-SIM stack with mandatory independent backend constraints.

Source preparation creates no Node, World or service. Runtime is an operator
fixture launcher; robot task commands still need Native MCP and rosclawd.
This does not select or qualify a held-out robot or replace the frozen P0 stack.
"""

import argparse
import base64
import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from backend_probe_fixture import validate_probe_declaration
from backend_probe_world import bounded_source
from backend_world_bundle import prepare_backend_world
from backend_world_ownership import WorldSourceOwner
from physics_fixture import prepare_physics
from probe_scene_geometry import decode_scene_json
from profiles import PROFILES

from experiments import gazebo_arguments

ROOT = Path(__file__).resolve().parent


def prepare_backend_stack(args):
    import stack

    output = args.directory
    if output.is_symlink() or not output.is_dir() or (output / "world.sdf").exists():
        raise ValueError("owned fresh known-SIM stack source output required")
    stack.OUTPUT = output
    stack.prepare(True, args.profile, args.coverage_preset, args.seed)
    brush = decode_scene_json(bounded_source(args.brush_binding))
    if type(brush) is not dict or set(brush) != {
        "run_id",
        "body_snapshot_hash",
        "attachment_hash",
        "producer_id",
    }:
        raise ValueError("actual frozen SIM Body/brush binding required")
    (output / "brush_binding.json").write_text(json.dumps(brush) + "\n")
    (output / "run_id.txt").write_text(brush["run_id"] + "\n")
    profile = PROFILES[args.profile]
    prepare_physics(
        output,
        config_path=args.physics_fixture,
        library_path=args.physics_plugin,
        brush_binding=brush,
        profile=profile,
    )
    declaration = validate_probe_declaration(
        decode_scene_json(bounded_source(args.probe_declaration))
    )
    if (
        declaration["run_id"] != brush["run_id"]
        or declaration["robot_model_name"] != profile.simulation_model
    ):
        raise ValueError("explicit owned instrument declaration differs from actual known Body/run")
    bundle = output / "backend-bundle"
    result = prepare_backend_world(
        bundle,
        scene_directory=output,
        declaration=declaration,
        contact_library=args.contact_plugin,
        robot_support_topics=decode_scene_json(bounded_source(output / "contact_topics.json")),
        robot_ground_collisions=[declaration["ground_collision_name"]],
        robot_pose_topic="/rosclaw_sim/ground_truth",
        robot_pose_frame=declaration["world_name"],
        probe_pose_frame=declaration["world_name"],
        instrument_service_binary=args.instrument_service_binary,
        instrument_service_binary_sha256=args.instrument_service_binary_sha256,
    )
    # Witness and Native dynamic admission must see every actual scene model.
    # All originals remain in the robot-source copy inside the final manifest.
    (output / "world.original-source.sdf").write_bytes((output / "world.sdf").read_bytes())
    (output / "physics_binding.original-source.json").write_bytes(
        (output / "physics_binding.json").read_bytes()
    )
    (output / "world.sdf").write_bytes((bundle / "world-source/world.sdf").read_bytes())
    (output / "physics_binding.json").write_bytes(
        (bundle / "world-source/physics_binding.json").read_bytes()
    )
    fixture = decode_scene_json(bounded_source(output / "physics_fixture.json"))
    fixture["binding"] = decode_scene_json(bounded_source(output / "physics_binding.json"))
    (output / "physics_fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    observer, controller = output / "backend-observer", output / "backend-controller"
    observer.mkdir(mode=0o700)
    controller.mkdir(mode=0o700)
    config = (bundle / "observer-source/backend_actor_constraint.json").read_bytes()
    (observer / "backend_actor_constraint.json").write_bytes(config)
    (output / "backend_actor_constraint.json").write_bytes(config)
    experiment_path = output / "experiment.json"
    (output / "experiment.original-source.json").write_bytes(experiment_path.read_bytes())
    experiment = decode_scene_json(bounded_source(experiment_path))
    experiment["backend_source_plan"] = {
        "world_arguments": gazebo_arguments(bundle / "world-source/world.sdf", args.seed),
        "required_independent_backend_observation": True,
        "physical_acceptance": "NOT_RUN",
    }
    experiment_path.write_text(json.dumps(experiment, indent=2) + "\n")
    partition = "rosclaw_backend_" + hashlib.sha256(brush["run_id"].encode()).hexdigest()[:32]
    world = bundle / "world-source/world.sdf"
    plan = {
        "role": "OWNED_KNOWN_SIM_BACKEND_SOURCE_PLAN_NOT_PHYSICAL_ACCEPTANCE",
        "profile": args.profile,
        "run_id": brush["run_id"],
        "body_snapshot_hash": brush["body_snapshot_hash"],
        "bundle_manifest_sha256": hashlib.sha256(
            (bundle / "backend-world-bundle.json").read_bytes()
        ).hexdigest(),
        "GZ_PARTITION": partition,
        "gazebo_arguments": gazebo_arguments(world, args.seed),
        "world_name": declaration["world_name"],
        "robot_model_name": profile.simulation_model,
        "physics_plugin_sha256": fixture["plugin_sha256"],
        "instrument_service_binary_sha256": args.instrument_service_binary_sha256,
        "constraint_policy_hash": result["constraint_policy_hash"],
        "actor_require_backend_observation": True,
        "guarded_robot_task_path": "Native_MCP_rosclawd_ONLY",
        "actual_world_source_admission": False,
        "physical_acceptance": "NOT_RUN",
        "authorization": False,
    }
    (output / "backend-stack-source-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def remaining_seconds(deadline, cap):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("immutable owned fixture deadline exhausted")
    return min(cap, remaining)


class OwnedStackChildren:
    def __init__(self, directory, deadline):
        self.directory = directory
        self.deadline = deadline
        self.children = []
        self.logs = []

    def start(self, name, argv):
        remaining_seconds(self.deadline, 1)
        log = (self.directory / (name + ".log")).open("x")
        self.logs.append(log)
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append((name, child))
        return child

    def failed(self):
        return [
            {"name": name, "returncode": child.poll()}
            for name, child in self.children
            if child.poll() is not None
        ]

    def check(self):
        failures = self.failed()
        if failures:
            raise ValueError("owned backend dependency exited: " + failures[0]["name"])

    def withdraw_observation(self):
        # The owned actor's immutable source-age watchdog closes its gate.
        # Keep the actor, witness and World alive for independent stop proof.
        for name, child in self.children:
            if name == "backend-independent-observer" and child.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGINT)

    def close(self):
        # World was started first and is stopped last, after source/audit flush.
        for _, child in reversed(self.children):
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
        for log in self.logs:
            log.close()


class RuntimeFaultLatch:
    """Original fault evidence; never a physical-stop assertion or permission."""

    def __init__(self, directory, children):
        self.path = directory / "backend-stack-fault.json"
        self.children = children
        self.latched = False

    def fail(self, error, *, original_observation=None, world_alive=True):
        if self.latched:
            return
        self.latched = True
        record = {
            "source": "owned_backend_stack_supervisor",
            "error": str(error),
            "captured_at_unix_ns": time.time_ns(),
            "failed_dependencies": self.children.failed(),
            "original_observation_base64": base64.b64encode(original_observation).decode()
            if original_observation is not None
            else None,
            "world_alive_at_fault": world_alive,
            "physical_stop_proof": "NOT_MEASURED" if world_alive else "MISSING_STOP_PROOF",
            "physical_acceptance": "FAILED",
            "authorization": False,
        }
        with self.path.open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
        self.children.withdraw_observation()


def launch_backend_stack(args, plan, *, deadline):
    out, bundle = args.directory, args.directory / "backend-bundle"
    os.environ["GZ_PARTITION"] = plan["GZ_PARTITION"]
    os.environ["PYTHONPATH"] = (
        str(ROOT.parents[2] / "src") + os.pathsep + os.getenv("PYTHONPATH", "")
    )
    os.environ["GZ_SIM_SYSTEM_PLUGIN_PATH"] = (
        str(bundle / "world-source") + os.pathsep + os.getenv("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    )
    children = OwnedStackChildren(out, deadline)

    def terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        world = children.start("backend-gazebo", plan["gazebo_arguments"])
        time.sleep(remaining_seconds(deadline, 3))
        children.check()
        subprocess.run(
            [
                "ros2",
                "run",
                "ros_gz_sim",
                "create",
                "-world",
                plan["world_name"],
                "-file",
                str(bundle / "robot-source/robot.sdf"),
                "-name",
                plan["robot_model_name"],
                "-z",
                "0.05",
            ],
            check=True,
            timeout=remaining_seconds(deadline, 30),
        )
        children.start(
            "backend-bridge",
            [
                "ros2",
                "run",
                "ros_gz_bridge",
                "parameter_bridge",
                "--ros-args",
                "-p",
                f"config_file:={bundle / 'world-source/bridge.yaml'}",
                "-p",
                "use_sim_time:=true",
            ],
        )
        children.start(
            "backend-robot-state",
            [
                "ros2",
                "run",
                "robot_state_publisher",
                "robot_state_publisher",
                "--ros-args",
                "-p",
                "use_sim_time:=true",
                "-p",
                "robot_description:=" + (out / "robot.urdf").read_text(),
            ],
        )
        subprocess.run(
            [
                "ros2",
                "run",
                "controller_manager",
                "spawner",
                "joint_state_broadcaster",
                "drive_controller",
                "--controller-manager-timeout",
                "30",
            ],
            check=True,
            timeout=remaining_seconds(deadline, 40),
        )
        owner = WorldSourceOwner(
            bundle,
            world_pid=world.pid,
            world_uid=os.getuid(),
            partition=plan["GZ_PARTITION"],
            seed=args.seed,
            physics_plugin=out / "librosclaw_passive_physics.so",
            physics_plugin_sha256=plan["physics_plugin_sha256"],
            contact_plugin_sha256=decode_scene_json(
                bounded_source(bundle / "backend-world-bundle.json")
            )["source_contact_library_sha256"],
        )
        ownership = owner.check()
        (out / "backend-world-source-process.json").write_text(
            json.dumps(ownership, indent=2) + "\n"
        )
        instrument = children.start(
            "backend-owned-instrument",
            [
                "python3",
                str(ROOT / "owned_probe_controller.py"),
                "--directory",
                str(out / "backend-controller"),
                "--bundle-directory",
                str(bundle),
                "--observer-directory",
                str(out / "backend-observer"),
                "--physics-plugin",
                str(out / "librosclaw_passive_physics.so"),
                "--physics-plugin-sha256",
                plan["physics_plugin_sha256"],
                "--world-pid",
                str(world.pid),
                "--world-uid",
                str(os.getuid()),
                "--partition",
                plan["GZ_PARTITION"],
                "--seed",
                str(args.seed),
                "--duration",
                str(args.duration),
            ],
        )
        children.start(
            "backend-independent-observer",
            [
                "python3",
                str(ROOT / "backend_observer.py"),
                "--directory",
                str(out / "backend-observer"),
                "--robot-directory",
                str(bundle / "robot-source"),
                "--probe-directory",
                str(bundle / "instrument-source"),
                "--robot-policy",
                str(bundle / "robot-source/native-policy.json"),
                "--probe-policy",
                str(bundle / "instrument-source/probe-policy.json"),
                "--robot-plugin",
                str(bundle / "robot-source/librosclaw_passive_contacts.so"),
                "--probe-plugin",
                str(bundle / "instrument-source/librosclaw_passive_contacts.so"),
                "--scene-binding",
                str(bundle / "world-source/physics_binding.json"),
                "--probe-declaration",
                str(bundle / "world-source/probe-declaration.json"),
                "--scene-directory",
                str(bundle / "world-source"),
                "--robot-pose-frame",
                plan["world_name"],
                "--probe-pose-frame",
                plan["world_name"],
                "--duration",
                str(args.duration),
                "--controller-pid",
                str(instrument.pid),
                "--controller-uid",
                str(os.getuid()),
                "--instrument-service-binary",
                str(bundle / "instrument-source/owned_instrument_service"),
                "--instrument-service-binary-sha256",
                plan["instrument_service_binary_sha256"],
            ],
        )
        children.start("nav2", ["ros2", "launch", str(ROOT / "nav2_launch.py")])
        children.start(
            "backend-coverage",
            [
                "ros2",
                "run",
                "opennav_coverage",
                "opennav_coverage",
                "--ros-args",
                "--params-file",
                str(out / "nav2.yaml"),
                "-p",
                "use_sim_time:=true",
            ],
        )
        children.start(
            "coverage_lifecycle",
            [
                "ros2",
                "run",
                "nav2_lifecycle_manager",
                "lifecycle_manager",
                "--ros-args",
                "-r",
                "__node:=coverage_lifecycle_manager",
                "-p",
                "use_sim_time:=true",
                "-p",
                "autostart:=true",
                "-p",
                "node_names:=['coverage_server']",
            ],
        )
        children.start(
            "backend-rosbridge",
            [
                "ros2",
                "run",
                "rosbridge_server",
                "rosbridge_websocket",
                "--ros-args",
                "-p",
                "use_sim_time:=true",
                "-p",
                "port:=9090",
            ],
        )
        children.start(
            "backend-rosapi",
            ["ros2", "run", "rosapi", "rosapi_node", "--ros-args", "-p", "use_sim_time:=true"],
        )
        children.start(
            "backend-witness",
            [
                "python3",
                str(ROOT / "witness.py"),
                "--ros-args",
                "-p",
                "use_sim_time:=true",
                "-p",
                "controller_watchdog:=true",
                "-p",
                "split_actuator:=true",
                "-p",
                "dynamic_physics:=true",
            ],
        )
        children.start(
            "backend-sim-actuator",
            [
                "python3",
                str(ROOT / "sim_actuator.py"),
                "--ros-args",
                "-p",
                "use_sim_time:=true",
                "-p",
                "controller_watchdog:=true",
                "-p",
                "require_backend_observation:=true",
            ],
        )
        children.start("backend-graph-probe", ["python3", str(ROOT.parent / "ros2/probe.py")])
        fault = RuntimeFaultLatch(out, children)
        (out / "backend-stack-runtime-deadline.json").write_text(
            json.dumps(
                {
                    "deadline_monotonic": deadline,
                    "duration_sec_including_source_preparation": args.duration,
                    "physical_stop_proof": "NOT_MEASURED",
                    "authorization": False,
                }
            )
            + "\n"
        )
        while time.monotonic() < deadline:
            original = None
            world_alive = world.poll() is None
            if not world_alive:
                fault.fail("owned World exited", world_alive=False)
                raise ValueError("owned World exited; independent physical stop proof missing")
            if not fault.latched:
                try:
                    children.check()
                    owner.check()
                    latest = out / "backend-observer/backend-observation-latest.json"
                    if latest.exists():
                        original = bounded_source(latest)
                        snapshot = decode_scene_json(original)["snapshot"]
                        if snapshot.get("source_fault"):
                            raise ValueError("independent backend observation source rejected")
                except (ValueError, OSError, UnicodeError) as error:
                    fault.fail(error, original_observation=original)
            # On failure, preserve World/witness until the fixed fixture deadline.
            # Host must request a guarded stop and measure it; process exit is not proof.
            time.sleep(0.05)
    finally:
        children.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("/evidence"))
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--coverage-preset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--duration", required=True, type=int)
    for name in (
        "brush-binding",
        "physics-fixture",
        "physics-plugin",
        "contact-plugin",
        "instrument-service-binary",
        "probe-declaration",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--instrument-service-binary-sha256", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 60 <= args.duration <= 1920:
        raise ValueError("immutable bounded known-SIM backend deadline required")
    if not args.prepare_only and args.directory != Path("/evidence"):
        raise ValueError("known-SIM runtime dependencies require owned /evidence root")
    deadline = time.monotonic() + args.duration
    plan = prepare_backend_stack(args)
    if args.prepare_only:
        print(json.dumps(plan))
        return
    launch_backend_stack(args, plan, deadline=deadline)


if __name__ == "__main__":
    main()
