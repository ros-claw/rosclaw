"""Unexecuted owned SIM bootstrap; no Body admission or robot task dispatch.

The future owned runtime must enforce its immutable deadline and read-only
source mounts before executing this description. Controllers remain inactive.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

from generic_navigation_launch import build_launch_description, navigation_launch_plan
from generic_stack_source import read_prepared_generic_stack

from experiments import gazebo_arguments, validate_seed
from rosclaw.connectors.ros.context.sim_attachment import validate_sim_attachment
from rosclaw.connectors.ros.context.sim_navigation_source import _yaml
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.mission.sim_endpoints import absolute_endpoint


def bootstrap_launch_plan(directory, declaration):
    captured = read_prepared_generic_stack(directory)
    files, manifest = captured["captured_files"], captured["manifest"]
    declarations = json.loads(files["source-declarations.json"])
    controller = declarations["controller"]
    controller_report = json.loads(files["controller-source-report.json"])
    navigation_report = json.loads(files["navigation-source-report.json"])
    navigation = navigation_launch_plan(directory)
    if (
        type(controller_report) is not dict
        or captured["manifest_sha256"] != navigation["workspace_manifest_sha256"]
        or controller_report.get("declaration_sha256") != digest(controller)
        or navigation_report.get("declaration_hash") != digest(declarations["navigation"])
        or navigation_report.get("controller_source_report_hash") != digest(controller_report)
        or navigation_report.get("attachment")
        != validate_sim_attachment(declarations["attachment"])
        or json.loads(files.get("contact-source/instrumentation-policy.json", b"null"))
        != declarations["contact"]
        or controller_report.get("authorization") is not False
        or controller_report.get("live_controller_admitted") is not False
        or controller_report.get("physical_acceptance") != "NOT_RUN"
        or controller_report.get("body_model_name") != manifest["body_model_name"]
        or controller_report.get("controller_parameters_sha256")
        != digest(_yaml(files["controller_params.yaml"]))
    ):
        raise ValueError("one consistent unadmitted controller/navigation source required")
    keys = {
        "schema_version",
        "source",
        "approved",
        "evidence_domain",
        "seed",
        "spawn_z_m",
        "joint_state_topic",
        "clock_gz_topic",
    }
    if type(declaration) is not dict or set(declaration) != keys:
        raise ValueError("closed explicit owned SIM bootstrap declaration required")
    if (
        declaration["schema_version"] != "rosclaw.generic_bootstrap_source.v1"
        or declaration["source"] != "simulator_operator_fixture_policy"
        or declaration["approved"] is not True
        or declaration["evidence_domain"] != "SIMULATION"
        or type(declaration["spawn_z_m"]) not in (int, float)
        or not math.isfinite(declaration["spawn_z_m"])
        or not 0 <= declaration["spawn_z_m"] <= 2
    ):
        raise ValueError("explicit bounded SIM-only bootstrap source required")
    validate_seed(declaration["seed"])
    if declaration["seed"] is None:
        raise ValueError("bootstrap requires a frozen seed")
    joint_topic = absolute_endpoint(declaration["joint_state_topic"])
    clock_topic = absolute_endpoint(declaration["clock_gz_topic"])
    if clock_topic != "/world/" + manifest["world_name"] + "/clock":
        raise ValueError("clock must bind the exact declared World source")
    if joint_topic in {
        "/clock",
        *controller_report["topics"].values(),
        *navigation_report["topics"].values(),
    }:
        raise ValueError("joint-state topic must not alias another runtime role")
    manager = absolute_endpoint(controller["controller_manager"])
    namespace = manager.rsplit("/", 1)[0] or "/"
    spawners = []
    for role in ("joint_state_broadcaster", "drive_controller"):
        full = absolute_endpoint(controller[role])
        if full.rsplit("/", 1)[0] != manager.rsplit("/", 1)[0]:
            raise ValueError("controller identity must match its exact manager namespace")
        spawners.append(
            {
                "role": role,
                "controller_name": full.rsplit("/", 1)[1],
                "manager": manager,
                "namespace": namespace,
                "inactive": True,
            }
        )
    spawn = declarations["navigation"]["spawn_xyyaw"]
    return {
        "schema_version": "rosclaw.generic_bootstrap_launch_plan.v1",
        "workspace_manifest_sha256": captured["manifest_sha256"],
        "bootstrap_declaration_hash": digest(declaration),
        "execution_workspace": "/evidence",
        "world_name": manifest["world_name"],
        "body_model_name": manifest["body_model_name"],
        "gazebo_arguments": gazebo_arguments(Path("/evidence/world.sdf"), declaration["seed"]),
        "spawn_xyz_yaw": [spawn[0], spawn[1], declaration["spawn_z_m"], spawn[2]],
        "robot_description": files["robot.urdf"].decode("utf-8"),
        "robot_description_sha256": hashlib.sha256(files["robot.urdf"]).hexdigest(),
        "robot_description_topic": absolute_endpoint(controller["robot_description_topic"]),
        "joint_state_topic": joint_topic,
        "clock_gz_topic": clock_topic,
        "controller_namespace": namespace,
        "controller_spawners": spawners,
        "navigation": navigation,
        "requires_owned_deadline_supervisor": True,
        "requires_read_only_source_mount": True,
        "requires_fresh_Graph_TF_map_sensor_and_independent_physics_admission": True,
        "task_dispatch_path": "Native_MCP_rosclawd_ONLY",
        "live_admission": False,
        "authorization": False,
        "physical_acceptance": "NOT_RUN",
        "heldout_asset": "NOT_SELECTED",
    }


def required_process_exit_handler(action, role):
    """Any required process exit terminates this owned bootstrap, including 0."""
    from launch.actions import EmitEvent, RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown

    return RegisterEventHandler(
        OnProcessExit(
            target_action=action,
            on_exit=lambda event, context: [
                EmitEvent(
                    event=Shutdown(
                        reason=f"owned bootstrap required {role} exited:{event.returncode}"
                    )
                )
            ],
        )
    )


def build_bootstrap_launch_description(directory, declaration, *, readiness_output=None):
    """Build SDK actions only. Never invoke LaunchService from this module."""
    if Path(directory) != Path("/evidence"):
        raise ValueError("bootstrap source must be mounted at its declared /evidence path")
    if readiness_output is not None:
        readiness_output = Path(readiness_output)
        if not readiness_output.is_absolute() or readiness_output.resolve().is_relative_to(
            Path("/evidence")
        ):
            raise ValueError("readiness evidence must be outside the immutable source mount")
    plan = bootstrap_launch_plan(directory, declaration)
    from launch import LaunchDescription
    from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch_ros.actions import Node

    world = ExecuteProcess(cmd=plan["gazebo_arguments"], output="screen")
    publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="generic_source_state_publisher",
        namespace=plan["controller_namespace"],
        parameters=[{"use_sim_time": True, "robot_description": plan["robot_description"]}],
        remappings=[
            ("robot_description", plan["robot_description_topic"]),
            ("joint_states", plan["joint_state_topic"]),
        ],
        output="screen",
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="generic_source_bridge",
        parameters=[{"config_file": "/evidence/bridge.yaml"}],
        output="screen",
    )
    clock = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="generic_source_clock",
        arguments=[plan["clock_gz_topic"] + "@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        remappings=[(plan["clock_gz_topic"], "/clock")],
        output="screen",
    )
    x, y, z, yaw = plan["spawn_xyz_yaw"]
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        name="generic_source_spawn",
        arguments=[
            "-world",
            plan["world_name"],
            "-file",
            "/evidence/robot.sdf",
            "-name",
            plan["body_model_name"],
            "-x",
            str(x),
            "-y",
            str(y),
            "-z",
            str(z),
            "-Y",
            str(yaw),
        ],
        output="screen",
    )
    controllers = [
        Node(
            package="controller_manager",
            executable="spawner",
            name="generic_source_" + row["role"],
            namespace=row["namespace"],
            arguments=[
                row["controller_name"],
                "-c",
                row["manager"],
                "--inactive",
                "--controller-manager-timeout",
                "30",
            ],
            output="screen",
        )
        for row in plan["controller_spawners"]
    ]

    def after_spawn(event, context):
        if event.returncode != 0:
            return [EmitEvent(event=Shutdown(reason="owned generic SIM robot spawn failed"))]
        return controllers

    controller_exits = {}
    probe = (
        ExecuteProcess(
            cmd=[
                sys.executable,
                str(Path(__file__).with_name("generic_controller_readiness.py")),
                "--output",
                str(readiness_output),
            ],
            output="screen",
        )
        if readiness_output is not None
        else None
    )

    def after_controller(role, event, context):
        if role in controller_exits or event.returncode != 0:
            return [
                EmitEvent(
                    event=Shutdown(reason="owned inactive controller spawn failed or repeated")
                )
            ]
        controller_exits[role] = event.returncode
        return [probe] if len(controller_exits) == len(controllers) and probe is not None else []

    controller_handlers = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=controller,
                on_exit=lambda event, context, role=role: after_controller(role, event, context),
            )
        )
        for role, controller in enumerate(controllers)
    ]
    if probe is not None:
        controller_handlers.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=probe,
                    on_exit=lambda event, context: (
                        []
                        if event.returncode == 0
                        else [
                            EmitEvent(
                                event=Shutdown(
                                    reason="owned inactive controller source probe refused"
                                )
                            )
                        ]
                    ),
                )
            )
        )

    navigation = list(build_launch_description(directory).entities)
    required = [
        (world, "World"),
        (publisher, "URDF publisher"),
        (bridge, "sensor bridge"),
        (clock, "clock bridge"),
    ]
    required.extend(
        (action, f"navigation process {index}") for index, action in enumerate(navigation)
    )
    return LaunchDescription(
        [
            # Register before starting any required child, including fast failures.
            *(required_process_exit_handler(action, role) for action, role in required),
            world,
            publisher,
            bridge,
            clock,
            RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=after_spawn)),
            spawn,
            *controller_handlers,
            *navigation,
        ]
    )
