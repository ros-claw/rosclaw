"""Launch a small real Gazebo/Nav2/opennav simulation acceptance stack."""

import argparse
import json
import os
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from profiles import PROFILES

from experiments import (
    INNER_RING_PROFILES,
    controller_parameters,
    gazebo_arguments,
    planning_parameters,
    validate_seed,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = Path("/evidence")


def prepare(
    controller_watchdog=True,
    profile_name="waffle",
    coverage_preset="baseline",
    seed=None,
    *,
    precise_repair_waypoints=False,
):
    profile = PROFILES[profile_name]
    candidate = planning_parameters(profile, coverage_preset)
    controller_candidate = controller_parameters(profile, coverage_preset)
    validate_seed(seed)
    if profile_name == "burger" and not controller_watchdog:
        raise ValueError("Burger acceptance requires the bottom-level controller watchdog")
    OUTPUT.mkdir(exist_ok=True)
    sim = Path(get_package_share_directory("nav2_minimal_tb3_sim"))
    (OUTPUT / "robot.urdf").write_bytes((sim / "urdf/turtlebot3_waffle.urdf").read_bytes())
    os.environ["GZ_SIM_RESOURCE_PATH"] = f"{sim / 'models'}:{sim.parent}:" + os.getenv(
        "GZ_SIM_RESOURCE_PATH", ""
    )
    robot = subprocess.check_output(
        ["xacro", str(sim / "urdf/gz_waffle.sdf.xacro"), "namespace:="], text=True
    )
    tree = ET.fromstring(robot)
    model = tree.find("model")
    if controller_watchdog:
        from controller import configure

        configure(model, OUTPUT)
    contact_topics = []
    for link in model.findall("link"):
        for sensor in link.findall("sensor"):
            if sensor.get("type") in {"camera", "depth", "depth_camera", "rgbd_camera"}:
                link.remove(sensor)
            elif sensor.get("type") == "gpu_lidar":
                sensor.find("update_rate").text = "20"
        collisions = link.findall("collision")
        if collisions:
            topic = "/rosclaw_sim/contacts/" + link.get("name")
            contact_topics.append(topic)
            sensor = ET.SubElement(link, "sensor", name="expert_contacts", type="contact")
            ET.SubElement(sensor, "always_on").text = "true"
            ET.SubElement(sensor, "update_rate").text = "20"
            ET.SubElement(sensor, "topic").text = topic
            contact = ET.SubElement(sensor, "contact")
            for collision in collisions:
                ET.SubElement(contact, "collision").text = collision.get("name")
    (OUTPUT / "contact_topics.json").write_text(json.dumps(contact_topics))
    plugin = ET.SubElement(
        model,
        "plugin",
        filename="gz-sim-pose-publisher-system",
        name="gz::sim::systems::PosePublisher",
    )
    for name, value in {
        "publish_model_pose": "true",
        "publish_link_pose": "false",
        "use_pose_vector_msg": "true",
        "update_frequency": "30",
        "topic": "/rosclaw_sim/ground_truth",
    }.items():
        ET.SubElement(plugin, name).text = value
    ET.ElementTree(tree).write(OUTPUT / "robot.sdf")
    world = """<sdf version="1.9"><world name="ros_expert">
      <physics name="physics" type="ignored"><max_step_size>0.01</max_step_size><real_time_factor>1</real_time_factor></physics>
      <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
      <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
      <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
      <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
      <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
      <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
      <light name="sun" type="directional"><pose>0 0 10 0 0 0</pose><direction>-0.5 0.1 -0.9</direction></light>
      <model name="floor"><static>true</static><link name="ground"><collision name="ground"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry></collision><visual name="ground"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry></visual></link></model>"""
    for name, pose, size in [
        ("east", "1.55 0 0.3 0 0 0", "0.1 3.2 0.6"),
        ("west", "-1.55 0 0.3 0 0 0", "0.1 3.2 0.6"),
        ("north", "0 1.55 0.3 0 0 0", "3.2 0.1 0.6"),
        ("south", "0 -1.55 0.3 0 0 0", "3.2 0.1 0.6"),
    ]:
        world += f'<model name="{name}"><static>true</static><pose>{pose}</pose><link name="wall"><collision name="wall"><geometry><box><size>{size}</size></box></geometry></collision><visual name="wall"><geometry><box><size>{size}</size></box></geometry></visual></link></model>'
    (OUTPUT / "world.sdf").write_text(world + "</world></sdf>")
    width = 64
    values = [
        0 if abs((x + 0.5) * 0.05 - 1.6) >= 1.5 or abs((y + 0.5) * 0.05 - 1.6) >= 1.5 else 254
        for y in range(width)
        for x in range(width)
    ]
    (OUTPUT / "room.pgm").write_bytes(b"P5\n64 64\n255\n" + bytes(values))
    (OUTPUT / "map.yaml").write_text(
        yaml.safe_dump(
            {
                "image": "room.pgm",
                "resolution": 0.05,
                "origin": [-1.6, -1.6, 0],
                "negate": 0,
                "occupied_thresh": 0.65,
                "free_thresh": 0.25,
            }
        )
    )
    params = yaml.safe_load(
        (Path(get_package_share_directory("nav2_bringup")) / "params/nav2_params.yaml").read_text()
    )
    demo = yaml.safe_load(
        Path("/ws/src/opennav_coverage/opennav_coverage_demo/params/demo_params.yaml").read_text()
    )
    params["coverage_server"] = demo["coverage_server"]
    params["collision_monitor"]["ros__parameters"].update(
        cmd_vel_in_topic="/cmd_vel_smoothed", cmd_vel_out_topic="/nav_cmd_vel"
    )
    params["collision_monitor"]["ros__parameters"]["scan"]["topic"] = "/scan"
    params["coverage_server"]["ros__parameters"].update(
        robot_width=0.5, operation_width=0.45, default_headland_width=0.5, min_turning_radius=0.1
    )
    params["controller_server"]["ros__parameters"].update(
        demo["controller_server"]["ros__parameters"]
    )
    params["controller_server"]["ros__parameters"]["FollowPath"].update(
        desired_linear_vel=0.2,
        lookahead_dist=0.1,
        min_lookahead_dist=0.05,
        max_lookahead_dist=0.15,
        regulated_linear_scaling_min_speed=0.03,
        regulated_linear_scaling_min_radius=0.3,
        use_rotate_to_heading=True,
        rotate_to_heading_min_angle=0.35,
        use_velocity_scaled_lookahead_dist=False,
        min_approach_linear_velocity=0.02,
        approach_velocity_scaling_dist=0.15,
    )
    params["controller_server"]["ros__parameters"]["progress_checker"].update(
        required_movement_radius=0.05, movement_time_allowance=30.0
    )
    params["controller_server"]["ros__parameters"]["general_goal_checker"].update(
        xy_goal_tolerance=0.025, yaw_goal_tolerance=0.1
    )
    params["bt_navigator"]["ros__parameters"].update(
        navigators=["navigate_to_pose", "navigate_through_poses", "navigate_complete_coverage"],
        navigate_complete_coverage={"plugin": "opennav_coverage_navigator/CoverageNavigator"},
        plugin_lib_names=demo["bt_navigator"]["ros__parameters"]["plugin_lib_names"],
        default_coverage_bt_xml=get_package_share_directory("opennav_coverage_bt")
        + "/behavior_trees/navigate_w_basic_complete_coverage_nav_to_start.xml",
    )
    if precise_repair_waypoints:
        from precise_through_poses_bt import prepare_precise_through_poses_bt

        original = (
            Path(get_package_share_directory("nav2_bt_navigator"))
            / "behavior_trees/navigate_through_poses_w_replanning_and_recovery.xml"
        ).read_bytes()
        prepare_precise_through_poses_bt(
            OUTPUT,
            original,
            xy_goal_tolerance=params["controller_server"]["ros__parameters"][
                "general_goal_checker"
            ]["xy_goal_tolerance"],
        )
        params["bt_navigator"]["ros__parameters"]["default_nav_through_poses_bt_xml"] = str(
            OUTPUT / "repair-through-poses.xml"
        )
    params["amcl"]["ros__parameters"].update(
        set_initial_pose=True,
        initial_pose={"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        transform_tolerance=0.3,
        update_min_d=0.02,
        update_min_a=0.02,
        alpha1=0.2,
        alpha2=0.2,
        alpha3=0.2,
        alpha4=0.2,
        sigma_hit=0.05,
        max_beams=180,
    )
    for costmap in ["global_costmap", "local_costmap"]:
        p = params[costmap][costmap]["ros__parameters"]
        p.update(robot_radius=0.25, resolution=0.05, publish_frequency=2.0)
        p["inflation_layer"].update(inflation_radius=0.27, cost_scaling_factor=5.0)
    (OUTPUT / "nav2.yaml").write_text(yaml.safe_dump(params))
    bridge = yaml.safe_load((sim / "configs/turtlebot3_waffle_bridge.yaml").read_text())
    if controller_watchdog:
        bridge = [
            entry
            for entry in bridge
            if entry.get("topic_name") not in {"cmd_vel", "odom", "tf", "joint_states"}
        ]
    bridge.extend(
        {
            "ros_topic_name": t,
            "gz_topic_name": "/world/ros_expert/model/turtlebot3_waffle/link/"
            + t.rsplit("/", 1)[-1]
            + "/sensor/expert_contacts/contact",
            "ros_type_name": "ros_gz_interfaces/msg/Contacts",
            "gz_type_name": "gz.msgs.Contacts",
            "direction": "GZ_TO_ROS",
        }
        for t in contact_topics
    )
    bridge.append(
        {
            "ros_topic_name": "/rosclaw_sim/ground_truth",
            "gz_topic_name": "/rosclaw_sim/ground_truth",
            "ros_type_name": "tf2_msgs/msg/TFMessage",
            "gz_type_name": "gz.msgs.Pose_V",
            "direction": "GZ_TO_ROS",
        }
    )
    (OUTPUT / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    if profile_name == "burger":
        from burger import prepare as prepare_burger

        prepare_burger(OUTPUT, profile)
    # Apply after profile-specific generation: Burger's original headland is
    # 0.3 m, not the Waffle default. Candidate settings never alter its geometry.
    params = yaml.safe_load((OUTPUT / "nav2.yaml").read_text())
    params["coverage_server"]["ros__parameters"].update(candidate)
    params["controller_server"]["ros__parameters"]["FollowPath"].update(controller_candidate)
    (OUTPUT / "nav2.yaml").write_text(yaml.safe_dump(params))
    (OUTPUT / "experiment.json").write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.sim_coverage_experiment.v1",
                "evidence_role": "fixture_configuration_not_measured_success",
                "profile": profile_name,
                "preset": coverage_preset,
                "boundary_pass": coverage_preset
                in (
                    "perimeter",
                    "perimeter_sequential",
                    "perimeter_stateless",
                    "perimeter_stateless_headland",
                    "perimeter_stateless_clearance",
                    "perimeter_stateless_clearance_inner_ring",
                    "perimeter_stateless_inner_ring",
                    "perimeter_stateless_overlap",
                ),
                "boundary_strategy": "sequential_inner_ring"
                if coverage_preset in INNER_RING_PROFILES
                else "sequential"
                if coverage_preset
                in (
                    "perimeter_sequential",
                    "perimeter_stateless",
                    "perimeter_stateless_headland",
                    "perimeter_stateless_clearance",
                    "perimeter_stateless_clearance_inner_ring",
                    "perimeter_stateless_overlap",
                )
                else "through_poses",
                **(
                    {"boundary_stage_budget_sec": 360, "inner_boundary_inset_cells": 1}
                    if coverage_preset in INNER_RING_PROFILES
                    else {}
                ),
                "seed": seed,
                "planning_parameters": candidate,
                "controller_parameters": controller_candidate,
                "start_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "gazebo_arguments": gazebo_arguments(OUTPUT / "world.sdf", seed),
            },
            indent=2,
        )
        + "\n"
    )
    bridge = yaml.safe_load((OUTPUT / "bridge.yaml").read_text())
    truth = [item for item in bridge if item.get("ros_topic_name") == "/rosclaw_sim/ground_truth"]
    for item in truth:
        item.update(publisher_queue=1, subscriber_queue=1, qos_profile="SENSOR_DATA")
    (OUTPUT / "truth_bridge.yaml").write_text(yaml.safe_dump(truth))
    (OUTPUT / "bridge.yaml").write_text(
        yaml.safe_dump([item for item in bridge if item not in truth])
    )
    (OUTPUT / "fixture_profile.json").write_text(json.dumps(profile.to_dict(), indent=2) + "\n")
    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="waffle")
    parser.add_argument(
        "--coverage-preset",
        choices=[
            "baseline",
            "diagonal",
            "headland",
            "perimeter",
            "perimeter_sequential",
            "perimeter_stateless",
            "perimeter_stateless_headland",
            "perimeter_stateless_clearance",
            "perimeter_stateless_clearance_inner_ring",
            "perimeter_stateless_inner_ring",
            "perimeter_stateless_overlap",
        ],
        default="baseline",
    )
    parser.add_argument("--precise-repair-waypoints", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fault-acceptance", action="store_true")
    parser.add_argument(
        "--controller-watchdog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the bottom-level command timeout; disable only for explicit legacy fixture replay",
    )
    args = parser.parse_args()
    from fixture_network import validate_ros_domain

    # Read this process's actual kernel range before any simulator/ROS child.
    # The host preflight alone cannot establish a container's network facts.
    network = validate_ros_domain(int(os.environ.get("ROS_DOMAIN_ID", "0")))
    network["source_process_network_namespace"] = os.readlink("/proc/self/ns/net")
    network["source"] = "owned_stack_process_actual_kernel_port_range"
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "container-network-preflight.json").open("x") as evidence:
        json.dump(network, evidence, indent=2)
    from fixture_middleware import configure_service_reply_discovery

    os.environ.update(configure_service_reply_discovery(ROOT, OUTPUT, os.environ))
    os.environ["PYTHONPATH"] = (
        str(ROOT.parents[2] / "src") + os.pathsep + os.getenv("PYTHONPATH", "")
    )
    prepare(
        args.controller_watchdog,
        args.profile,
        args.coverage_preset,
        args.seed,
        precise_repair_waypoints=args.precise_repair_waypoints,
    )
    (OUTPUT / "run_id.txt").write_text(uuid.uuid4().hex + "\n")
    profile = PROFILES[args.profile]
    children = []
    labels = {}
    exited = set()

    def interrupt_stack(_signum, _frame):
        # Docker stop sends SIGTERM. Preserve the same fail-closed cleanup as
        # interactive SIGINT, including child observer evidence flushing.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_stack)

    def start(name, argv):
        log = (OUTPUT / f"{name}.log").open("w")
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        children.append(p)
        labels[p] = name
        return p

    try:
        start("gazebo", gazebo_arguments(OUTPUT / "world.sdf", args.seed))
        time.sleep(3)
        subprocess.run(
            [
                "ros2",
                "run",
                "ros_gz_sim",
                "create",
                "-world",
                "ros_expert",
                "-file",
                str(OUTPUT / "robot.sdf"),
                "-name",
                profile.simulation_model,
                "-z",
                "0.05",
            ],
            check=True,
            timeout=30,
        )
        start(
            "bridge",
            [
                "ros2",
                "run",
                "ros_gz_bridge",
                "parameter_bridge",
                "--ros-args",
                "-p",
                f"config_file:={OUTPUT / 'bridge.yaml'}",
                "-p",
                "use_sim_time:=true",
            ],
        )
        start(
            "robot_state",
            [
                "ros2",
                "run",
                "robot_state_publisher",
                "robot_state_publisher",
                "--ros-args",
                "-p",
                "use_sim_time:=true",
                "-p",
                "robot_description:=" + (OUTPUT / "robot.urdf").read_text(),
            ],
        )
        if args.controller_watchdog:
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
                timeout=40,
            )
        start("nav2", ["ros2", "launch", str(ROOT / "nav2_launch.py")])
        start(
            "truth_bridge",
            [
                "ros2",
                "run",
                "ros_gz_bridge",
                "parameter_bridge",
                "--ros-args",
                "-p",
                f"config_file:={OUTPUT / 'truth_bridge.yaml'}",
                "-p",
                "use_sim_time:=true",
            ],
        )
        start(
            "coverage",
            [
                "ros2",
                "run",
                "opennav_coverage",
                "opennav_coverage",
                "--ros-args",
                "--params-file",
                str(OUTPUT / "nav2.yaml"),
                "-p",
                "use_sim_time:=true",
            ],
        )
        start(
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
        start(
            "rosbridge",
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
        start(
            "rosapi",
            ["ros2", "run", "rosapi", "rosapi_node", "--ros-args", "-p", "use_sim_time:=true"],
        )
        start(
            "witness",
            [
                "python3",
                str(ROOT / "witness.py"),
                "--ros-args",
                "-p",
                "use_sim_time:=true",
                "-p",
                f"controller_watchdog:={'true' if args.controller_watchdog else 'false'}",
            ],
        )
        start("lifecycle_probe", ["python3", str(ROOT / "lifecycle_readiness.py")])
        start("probe", ["python3", str(ROOT.parent / "ros2/probe.py")])
        while True:
            for p in children:
                if p.poll() is not None:
                    if args.fault_acceptance and labels[p] in {"rosbridge", "rosapi", "probe"}:
                        if p not in exited:
                            exited.add(p)
                            with (OUTPUT / "fixture_process_failures.jsonl").open("a") as trace:
                                trace.write(
                                    json.dumps({"process": labels[p], "exit_code": p.returncode})
                                    + "\n"
                                )
                        continue
                    raise RuntimeError(f"stack process exited: {p.args}")
            time.sleep(1)
    finally:
        for p in children:
            p.send_signal(signal.SIGINT)
        for p in children:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
