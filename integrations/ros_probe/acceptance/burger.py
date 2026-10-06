"""Build the known Burger fixture from actual installed vendor geometry and sensors.

Shared room files are prepared by stack.py. No Waffle collision/inertial model
or sensor is substituted. The cleaner is an explicit simulation attachment.
"""

import copy
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from controller import configure


def prepare(output, profile):
    description = Path(get_package_share_directory("turtlebot3_description"))
    gazebo = Path(get_package_share_directory("turtlebot3_gazebo"))
    raw = subprocess.check_output(["xacro", str(description / "urdf/turtlebot3_burger.urdf")])
    (output / "burger.vendor.urdf").write_bytes(raw)
    tree = ET.fromstring(
        subprocess.check_output(["gz", "sdf", "-p", str(output / "burger.vendor.urdf")])
    )
    model = tree.find("model")
    model.set("name", profile.simulation_model)
    body = model.find("link[@name='base_footprint']")
    for link_name, sensor_type, pose in (
        ("base_scan", "gpu_lidar", "-0.032 0 0.182 0 0 0"),
        ("imu_link", "imu", "-0.032 0 0.078 0 0 0"),
    ):
        vendor = ET.parse(gazebo / "models/turtlebot3_burger/model.sdf").getroot().find("model")
        sensor = copy.deepcopy(
            vendor.find(f"link[@name='{link_name}']/sensor[@type='{sensor_type}']")
        )
        if sensor_type == "gpu_lidar":
            sensor.find("update_rate").text = "20"
        element = sensor.find("pose")
        if element is None:
            element = ET.SubElement(sensor, "pose")
        element.text = pose
        body.append(sensor)
    topics = []
    for link in model.findall("link"):
        collisions = link.findall("collision")
        if collisions:
            topic = "/rosclaw_sim/contacts/" + link.get("name")
            topics.append(topic)
            sensor = ET.SubElement(link, "sensor", name="expert_contacts", type="contact")
            ET.SubElement(sensor, "always_on").text = "true"
            ET.SubElement(sensor, "update_rate").text = "20"
            ET.SubElement(sensor, "topic").text = topic
            contact = ET.SubElement(sensor, "contact")
            for collision in collisions:
                ET.SubElement(contact, "collision").text = collision.get("name")
    (output / "contact_topics.json").write_text(json.dumps(topics))
    diff = copy.deepcopy(vendor.find("plugin[@name='gz::sim::systems::DiffDrive']"))
    if diff is None:
        raise ValueError("vendor Burger differential drive geometry is required")
    model.append(diff)
    (output / "robot.urdf").write_bytes(raw)
    configure(model, output)
    pose_plugin = ET.SubElement(
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
        ET.SubElement(pose_plugin, name).text = value
    ET.ElementTree(tree).write(output / "robot.sdf")
    nav = yaml.safe_load((output / "nav2.yaml").read_text())
    nav["coverage_server"]["ros__parameters"].update(
        robot_width=profile.coverage_width_m,
        operation_width=profile.operation_width_m,
        default_headland_width=profile.coverage_width_m,
    )
    for name in ("global_costmap", "local_costmap"):
        params = nav[name][name]["ros__parameters"]
        params["robot_radius"] = profile.physical_radius_m
        params["inflation_layer"]["inflation_radius"] = profile.physical_radius_m + 0.02
    (output / "nav2.yaml").write_text(yaml.safe_dump(nav))
    bridge = yaml.safe_load((output / "bridge.yaml").read_text())
    bridge = [
        item
        for item in bridge
        if not item.get("ros_topic_name", "").startswith("/rosclaw_sim/contacts/")
    ]
    bridge.extend(
        {
            "ros_topic_name": topic,
            "gz_topic_name": f"/world/ros_expert/model/{profile.simulation_model}/link/"
            + topic.rsplit("/", 1)[-1]
            + "/sensor/expert_contacts/contact",
            "ros_type_name": "ros_gz_interfaces/msg/Contacts",
            "gz_type_name": "gz.msgs.Contacts",
            "direction": "GZ_TO_ROS",
        }
        for topic in topics
    )
    (output / "bridge.yaml").write_text(yaml.safe_dump(bridge))
