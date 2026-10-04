"""Configure official ros2_control deadman in the disposable physics fixture."""

import xml.etree.ElementTree as ET

import yaml


def configure(model, output):
    diff = next(
        p for p in model.findall("plugin") if p.get("name") == "gz::sim::systems::DiffDrive"
    )
    left, right = diff.findtext("left_joint"), diff.findtext("right_joint")
    separation, radius = (
        float(diff.findtext("wheel_separation")),
        float(diff.findtext("wheel_radius")),
    )
    model.remove(diff)
    control = ET.Element("ros2_control", name="GazeboSimSystem", type="system")
    ET.SubElement(
        ET.SubElement(control, "hardware"), "plugin"
    ).text = "gz_ros2_control/GazeboSimSystem"
    for name in (left, right):
        joint = ET.SubElement(control, "joint", name=name)
        ET.SubElement(joint, "command_interface", name="velocity")
        ET.SubElement(joint, "state_interface", name="position")
        ET.SubElement(joint, "state_interface", name="velocity")
    model.append(control)
    urdf = ET.fromstring((output / "robot.urdf").read_text())
    urdf.append(ET.fromstring(ET.tostring(control)))
    ET.ElementTree(urdf).write(output / "robot.urdf")
    plugin = ET.SubElement(
        model,
        "plugin",
        filename="libgz_ros2_control-system.so",
        name="gz_ros2_control::GazeboSimROS2ControlPlugin",
    )
    ET.SubElement(plugin, "parameters").text = str(output / "controllers.yaml")
    ET.SubElement(ET.SubElement(plugin, "ros"), "remapping").text = "/drive_controller/odom:=/odom"
    (output / "controllers.yaml").write_text(
        yaml.safe_dump(
            {
                "controller_manager": {
                    "ros__parameters": {
                        "update_rate": 100,
                        "use_sim_time": True,
                        "joint_state_broadcaster": {
                            "type": "joint_state_broadcaster/JointStateBroadcaster"
                        },
                        "drive_controller": {"type": "diff_drive_controller/DiffDriveController"},
                    }
                },
                "drive_controller": {
                    "ros__parameters": {
                        "left_wheel_names": [left],
                        "right_wheel_names": [right],
                        "wheel_separation": separation,
                        "wheel_radius": radius,
                        "odom_frame_id": "odom",
                        "base_frame_id": "base_footprint",
                        "publish_rate": 50.0,
                        "enable_odom_tf": True,
                        "cmd_vel_timeout": 0.2,
                        "use_sim_time": True,
                    }
                },
            }
        )
    )
