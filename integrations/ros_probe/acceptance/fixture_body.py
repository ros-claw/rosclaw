"""Create the fixture's static Body through the existing Body compiler.

Vendor URDF is supplied by the pinned Nav2 simulation package. Runtime ROS
observations and task area do not overwrite the physical model. The cleaner is
explicitly a simulation attachment; this template grants no REAL authority.
"""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from rosclaw.body.compiler import compute_checksum
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import BodyYaml, CalibrationYaml, EurdfProfile


def configure_fixture_body(home: Path, specification: dict, urdf_path: Path):
    raw = urdf_path.read_bytes()
    robot = ET.fromstring(raw)
    if robot.tag != "robot" or robot.find("link[@name='base_footprint']") is None:
        raise ValueError("the actual fixture vendor URDF is required")
    joints = []
    for joint in robot.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "limits": {k: float(v) for k, v in limit.attrib.items()}
                if limit is not None
                else {},
            }
        )
    profile = EurdfProfile(
        profile_id="turtlebot3_waffle_gazebo",
        profile_version="1.0.0",
        vendor="ROBOTIS / Nav2 minimal simulation",
        model=robot.get("name"),
        display_name="TurtleBot3 Waffle Gazebo acceptance fixture",
        description="Pinned vendor model with an explicitly simulated cleaning attachment.",
        assets={"urdf": "refs/robot.urdf"},
        identity={"robot_class": "mobile_base"},
        frames={"root": "base_footprint", "base": "base_link", "map": "map", "odom": "odom"},
        joints=joints,
        sensors=[
            {"name": "lidar", "type": "lidar", "parent_link": "base_scan"},
            {"name": "imu", "type": "imu", "parent_link": "imu_link"},
        ],
        actuators=[{"name": j["name"], "type": "wheel_motor", "joint": j["name"]} for j in joints],
        provider_interfaces={
            "ros_capability_bindings": {
                semantic: {
                    "name": "/rosclaw_sim/cleaning",
                    "srv_type": "std_srvs/srv/SetBool",
                    "data": enabled,
                    "state_topic": "/rosclaw_sim/cleaning_state",
                    "state_type": "std_msgs/msg/Bool",
                }
                for semantic, enabled in [("cleaning.enable", True), ("cleaning.disable", False)]
            }
        },
        capability_hints={
            "all": [
                "navigation.navigate_to_pose",
                "coverage.execute",
                "localization.set_initial_pose",
            ]
        },
        safety={"safety_level": "STRICT", "environment": {"real_robot_execution_allowed": False}},
        sandbox={"compatible_engines": ["gazebo"], "preferred_engine": "gazebo"},
        metadata={
            "evidence_domain": "SIMULATION",
            "urdf_sha256": hashlib.sha256(raw).hexdigest(),
            "physical_radius_m": specification["physical_radius_m"],
            "cleaning_polygon": specification["cleaning_polygon"],
            "cleaner_kind": "simulated_attachment",
        },
    )
    profile.provider_interfaces["ros_capability_bindings"].update(
        {
            "coverage.verify": {
                "name": "rosclaw.coverage_verifier.v1",
                "cleaning_polygon": specification["cleaning_polygon"],
            },
            "safety.collision_monitor": {
                "name": "/collision_monitor",
                "input_topic": "/cmd_vel_smoothed",
                "output_topic": "/nav_cmd_vel",
                "sensor_topic": "/scan",
            },
        }
    )
    resolver = BodyResolver(workspace=home)
    resolver.ensure_body_dir()
    resolver.eurdf_profile_path.write_text(yaml.safe_dump(profile.to_dict(), sort_keys=False))
    resolver.eurdf_profile_path.with_name("robot.urdf").write_bytes(raw)
    uri = "rosclaw://eurdf/turtlebot3_waffle_gazebo@1.0.0"
    checksum = compute_checksum(resolver.eurdf_profile_path)
    resolver.eurdf_lock_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "rosclaw.eurdf_lock.v1",
                "profile_id": profile.profile_id,
                "profile_version": profile.profile_version,
                "uri": uri,
                "source": "pinned_vendor_simulation",
                "checksum": checksum,
            }
        )
    )
    body = BodyYaml(
        body_instance={"id": specification["body_id"], "robot_model": profile.profile_id},
        model_ref={"eurdf_uri": uri, "checksum": checksum},
        installed_components={
            "sensors": {
                s["name"]: {"installed": True, "status": "available"} for s in profile.sensors
            },
            "actuators": {
                a["name"]: {"installed": True, "status": "available"} for a in profile.actuators
            },
        },
        agent_policy={"direct_real_robot_execution_allowed": False},
        metadata={"evidence_domain": "SIMULATION", "source": "acceptance_fixture_template"},
    )
    resolver.body_yaml_path.write_text(yaml.safe_dump(body.to_dict(), sort_keys=False))
    resolver.calibration_yaml_path.write_text(yaml.safe_dump(CalibrationYaml().to_dict()))
    effective = resolver.recompile_effective_body()
    return effective.compute_hash()
