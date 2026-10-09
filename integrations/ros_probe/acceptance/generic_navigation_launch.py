"""Unactivated SDK launch source from an integrity-checked generic workspace.

Constructing a launch description does not discover or admit a Body. Lifecycle
activation and guarded motion require the later independent runtime admission.
"""

import json
from pathlib import Path

from generic_stack_source import read_prepared_generic_stack

from rosclaw.connectors.ros.context.sim_navigation_source import NODE_ROLES, _yaml
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.mission.sim_endpoints import absolute_endpoint


def navigation_launch_plan(directory):
    captured = read_prepared_generic_stack(directory)
    files = captured["captured_files"]
    rows = json.loads(files["navigation-launch-source.json"])
    report = json.loads(files["navigation-source-report.json"])
    parameters = _yaml(files["nav2.yaml"])
    if (
        type(report) is not dict
        or report.get("schema_version") != "rosclaw.sim_navigation_source_candidate.v1"
        or report.get("authorization") is not False
        or report.get("physical_acceptance") != "NOT_RUN"
        or report.get("usable_for_real_execution") is not False
        or report.get("requires_actual_Graph_TF_Body_source_admission") is not True
        or digest(rows) != report.get("launch_spec_hash")
        or digest(parameters) != report.get("parameters_hash")
        or type(rows) is not list
        or len(rows) != len(NODE_ROLES)
        or type(report.get("nodes")) is not dict
        or set(report["nodes"]) != set(NODE_ROLES)
    ):
        raise ValueError("exact unadmitted navigation source report required")
    roles, names, namespaces = set(), [], set()
    for row in rows:
        if type(row) is not dict or set(row) != {
            "role",
            "package",
            "executable",
            "name",
            "namespace",
            "remappings",
        }:
            raise ValueError("closed navigation launch source row required")
        role = row["role"]
        if type(role) is not str or role not in NODE_ROLES or role in roles:
            raise ValueError("all unique explicit navigation roles required")
        roles.add(role)
        if (row["package"], row["executable"]) != NODE_ROLES[role][:2]:
            raise ValueError("installed standard navigation executables required")
        full_name = absolute_endpoint(row["namespace"].rstrip("/") + "/" + row["name"])
        if full_name != report["nodes"][role] or full_name not in parameters:
            raise ValueError("launch node and exact source parameter identity differ")
        if type(row["remappings"]) is not list or len(row["remappings"]) > 32:
            raise ValueError("bounded explicit navigation remappings required")
        for remap in row["remappings"]:
            if type(remap) is not list or len(remap) != 2:
                raise ValueError("closed explicit navigation remapping pair required")
            absolute_endpoint("/" + remap[0])
            absolute_endpoint(remap[1])
        namespaces.add(row["namespace"])
        names.append(full_name)
    if len(namespaces) != 1 or len(set(names)) != len(names):
        raise ValueError("one explicit namespace with distinct navigation nodes required")
    return {
        "schema_version": "rosclaw.generic_navigation_launch_plan.v1",
        "workspace_manifest_sha256": captured["manifest_sha256"],
        "nodes": rows,
        "parameters": parameters,
        "lifecycle_namespace": namespaces.pop(),
        "lifecycle_node_names": names,
        "lifecycle_autostart": False,
        "live_admission": False,
        "authorization": False,
        "physical_acceptance": "NOT_RUN",
    }


def build_launch_description(directory):
    """Build unexecuted SDK actions including child costmap file parameters.

    A future launcher must recheck and mount the sealed workspace read-only
    before execution. Construction grants no runtime source admission.
    """
    plan = navigation_launch_plan(directory)
    from launch import LaunchDescription
    from launch_ros.actions import Node

    nodes = [
        Node(
            package=row["package"],
            executable=row["executable"],
            name=row["name"],
            namespace=row["namespace"],
            output="screen",
            parameters=[str(Path(directory) / "nav2.yaml")],
            remappings=[tuple(pair) for pair in row["remappings"]],
        )
        for row in plan["nodes"]
    ]
    nodes.append(
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_generic_navigation",
            namespace=plan["lifecycle_namespace"],
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "autostart": False,
                    "node_names": plan["lifecycle_node_names"],
                }
            ],
        )
    )
    return LaunchDescription(nodes)
