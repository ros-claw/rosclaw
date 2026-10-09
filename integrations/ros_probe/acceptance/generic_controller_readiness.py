"""Read-only inactive-controller correspondence, never Body/action admission."""

import argparse
import json
import time
from pathlib import Path
from xml.etree import ElementTree as ET

from generic_bootstrap_runtime import require_isolated_environment, require_read_only_workspace
from generic_stack_source import read_prepared_generic_stack

from rosclaw.connectors.ros.context.sim_navigation_source import _yaml


def controller_expectations(directory):
    source = read_prepared_generic_stack(directory)
    files = source["captured_files"]
    report = json.loads(files["controller-source-report.json"])
    declaration = json.loads(files["source-declarations.json"])["controller"]
    parameters = _yaml(files["controller_params.yaml"])
    manager = declaration["controller_manager"]
    manager_parameters = parameters[manager]["ros__parameters"]
    controllers = {}
    for role in ("drive_controller", "joint_state_broadcaster"):
        name = declaration[role].rsplit("/", 1)[-1]
        controllers[name] = manager_parameters[name]["type"]
    urdf = ET.fromstring(files["robot.urdf"])
    controls = urdf.findall("ros2_control")
    if len(controls) != 1:
        raise ValueError("one exact sealed SIM hardware component required")
    control = controls[0]
    plugin = control.findtext("hardware/plugin")
    if not plugin or not control.get("name") or control.get("type") != "system":
        raise ValueError("exact sealed SIM hardware plugin/name/system required")
    interfaces, interface_types = {}, {}
    for kind in ("command", "state"):
        names = []
        types = {}
        for joint in control.findall("joint"):
            for interface in joint.findall(kind + "_interface"):
                name = joint.get("name") + "/" + interface.get("name")
                names.append(name)
                types[name] = interface.get("data_type", "double")
        if not names or len(names) != len(set(names)):
            raise ValueError("unique complete sealed hardware interfaces required")
        interfaces[kind] = sorted(names)
        interface_types[kind] = types
    if report["live_controller_admitted"] is not False:
        raise ValueError("source report cannot claim controller admission")
    return {
        "workspace_manifest_sha256": source["manifest_sha256"],
        "controller_manager": manager,
        "controllers": controllers,
        "hardware_name": control.get("name"),
        "hardware_type": control.get("type"),
        "hardware_plugin": plugin,
        "interfaces": interfaces,
        "interface_types": interface_types,
    }


def inactive_controller_readiness(expected, observations, *, now):
    """Compare original typed replies with source; all unknowns refuse."""
    reasons = []
    for role in ("controllers", "hardware"):
        row = observations.get(role)
        service = expected["controller_manager"] + (
            "/list_controllers" if role == "controllers" else "/list_hardware_components"
        )
        if (
            type(row) is not dict
            or row.get("service") != service
            or type(row.get("received_monotonic_sec")) not in (int, float)
            or not 0 <= now - row["received_monotonic_sec"] <= 2
            or type(row.get("response")) is not dict
        ):
            reasons.append(role + ":missing_or_stale_original_reply")
    if not reasons:
        controllers = observations["controllers"]["response"].get("controller")
        if (
            type(controllers) is not list
            or len(controllers) != len(expected["controllers"])
            or any(type(row) is not dict or type(row.get("name")) is not str for row in controllers)
            or {row.get("name") for row in controllers} != set(expected["controllers"])
        ):
            reasons.append("controller_inventory_differs")
        else:
            for row in controllers:
                if (
                    row.get("type") != expected["controllers"][row["name"]]
                    or row.get("state") != "inactive"
                    or row.get("claimed_interfaces") != []
                ):
                    reasons.append("controller_not_exact_unclaimed_inactive:" + row["name"])
        components = observations["hardware"]["response"].get("component")
        if type(components) is not list or len(components) != 1 or type(components[0]) is not dict:
            reasons.append("hardware_inventory_differs")
        else:
            component = components[0]
            state = component.get("state")
            if (
                component.get("name") != expected["hardware_name"]
                or component.get("type") != expected["hardware_type"]
                or component.get("plugin_name") != expected["hardware_plugin"]
                or type(state) is not dict
                or type(state.get("id")) is not int
                or state["id"] != 3
                or state.get("label") != "active"
            ):
                reasons.append("hardware_source_or_active_state_differs")
            for kind, names in expected["interfaces"].items():
                rows = component.get(kind + "_interfaces")
                if (
                    type(rows) is not list
                    or len(rows) != len(names)
                    or any(
                        type(row) is not dict or type(row.get("name")) is not str for row in rows
                    )
                    or sorted(row.get("name", "") for row in rows) != names
                    or any(
                        row.get("is_available") is not True
                        or row.get("data_type") != expected["interface_types"][kind][row["name"]]
                        or (kind == "command" and row.get("is_claimed") is not False)
                        for row in rows
                    )
                ):
                    reasons.append("hardware_" + kind + "_interfaces_differs_or_unavailable")
    return {
        "schema_version": "rosclaw.generic_inactive_controller_readiness.v1",
        "source": "actual_read_only_controller_manager_responses",
        "workspace_manifest_sha256": expected["workspace_manifest_sha256"],
        "ready_for_next_stage": not reasons,
        "refusals": reasons,
        "controller_activation": False,
        "live_body_admitted": False,
        "authorization": False,
        "physical_acceptance": "NOT_EVALUATED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_read_only_workspace("/evidence")
    require_isolated_environment()
    expected = controller_expectations("/evidence")
    deadline = time.monotonic() + 5
    import rclpy
    from controller_manager_msgs.srv import ListControllers, ListHardwareComponents
    from rosidl_runtime_py.convert import message_to_ordereddict

    rclpy.init()
    node = rclpy.create_node("generic_inactive_controller_source_probe")
    observations, pending, clients = {}, {}, {}
    try:
        for role, service_type, suffix in (
            ("controllers", ListControllers, "/list_controllers"),
            ("hardware", ListHardwareComponents, "/list_hardware_components"),
        ):
            service = expected["controller_manager"] + suffix
            clients[role] = (node.create_client(service_type, service), service_type, service)
        while time.monotonic() < deadline and len(observations) < len(clients):
            for role, (client, service_type, service) in clients.items():
                if role not in pending and client.service_is_ready():
                    pending[role] = client.call_async(service_type.Request())
                future = pending.get(role)
                if role not in observations and future is not None and future.done():
                    response = json.loads(json.dumps(message_to_ordereddict(future.result())))
                    if len(json.dumps(response)) > 1_000_000:
                        raise ValueError("bounded original controller manager reply required")
                    observations[role] = {
                        "service": service,
                        "received_monotonic_sec": time.monotonic(),
                        "response": response,
                    }
            rclpy.spin_once(node, timeout_sec=min(0.01, max(0, deadline - time.monotonic())))
        current = controller_expectations("/evidence")
        if current != expected:
            raise ValueError("sealed controller source changed during read-only probe")
        report = inactive_controller_readiness(expected, observations, now=time.monotonic())
        report["original_responses"] = observations
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
        return 0 if report["ready_for_next_stage"] else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
