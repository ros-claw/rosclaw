"""Pure contracts for fallible read-only ROS graph collection."""


def collect_node_actions(read_nodes, read_actions):
    errors = []

    def record(operation, error, node=None):
        errors.append(
            {
                "operation": operation,
                "node": node,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    try:
        names = sorted(read_nodes())
    except RuntimeError as error:
        record("get_node_names_and_namespaces", error)
        return [], [], errors
    nodes = [{"name": ns.rstrip("/") + "/" + name, "namespace": ns} for name, ns in names]
    actions = {}
    for name, namespace in names:
        try:
            observed = read_actions(name, namespace)
        except RuntimeError as error:
            record(
                "get_action_server_names_and_types_by_node",
                error,
                namespace.rstrip("/") + "/" + name,
            )
            continue
        for action, types in observed:
            actions[action] = {"name": action, "action_type": types[0] if len(types) == 1 else ""}
    return nodes, list(actions.values()), errors


def unknown_graph_snapshot(stamp, environment, error):
    """An actual failed read creates UNKNOWN, never a refreshed cached snapshot."""
    return {
        "schema_version": "rosclaw.ros_probe.v1",
        "captured_at": stamp,
        "environment": environment,
        "graph": {"topics": [], "nodes": [], "actions": [], "services": []},
        "signals": [],
        "transforms": [],
        "lifecycle": [],
        "qos": {"supported": False, "endpoints": [], "incompatible_pairs": []},
        "navigation": {},
        "observations": {
            "graph_read_errors": [
                {
                    "operation": "discover_or_snapshot",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            ]
        },
        "completeness": dict.fromkeys(
            ["graph", "qos", "signals", "tf", "lifecycle", "time"], False
        ),
        "errors": ["read-only graph collection failed: " + str(error)],
    }
