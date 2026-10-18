"""Closed disposable-SIM launch clock overrides, never a live reconfiguration API."""

SERVER_NAMES = frozenset(
    {
        "map_server",
        "amcl",
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "velocity_smoother",
        "collision_monitor",
    }
)


def counterfactual_clock_parameters(descriptor, node_name):
    """Read a staged explicit override for one existing SDK Nav2 process.

    Default launch behavior is unchanged. The enclosing prepared SDK launch owns
    this file and the process; this function performs no ROS or filesystem I/O.
    A proposal does not imply that the actual node accepted this parameter.
    """
    stage = (descriptor or {}).get("counterfactual_sim")
    if stage is None:
        return {"use_sim_time": True}
    if (
        type(stage) is not dict
        or set(stage) != {"schema_version", "node_clock_overrides"}
        or stage["schema_version"] != "rosclaw.sim_counterfactual_launch.v1"
        or type(stage["node_clock_overrides"]) is not dict
        or not 1 <= len(stage["node_clock_overrides"]) <= 2
        or any(
            name not in SERVER_NAMES or type(value) is not bool
            for name, value in stage["node_clock_overrides"].items()
        )
    ):
        raise ValueError("bounded explicit SDK SIM node clock overrides required")
    return {"use_sim_time": stage["node_clock_overrides"].get(node_name, True)}
