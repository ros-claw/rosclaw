"""Closed disposable-SIM launch clock overrides, never a live reconfiguration API."""

import re
from datetime import datetime

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


def validate_counterfactual_capture_context(settings):
    """Allow an unchanged staged clock configuration in an unadmitted replica.

    Observation roles terminate in the existing source-only capture path, with
    no goals or Body admission. They do not need an injected fault label to
    reproduce the operator's original typed configuration.
    """
    if type(settings) is not dict or type(settings.get("engineering_capture")) is not dict:
        raise ValueError("owned isolated counterfactual capture required")
    capture = settings["engineering_capture"]
    role, phase = capture.get("case"), capture.get("phase")
    if type(role) is not str or type(phase) is not str:
        raise ValueError("closed counterfactual capture role and phase required")
    # Use the same isolation, run, no-fault and phase checks as the fixture's
    # authoritative unadmitted capture path; default Nav2 imports remain lazy.
    from simulation_engineering_capture import OBSERVATION_CASES, validate_engineering_capture

    validate_engineering_capture(settings)
    readonly = role in OBSERVATION_CASES
    if role != "rsk_node_clock_mismatch" and not readonly:
        raise ValueError(
            "clock override requires a closed counterfactual or source-only capture role"
        )
    if readonly and (
        type(settings["run_id"]) is not str
        or re.fullmatch(r"[0-9a-f]{32}", settings["run_id"]) is None
        or type(settings["governed_sink"]) is not dict
        or settings["governed_sink"].get("schema_version") != "rosclaw.governed_sink_bootstrap.v1"
    ):
        raise ValueError(
            "source-only observation role with its original run and guarded sink required"
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


def configured_clock_readback_available(native, descriptor):
    """Require actual fresh reads of declared clocks, independent of fault labels."""
    stage = (descriptor or {}).get("counterfactual_sim")
    if stage is None:
        return True
    counterfactual_clock_parameters(descriptor, "controller_server")
    namespace = descriptor.get("namespace")
    if namespace == "/":
        namespace = ""
    if type(namespace) is not str or (
        namespace and re.fullmatch(r"(/[A-Za-z][A-Za-z0-9_]*)+", namespace) is None
    ):
        raise ValueError("original absolute or root clock namespace required")
    if type(native) is not dict or type(native.get("observations")) is not dict:
        return False
    observations = native["observations"]
    params, stamps = observations.get("node_parameters"), observations.get("parameter_captured_at")
    if type(params) is not dict or type(stamps) is not dict:
        return False
    try:
        now = datetime.fromisoformat(native["captured_at"])
        if now.tzinfo is None:
            return False
        for name, value in stage["node_clock_overrides"].items():
            node = namespace + "/" + name
            actual = params.get(node)
            if type(actual) is not dict or actual.get("use_sim_time") is not value:
                return False
            at = datetime.fromisoformat(stamps[node])
            if at.tzinfo is None or not 0 <= (now - at).total_seconds() <= 0.5:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True
