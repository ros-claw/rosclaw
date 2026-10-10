"""Predeclared SIM planning candidates; predictions grant no execution credit."""

import math

INNER_RING_PROFILES = {
    "perimeter_stateless_clearance_inner_ring": "burger",
    "perimeter_stateless_inner_ring": "waffle",
}

CONTINUOUS_BOUNDARY_PRESETS = {
    "perimeter_stateless_overlap_continuous": ("waffle", "perimeter_stateless_overlap"),
    "perimeter_stateless_clearance_continuous": ("burger", "perimeter_stateless_clearance"),
    "perimeter_stateless_overlap_boundary_tracking": ("waffle", "perimeter_stateless_overlap"),
    "perimeter_stateless_clearance_boundary_tracking": ("burger", "perimeter_stateless_clearance"),
}


BOUNDARY_TRACKING_PRESETS = frozenset(
    (
        "perimeter_stateless_overlap_boundary_tracking",
        "perimeter_stateless_clearance_boundary_tracking",
    )
)


def continuous_boundary_strategy(preset):
    if preset not in CONTINUOUS_BOUNDARY_PRESETS:
        raise ValueError("registered continuous boundary preset required")
    return (
        "through_poses_tracking_midpoints"
        if preset in BOUNDARY_TRACKING_PRESETS
        else "through_poses_midpoints"
    )


def valid_boundary_tracking_sha256(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def planning_parameters(profile, preset="baseline"):
    """Keep Body dimensions and original profile-specific baseline unchanged."""
    if preset in CONTINUOUS_BOUNDARY_PRESETS:
        expected, original = CONTINUOUS_BOUNDARY_PRESETS[preset]
        if profile.name != expected:
            raise ValueError("continuous boundary candidate requires its registered known profile")
        return planning_parameters(profile, original)
    params = {
        "default_headland_width": profile.coverage_width_m,
        "default_swath_angle_type": "SET_ANGLE",
        "default_swath_angle": 0.0,
        "default_route_type": "BOUSTROPHEDON",
    }
    if preset == "diagonal":
        if profile.name != "waffle":
            raise ValueError("diagonal candidate lacks safe offline clearance for this profile")
        params["default_swath_angle"] = math.pi / 4
    elif preset == "headland":
        # Offline-screened known fixtures, not a claim of generic Body adaptation.
        params["default_headland_width"] = {"waffle": 0.45, "burger": 0.35}[profile.name]
    elif preset == "perimeter_stateless_headland":
        # Burger's baseline planned turns have only24mm conservative clearance;
        # use the already screened35cm headland without changing any controller guard.
        if profile.name != "burger":
            raise ValueError("safe-headland combination is preregistered only for Burger")
        params["default_headland_width"] = 0.35
    elif preset in ("perimeter_stateless_clearance", "perimeter_stateless_clearance_inner_ring"):
        # Known-Burger diagnostic only. Source7a main-path tracking deviation
        # exceeded the 74mm static margin; 0.50m predicts 224mm clearance.
        # This is a hypothesis, not a physical safety/efficiency guarantee.
        if profile.name != "burger":
            raise ValueError("clearance diagnostic is declared only for known Burger")
        params["default_headland_width"] = 0.50
    elif preset == "perimeter_stateless_inner_ring":
        # Existing Waffle planner, extra legal inset targets only. Nominal
        # pilot replay is not actual Nav2 credit or a measured time saving.
        if profile.name != "waffle":
            raise ValueError("inner-ring diagnostic is declared only for known Waffle")
    elif preset == "perimeter_stateless_overlap":
        # Explicit known-fixture diagnostic: more overlap addresses measured
        # strip/endpoint misses without shrinking the real verifier brush or
        # changing robot geometry, speeds, controller guards or repair costs.
        # New frozen development variant: Waffle's 0.35 m spacing has the
        # same 0.50 m headland and sampled 0.124 m turn clearance as 0.40 m.
        # Offline coverage gain is a prediction, not measured task credit.
        params["operation_width"] = {"waffle": 0.35, "burger": 0.27}[profile.name]
        params["default_headland_width"] = {"waffle": 0.50, "burger": 0.35}[profile.name]
    elif preset not in ("baseline", "perimeter", "perimeter_sequential", "perimeter_stateless"):
        raise ValueError("unknown predeclared coverage preset")
    return params


def controller_parameters(profile, preset="baseline"):
    """Only goal-rotation statefulness differs; speed/stop guards stay original."""
    planning_parameters(profile, preset)
    if preset in CONTINUOUS_BOUNDARY_PRESETS:
        preset = CONTINUOUS_BOUNDARY_PRESETS[preset][1]
    return (
        {"stateful": False}
        if preset
        in (
            "perimeter_stateless",
            "perimeter_stateless_headland",
            "perimeter_stateless_overlap",
            "perimeter_stateless_clearance",
            "perimeter_stateless_clearance_inner_ring",
            "perimeter_stateless_inner_ring",
        )
        else {}
    )


def validate_continuous_boundary_experiment(experiment):
    """Reject incomplete candidate declarations before starting the daemon."""
    if type(experiment) is not dict:
        raise ValueError("explicit experiment mapping required")
    preset = experiment.get("preset")
    registered = CONTINUOUS_BOUNDARY_PRESETS.get(preset) if isinstance(preset, str) else None
    tracking = isinstance(preset, str) and preset in BOUNDARY_TRACKING_PRESETS
    tracking_keys = ("boundary_tracking_prune_radius_m", "boundary_tracking_bt_sha256")
    if any(k in experiment for k in tracking_keys) and not tracking:
        raise ValueError("boundary tracking metadata requires its registered candidate")
    if registered is None and experiment.get("boundary_strategy") not in (
        "through_poses_midpoints",
        "through_poses_tracking_midpoints",
    ):
        return
    if (
        registered is None
        or experiment.get("profile") != registered[0]
        or experiment.get("boundary_strategy") != continuous_boundary_strategy(preset)
        or experiment.get("boundary_pass") is not True
        or experiment.get("precise_through_poses") is not True
        or type(experiment.get("boundary_stage_budget_sec")) is not int
        or experiment["boundary_stage_budget_sec"] != 180
        or type(experiment.get("boundary_waypoint_count")) is not int
        or experiment["boundary_waypoint_count"] != 9
    ):
        raise ValueError("continuous boundary experiment requires nine precise bounded waypoints")
    if tracking and (
        type(experiment.get("boundary_tracking_prune_radius_m")) not in (int, float)
        or experiment["boundary_tracking_prune_radius_m"] != 0.1
        or not valid_boundary_tracking_sha256(experiment.get("boundary_tracking_bt_sha256"))
    ):
        raise ValueError(
            "boundary tracking requires registered 100mm checkpoints and source SHA256"
        )


def validate_seed(seed):
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**31 - 1):
        raise ValueError("SIM seed must be an integer from 0 through 2147483647")
    return seed


def gazebo_arguments(world, seed=None):
    validate_seed(seed)
    args = ["gz", "sim", "-r", "-s", "--headless-rendering"]
    if seed is not None:
        args.extend(["--seed", str(seed)])
    return args + [str(world)]


def validate_inner_ring_experiment(experiment):
    """Fail before daemon Runtime startup if a new ring's declaration drifts."""
    if type(experiment) is not dict:
        raise ValueError("explicit experiment mapping required")
    preset = experiment.get("preset")
    expected_profile = INNER_RING_PROFILES.get(preset) if isinstance(preset, str) else None
    if expected_profile is None and experiment.get("boundary_strategy") != "sequential_inner_ring":
        return
    if (
        expected_profile is None
        or experiment.get("profile") != expected_profile
        or experiment.get("boundary_strategy") != "sequential_inner_ring"
        or experiment.get("boundary_pass") is not True
        or type(experiment.get("boundary_stage_budget_sec")) is not int
        or experiment["boundary_stage_budget_sec"] != 360
        or type(experiment.get("inner_boundary_inset_cells")) is not int
        or experiment["inner_boundary_inset_cells"] != 1
    ):
        raise ValueError("inner boundary experiment must match the registered known-fixture stage")


def validate_boundary_tracking_runtime_registration(experiment, protocol):
    """Reject a generated boundary BT that differs from the frozen protocol."""
    validate_continuous_boundary_experiment(experiment)
    if experiment.get("preset") not in BOUNDARY_TRACKING_PRESETS:
        return
    if not isinstance(protocol, dict):
        raise ValueError("boundary tracking requires a preregistered runtime protocol")
    for field in (
        "boundary_strategy",
        "boundary_stage_budget_sec",
        "boundary_waypoint_count",
        "boundary_tracking_prune_radius_m",
        "boundary_tracking_bt_sha256",
    ):
        registered = protocol.get("candidate_" + field)
        if type(registered) is not type(experiment[field]) or registered != experiment[field]:
            raise ValueError("boundary tracking runtime differs from preregistered " + field)
    if protocol.get("precise_repair_waypoints") is not True:
        raise ValueError("boundary tracking runtime requires preregistered precise repair BT")
