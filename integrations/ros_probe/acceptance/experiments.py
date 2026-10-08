"""Predeclared SIM planning candidates; predictions grant no execution credit."""

import math


def planning_parameters(profile, preset="baseline"):
    """Keep Body dimensions and original profile-specific baseline unchanged."""
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
    elif preset not in ("baseline", "perimeter", "perimeter_sequential", "perimeter_stateless"):
        raise ValueError("unknown predeclared coverage preset")
    return params


def controller_parameters(profile, preset="baseline"):
    """Only goal-rotation statefulness differs; speed/stop guards stay original."""
    planning_parameters(profile, preset)
    return (
        {"stateful": False}
        if preset in ("perimeter_stateless", "perimeter_stateless_headland")
        else {}
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
