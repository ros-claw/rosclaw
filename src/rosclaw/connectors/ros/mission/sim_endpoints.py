"""Frozen simulator endpoint names; configuration never grants authority."""

import re
from types import MappingProxyType

DEFAULT_ENDPOINTS = MappingProxyType(
    {
        "navigate_to_pose": "/navigate_to_pose",
        "navigate_through_poses": "/navigate_through_poses",
        "navigate_complete_coverage": "/navigate_complete_coverage",
        "set_initial_pose": "/set_initial_pose",
        "lease": "/rosclaw_sim/lease",
        "cleaning": "/rosclaw_sim/cleaning",
        "hold": "/rosclaw_sim/hold",
    }
)


def absolute_endpoint(name):
    if (
        type(name) is not str
        or len(name) > 256
        or not re.fullmatch(r"/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*", name)
    ):
        raise ValueError("bounded fully qualified ROS endpoint required")
    return name


def freeze_sim_endpoints(endpoints=None):
    """Require a whole configured set; never infer an unknown namespace."""
    if endpoints is None:
        return DEFAULT_ENDPOINTS
    if type(endpoints) is not dict or set(endpoints) not in (
        set(DEFAULT_ENDPOINTS),
        set(DEFAULT_ENDPOINTS) - {"hold"},
    ):
        raise ValueError("complete explicit simulator endpoint configuration required")
    frozen = {key: absolute_endpoint(value) for key, value in endpoints.items()}
    frozen.setdefault("hold", DEFAULT_ENDPOINTS["hold"])
    if len(set(frozen.values())) != len(frozen):
        raise ValueError("simulator endpoint roles must be distinct")
    return MappingProxyType(frozen)


def freeze_sim_spawn(spawn=(0.0, 0.0, 0.0)):
    """Operator-configured startup pose, not an action-time GT correction."""
    if (
        type(spawn) not in (list, tuple)
        or len(spawn) != 3
        or any(type(v) not in (int, float) or not -1_000_000 <= v <= 1_000_000 for v in spawn)
    ):
        raise ValueError("bounded finite configured simulator spawn required")
    return tuple(float(v) for v in spawn)
