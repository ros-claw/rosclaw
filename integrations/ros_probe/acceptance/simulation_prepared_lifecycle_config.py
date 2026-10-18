"""Respect declared Nav2 lifecycle-manager configuration in SDK staging."""

import yaml


def managed_lifecycle_nodes(raw, namespace, available):
    config = yaml.safe_load(raw)
    if type(config) is not dict:
        raise ValueError("Nav2 staging YAML object required")
    group = config[namespace] if namespace else config
    params = group.get("lifecycle_manager_navigation", {}).get("ros__parameters", {})
    declared = params.get("node_names", list(available))
    if (
        type(declared) is not list
        or not declared
        or any(type(name) is not str or name not in available for name in declared)
        or len(set(declared)) != len(declared)
    ):
        raise ValueError("unique declared existing Nav2 lifecycle nodes required")
    return list(declared)


def preinitialization_nodes(requirements, descriptor, engineering_capture):
    """Initialization is not L1: a new stack may need a pose before activating.

    Only the explicit owned bootstrap descriptor changes the legacy wait. Full
    navigation lifecycle/source/Body checks still run after initialization.
    """
    boundary = descriptor.get("initialization_boundary") if descriptor else None
    if boundary not in (None, "LOCALIZATION_ACTIVE_BEFORE_NAVIGATION"):
        raise ValueError("explicit supported initialization boundary required")
    if engineering_capture or boundary is not None:
        return [requirements["lifecycle_nodes"]["localization"]]
    return list(requirements["lifecycle_nodes"].values())
