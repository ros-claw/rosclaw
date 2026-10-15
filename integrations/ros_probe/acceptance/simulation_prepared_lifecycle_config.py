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
