"""Read-only Nav2 parameter diagnosis. Configuration risk is not observed collision."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rosclaw.connectors.ros.intelligence import RosSystemModel

COSTMAP_SOURCE = "https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/costmap_2d/"
STATIC_SOURCE = COSTMAP_SOURCE + "costmap_plugins/static/"


def parameter_issues(model: RosSystemModel, *, now: datetime) -> list[dict[str, Any]]:
    issues = []
    parameters = model.observations.get("node_parameters", {})
    captured = model.observations.get("parameter_captured_at", {})
    for node, values in parameters.items():
        if not isinstance(values, dict) or "costmap" not in node:
            continue
        stamp = captured.get(node)
        try:
            observed_at = datetime.fromisoformat(stamp)
            fresh = -100 <= (now - observed_at).total_seconds() * 1000 <= 5000
        except (TypeError, ValueError):
            fresh = False
        if not fresh:
            continue
        plugins = values.get("plugins")
        if not isinstance(plugins, list) or not all(isinstance(p, str) for p in plugins):
            continue
        # Resolve actual plugin classes, not conventional instance names. Custom
        # layers and unspecified classes are deliberately left unclassified.
        kinds = {p: values.get(p + ".plugin", "").replace("/", "::") for p in plugins}
        obstacles = [
            p
            for p in plugins
            if kinds[p] in {"nav2_costmap_2d::VoxelLayer", "nav2_costmap_2d::ObstacleLayer"}
            and values.get(p + ".enabled") is not False
        ]
        static = [
            p
            for p in plugins
            if kinds[p] == "nav2_costmap_2d::StaticLayer"
            and values.get(p + ".enabled") is not False
        ]
        for layer in static:
            earlier = [p for p in obstacles if plugins.index(p) < plugins.index(layer)]
            # Missing combination evidence remains a hypothesis; do not infer
            # a distro default from a partial parameter capture.
            if not earlier or values.get(layer + ".use_maximum") is not False:
                continue
            observation = {
                "node": node,
                "plugins": plugins,
                "static_layer": layer,
                "earlier_obstacle_layers": earlier,
                "use_maximum": False,
            }
            issues.append(
                {
                    "issue_code": "NAV2_COSTMAP_003",
                    "severity": "warning",
                    "evidence": [
                        {
                            "source": model.snapshot_id,
                            "observation": observation,
                            "timestamp": stamp,
                        }
                    ],
                    "confidence": 1.0,
                    "hypotheses": [
                        "A later StaticLayer may overwrite earlier obstacle costs; this configuration alone does not prove an observed obstacle was lost."
                    ],
                    "next_checks": [
                        "Compare timestamp-aligned LiDAR hits and master costmap cells at the same obstacle; inspect layer enabled state and combination method."
                    ],
                    "recommended_repairs": [
                        "Review ordering static → obstacle/voxel → inflation in an isolated simulation; compare obstacle cells, footprint clearance and collisions before adopting a configuration change."
                    ],
                    "official_sources": [COSTMAP_SOURCE, STATIC_SOURCE],
                    "validation_scope": "configuration_risk; isolated_simulation_required",
                    "runtime_mutation_required": False,
                }
            )
        inflations = [
            p
            for p in plugins
            if kinds[p] == "nav2_costmap_2d::InflationLayer"
            and values.get(p + ".enabled") is not False
        ]
        for layer in inflations:
            later = [p for p in obstacles if plugins.index(p) > plugins.index(layer)]
            if later:
                issues.append(
                    {
                        "issue_code": "NAV2_COSTMAP_004",
                        "severity": "warning",
                        "evidence": [
                            {
                                "source": model.snapshot_id,
                                "observation": {
                                    "node": node,
                                    "plugins": plugins,
                                    "inflation_layer": layer,
                                    "later_obstacle_layers": later,
                                },
                                "timestamp": stamp,
                            }
                        ],
                        "confidence": 1.0,
                        "hypotheses": [
                            "Obstacle costs added after inflation may lack the intended clearance gradient."
                        ],
                        "next_checks": [
                            "Inspect inflation around timestamp-aligned obstacle cells and compare against measured Body footprint."
                        ],
                        "recommended_repairs": [
                            "Review placing inflation after contributing obstacle layers and validate in an isolated simulation."
                        ],
                        "official_sources": [COSTMAP_SOURCE],
                        "validation_scope": "configuration_risk; isolated_simulation_required",
                        "runtime_mutation_required": False,
                    }
                )
    return issues
