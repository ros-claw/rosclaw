"""Task semantics and source-backed solution cards. Resolution does not install."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from rosclaw.connectors.ros.intelligence import RosSystemModel

from .capability_resolver import resolve_capabilities

COVERAGE_REQUIREMENTS = [
    "navigation.navigate_to_pose",
    "localization.pose_estimate",
    "mapping.occupancy_map",
    "state.odometry",
    "sensing.lidar",
    "coverage.compute_path",
    "coverage.execute",
    "coverage.verify",
    "safety.collision_monitor",
    "cleaning.enable",
    "cleaning.disable",
]


def task_class(task: str) -> str:
    lowered = task.casefold()
    if any(token in lowered for token in ("clean", "coverage", "清扫", "扫地", "扫一遍", "打扫")):
        return "complete_area_cleaning"
    if any(token in lowered for token in ("navigat", "导航", "目标点")):
        return "navigate_to_pose"
    raise ValueError("unsupported task; specify coverage cleaning or navigation")


def resolve_task(model: RosSystemModel, task: str, *, now: datetime | None = None) -> dict:
    category = task_class(task)
    required = (
        COVERAGE_REQUIREMENTS
        if category == "complete_area_cleaning"
        else ["navigation.navigate_to_pose"]
    )
    capabilities = {c["semantic_id"]: c for c in resolve_capabilities(model, now=now)}
    cards_path = Path(__file__).parent.parent / "knowledge" / "catalog.yaml"
    cards = yaml.safe_load(cards_path.read_text(encoding="utf-8"))["cards"]
    generation = model.environment.get("ros_generation")
    distro = model.environment.get("distro")
    missing = [r for r in required if capabilities[r]["status"] == "MISSING"]
    recommendations, alternatives = [], []
    for card in cards:
        if not set(card["provides"]) & set(missing):
            continue
        selection = {**card, "reason": "Provides missing task semantics; reuse existing runtime."}
        if generation in card["supported_ros"] and distro in card["supported_distros"]:
            recommendations.append(selection)
        else:
            alternatives.append(
                {**selection, "reason": "Compatibility not established for this environment."}
            )
    return {
        "schema_version": "rosclaw.ros_solution.v1",
        "task_class": category,
        "snapshot_id": model.snapshot_id,
        "available": [r for r in required if capabilities[r]["status"] == "AVAILABLE"],
        "missing": missing,
        "blocked": [r for r in required if capabilities[r]["status"] == "BLOCKED"],
        "unknown": [r for r in required if capabilities[r]["status"] == "UNKNOWN"],
        "recommended": recommendations,
        "alternatives": alternatives,
        "body_requirements": ["cleaning_footprint", "coverage_polygon", "exclusion_zones"],
        "execution_entry": "request_action",
        "configured": False,
    }
