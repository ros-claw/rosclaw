"""Opt-in owned Nav2 BT source; predictions and waypoint pruning grant no credit."""

import hashlib
import json
import math
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET


def prepare_precise_through_poses_bt(output, original, *, xy_goal_tolerance):
    if type(original) is not bytes or not 0 < len(original) <= 128_000:
        raise ValueError("bounded original installed through-poses BT required")
    if b"<!DOCTYPE" in original.upper() or b"<!ENTITY" in original.upper():
        raise ValueError("external BT source entities refused")
    if (
        type(xy_goal_tolerance) not in (int, float)
        or not math.isfinite(xy_goal_tolerance)
        or not 0 < xy_goal_tolerance <= 0.05
    ):
        raise ValueError("existing precise goal-checker position tolerance required")
    tree = ET.fromstring(original, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    pipelines = tree.findall(".//PipelineSequence[@name='NavigateWithReplanning']")
    removals = tree.findall(".//RemovePassedGoals")
    if tree.get("BTCPP_format") != "4" or len(pipelines) != 1 or len(removals) != 1:
        raise ValueError("one explicit installed Nav2 replanning pipeline required")
    pipeline, removal = pipelines[0], removals[0]
    rates = pipeline.findall("RateController")
    if len(rates) != 1:
        raise ValueError("one original bounded planner rate controller required")
    sequences = rates[0].findall("./RecoveryNode/ReactiveSequence")
    if (
        len(sequences) != 1
        or removal not in list(sequences[0])
        or removal.attrib != {"input_goals": "{goals}", "output_goals": "{goals}", "radius": "0.7"}
        or len(sequences[0].findall("ComputePathThroughPoses")) != 1
    ):
        raise ValueError("exact original coarse waypoint pruning source required")
    sequences[0].remove(removal)
    removal.set("radius", str(xy_goal_tolerance))
    # The planner retains its original rate. Pruning is checked every BT tick
    # so a precise waypoint crossing cannot disappear between planner cycles.
    pipeline.insert(list(pipeline).index(rates[0]), removal)
    modified = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
    root = Path(output)
    paths = {
        "original": root / "repair-through-poses.original.xml",
        "modified": root / "repair-through-poses.xml",
        "report": root / "repair-through-poses-source.json",
    }
    report = {
        "schema_version": "rosclaw.precise_through_poses_source.v1",
        "source_original_sha256": hashlib.sha256(original).hexdigest(),
        "source_output_sha256": hashlib.sha256(modified).hexdigest(),
        "original_prune_radius_m": 0.7,
        "prune_radius_m": xy_goal_tolerance,
        "original_planner_rate_hz": rates[0].get("hz"),
        "pruning_outside_planner_rate_controller": True,
        "actual_waypoint_reached": "NOT_MEASURED",
        "coverage_credit": "INDEPENDENT_BRUSH_OBSERVATIONS_ONLY",
        "authorization": False,
        "physical_acceptance": "NOT_RUN",
    }
    for key, raw in [
        ("original", original),
        ("modified", modified),
        ("report", (json.dumps(report, indent=2) + "\n").encode()),
    ]:
        with paths[key].open("xb") as stream:
            stream.write(raw)
    return report


def prepare_repair_tracking_through_poses_bt(
    output, original, *, xy_goal_tolerance, controller_lookahead_m, tracking_radius_m
):
    """Separate opt-in repair checkpoints; global/final precise policy remains.

    Reuse the validated installed-tree transformation in an isolated directory.
    Pruned intermediate targets grant neither arrival nor coverage credit.
    """
    root = Path(output)
    with tempfile.TemporaryDirectory(prefix=".repair-tracking-bt-", dir=root) as temporary:
        report = prepare_boundary_tracking_through_poses_bt(
            temporary,
            original,
            xy_goal_tolerance=xy_goal_tolerance,
            controller_lookahead_m=controller_lookahead_m,
            tracking_radius_m=tracking_radius_m,
        )
        modified = (Path(temporary) / "boundary-through-poses.xml").read_bytes()
    report = {
        **report,
        "schema_version": "rosclaw.repair_tracking_through_poses_source.v1",
        "scope": "REPAIR_INTERMEDIATE_CHECKPOINTS_ONLY",
    }
    for name, raw in [
        ("repair-tracking-through-poses.original.xml", original),
        ("repair-tracking-through-poses.xml", modified),
        (
            "repair-tracking-through-poses-source.json",
            (json.dumps(report, indent=2) + "\n").encode(),
        ),
    ]:
        with (root / name).open("xb") as stream:
            stream.write(raw)
    return report


def prepare_boundary_tracking_through_poses_bt(
    output, original, *, xy_goal_tolerance, controller_lookahead_m, tracking_radius_m
):
    """Separate bounded boundary checkpoints; final goal and repair BT stay unchanged.

    This known-fixture hypothesis uses the existing 100mm controller lookahead
    for intermediate waypoint pruning. It grants no arrival or cleaning credit.
    The existing precise source validator preserves every recovery/stop node.
    """
    if (
        type(xy_goal_tolerance) not in (int, float)
        or not math.isfinite(xy_goal_tolerance)
        or not 0 < xy_goal_tolerance <= 0.05
        or type(controller_lookahead_m) not in (int, float)
        or controller_lookahead_m != 0.1
        or type(tracking_radius_m) not in (int, float)
        or tracking_radius_m != 0.1
    ):
        raise ValueError("boundary checkpoint requires registered 100mm existing lookahead")
    root = Path(output)
    with tempfile.TemporaryDirectory(prefix=".boundary-bt-", dir=root) as temporary:
        precise = prepare_precise_through_poses_bt(
            temporary, original, xy_goal_tolerance=xy_goal_tolerance
        )
        tree = ET.fromstring(
            (Path(temporary) / "repair-through-poses.xml").read_bytes(),
            parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)),
        )
    removal = tree.find(".//PipelineSequence[@name='NavigateWithReplanning']/RemovePassedGoals")
    if removal is None or removal.get("radius") != str(xy_goal_tolerance):
        raise ValueError("validated precise waypoint source required")
    removal.set("radius", str(tracking_radius_m))
    modified = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
    report = {
        **precise,
        "schema_version": "rosclaw.boundary_tracking_through_poses_source.v1",
        "source_output_sha256": hashlib.sha256(modified).hexdigest(),
        "prune_radius_m": tracking_radius_m,
        "scope": "BOUNDARY_INTERMEDIATE_CHECKPOINTS_ONLY",
        "unchanged_final_goal_xy_tolerance_m": xy_goal_tolerance,
        "unchanged_controller_lookahead_m": controller_lookahead_m,
        "global_precise_repair_bt_changed": False,
    }
    for name, raw in [
        ("boundary-through-poses.original.xml", original),
        ("boundary-through-poses.xml", modified),
        ("boundary-through-poses-source.json", (json.dumps(report, indent=2) + "\n").encode()),
    ]:
        with (root / name).open("xb") as stream:
            stream.write(raw)
    return report
