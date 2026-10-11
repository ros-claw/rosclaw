"""Owned fixture BT semantics; these source checks grant no physical credit."""

import hashlib
import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "precise_bt", ROOT / "integrations/ros_probe/acceptance/precise_through_poses_bt.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ORIGINAL = (ROOT / "tests/fixtures/ros_expert/nav2/installed-through-poses.xml").read_bytes()


def test_only_pruning_location_and_existing_tolerance_change(tmp_path):
    report = m.prepare_precise_through_poses_bt(tmp_path, ORIGINAL, xy_goal_tolerance=0.025)
    original = ET.fromstring(ORIGINAL)
    modified = ET.fromstring((tmp_path / "repair-through-poses.xml").read_bytes())
    pipeline = modified.find(".//PipelineSequence")
    removal = pipeline.find("RemovePassedGoals")
    rate = pipeline.find("RateController")
    assert list(pipeline).index(removal) + 1 == list(pipeline).index(rate)
    assert removal.get("radius") == "0.025"
    assert rate.get("hz") == "0.333"
    assert rate.find(".//RemovePassedGoals") is None
    # Undo the two intended semantic changes: the whole tree must match.
    pipeline.remove(removal)
    removal.set("radius", "0.7")
    rate.find("./RecoveryNode/ReactiveSequence").insert(0, removal)
    assert ET.tostring(modified) == ET.tostring(original)
    assert report["source_original_sha256"] == hashlib.sha256(ORIGINAL).hexdigest()
    assert report["physical_acceptance"] == "NOT_RUN"
    assert report["authorization"] is False
    assert report["actual_waypoint_reached"] == "NOT_MEASURED"
    assert json.loads((tmp_path / "repair-through-poses-source.json").read_text()) == report


def test_boundary_checkpoint_bt_changes_only_radius_and_keeps_global_repair_bytes(tmp_path):
    m.prepare_precise_through_poses_bt(tmp_path, ORIGINAL, xy_goal_tolerance=0.025)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    report = m.prepare_boundary_tracking_through_poses_bt(
        tmp_path,
        ORIGINAL,
        xy_goal_tolerance=0.025,
        controller_lookahead_m=0.1,
        tracking_radius_m=0.1,
    )
    assert all((tmp_path / name).read_bytes() == raw for name, raw in before.items())

    def parser():
        return ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))

    precise = ET.fromstring(before["repair-through-poses.xml"], parser=parser())
    boundary_raw = (tmp_path / "boundary-through-poses.xml").read_bytes()
    boundary = ET.fromstring(boundary_raw, parser=parser())
    removal = boundary.find(".//PipelineSequence/RemovePassedGoals")
    assert removal.get("radius") == "0.1"
    removal.set("radius", "0.025")
    # Restore the sole boundary-specific change and compare the entire tree,
    # including comments, planner/recovery nodes and all control attributes.
    assert ET.tostring(boundary) == ET.tostring(precise)
    assert report["source_output_sha256"] == hashlib.sha256(boundary_raw).hexdigest()
    assert report["unchanged_final_goal_xy_tolerance_m"] == 0.025
    assert report["unchanged_controller_lookahead_m"] == 0.1
    assert report["global_precise_repair_bt_changed"] is False
    assert report["physical_acceptance"] == "NOT_RUN"
    assert report["authorization"] is False
    assert report["actual_waypoint_reached"] == "NOT_MEASURED"
    assert not any(p.name.startswith(".boundary-bt-") for p in tmp_path.iterdir())


@pytest.mark.parametrize(
    "radius", [True, 0.025, 0.05, 0.1001, 0.7, float("nan"), float("inf"), "0.1"]
)
def test_boundary_checkpoint_refuses_unregistered_radius_before_output(tmp_path, radius):
    with pytest.raises(ValueError, match="registered 100mm"):
        m.prepare_boundary_tracking_through_poses_bt(
            tmp_path,
            ORIGINAL,
            xy_goal_tolerance=0.025,
            controller_lookahead_m=0.1,
            tracking_radius_m=radius,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("lookahead", [True, 0.05, 0.2, float("nan")])
def test_boundary_checkpoint_cannot_silently_change_existing_lookahead(tmp_path, lookahead):
    with pytest.raises(ValueError, match="existing lookahead"):
        m.prepare_boundary_tracking_through_poses_bt(
            tmp_path,
            ORIGINAL,
            xy_goal_tolerance=0.025,
            controller_lookahead_m=lookahead,
            tracking_radius_m=0.1,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("tolerance", [True, 0, -0.1, 0.051, float("nan"), float("inf"), "0.025"])
def test_invalid_or_relaxed_tolerance_rejected_before_write(tmp_path, tolerance):
    with pytest.raises(ValueError):
        m.prepare_precise_through_poses_bt(tmp_path, ORIGINAL, xy_goal_tolerance=tolerance)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "source",
    [
        b"",
        b" " * 128_001,
        b'<!DOCTYPE root SYSTEM "file:///etc/passwd">' + ORIGINAL,
        ORIGINAL.replace(b'radius="0.7"', b'radius="0.8"'),
        ORIGINAL.replace(b'BTCPP_format="4"', b'BTCPP_format="3"'),
        ORIGINAL.replace(b"<ComputePathThroughPoses ", b"<OtherPlanner "),
        ORIGINAL.replace(b"<RemovePassedGoals ", b"<RemovePassedGoals/><RemovePassedGoals "),
    ],
)
def test_unexpected_installed_template_rejected_before_write(tmp_path, source):
    with pytest.raises(ValueError):
        m.prepare_precise_through_poses_bt(tmp_path, source, xy_goal_tolerance=0.025)
    assert not list(tmp_path.iterdir())


def test_original_evidence_cannot_be_overwritten(tmp_path):
    m.prepare_precise_through_poses_bt(tmp_path, ORIGINAL, xy_goal_tolerance=0.025)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        m.prepare_precise_through_poses_bt(tmp_path, ORIGINAL, xy_goal_tolerance=0.025)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
