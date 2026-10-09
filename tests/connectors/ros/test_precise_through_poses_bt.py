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
