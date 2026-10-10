"""Only daemon-owned source-bound boundary goals may choose the checkpoint BT."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "boundary_tracking_bt", ROOT / "integrations/ros_probe/acceptance/precise_through_poses_bt.py"
)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)
ORIGINAL = (ROOT / "tests/fixtures/ros_expert/nav2/installed-through-poses.xml").read_bytes()
CENTERS = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]


def prepare(tmp_path):
    return bt.prepare_boundary_tracking_through_poses_bt(
        tmp_path,
        ORIGINAL,
        xy_goal_tolerance=0.025,
        controller_lookahead_m=0.1,
        tracking_radius_m=0.1,
    )["source_output_sha256"]


def executor(tmp_path, **changes):
    return RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(fresh=lambda: {"x": 0.9, "y": 0.9}),
        output=tmp_path / "actions",
        body_id="fixture",
        body_snapshot_hash="body",
        grid={"frame_id": "map"},
        recovery_centers=CENTERS,
        **{
            "boundary_pass": True,
            "boundary_strategy": "through_poses_tracking_midpoints",
            **changes,
        },
    )


@pytest.mark.parametrize("status, expected", [(4, "SUCCEEDED"), (5, "FAILED")])
def test_source_bound_boundary_override_keeps_original_targets_deadline_and_terminal(
    tmp_path, status, expected
):
    digest = prepare(tmp_path)
    instance = executor(tmp_path, boundary_tracking_bt_sha256=digest)
    calls = []
    terminal = {"status": status, "result": {"error_code": 0}}
    if status == 5:
        terminal["timed_out"] = True
    instance._run_goal = lambda *args, **kwargs: calls.append((args, kwargs)) or terminal
    result = instance._boundary({"status": 4}, "root", 123)
    assert result["status"] == expected and result["nav_goal_result"] is terminal
    args, kwargs = calls[0]
    assert args[:2] == ("/navigate_through_poses", "nav2_msgs/action/NavigateThroughPoses")
    assert args[3:] == ("root:boundary", 123)
    assert kwargs == {"goal_timeout_sec": 180, "stage": "BOUNDARY_PASS"}
    assert args[2]["behavior_tree"] == str(tmp_path / "boundary-through-poses.xml")
    assert len(args[2]["poses"]) == result["waypoint_count"] == 9
    assert all(
        (p["pose"]["position"]["x"], p["pose"]["position"]["y"]) in CENTERS
        for p in args[2]["poses"]
    )
    assert "coverage_ratio" not in result
    assert instance._boundary({"status": 6}, "root", 123)["status"] == "SKIPPED"
    assert len(calls) == 1


@pytest.mark.parametrize("digest", [None, True, "a" * 63, "G" * 64])
def test_boundary_override_requires_explicit_valid_source_digest(tmp_path, digest):
    with pytest.raises(ValueError, match="source-bound"):
        executor(tmp_path, boundary_tracking_bt_sha256=digest)


def test_missing_redirected_or_wrong_source_bt_refused_before_any_action(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        executor(tmp_path, boundary_tracking_bt_sha256="a" * 64)
    digest = prepare(tmp_path)
    with pytest.raises(RuntimeError, match="SHA256"):
        executor(tmp_path, boundary_tracking_bt_sha256="0" * 64)
    path = tmp_path / "boundary-through-poses.xml"
    redirected = tmp_path / "other.xml"
    path.rename(redirected)
    path.symlink_to(redirected)
    with pytest.raises(RuntimeError, match="redirected"):
        executor(tmp_path, boundary_tracking_bt_sha256=digest)


def test_bt_tampering_after_startup_refuses_dispatch_and_legacy_ignores_override(tmp_path):
    digest = prepare(tmp_path)
    instance = executor(tmp_path, boundary_tracking_bt_sha256=digest)
    (tmp_path / "boundary-through-poses.xml").write_bytes(b"modified")
    calls = []
    instance._run_goal = lambda *args, **kwargs: calls.append(args) or {"status": 4}
    with pytest.raises(RuntimeError, match="SHA256"):
        instance._boundary({"status": 4}, "root", 123)
    assert not calls
    legacy = executor(tmp_path, boundary_strategy="through_poses_midpoints")
    legacy._run_goal = lambda *args, **kwargs: calls.append(args) or {"status": 4}
    assert legacy._boundary({"status": 4}, "root", 123)["status"] == "SUCCEEDED"
    assert "behavior_tree" not in calls[0][2]
    with pytest.raises(ValueError, match="other boundary strategies"):
        executor(
            tmp_path,
            boundary_strategy="through_poses_midpoints",
            boundary_tracking_bt_sha256=digest,
        )


def test_boundary_bt_read_is_bounded_even_if_expected_digest_matches_large_file(tmp_path):
    raw = b"x" * 128_001
    (tmp_path / "boundary-through-poses.xml").write_bytes(raw)
    with pytest.raises(RuntimeError, match="SHA256"):
        executor(tmp_path, boundary_tracking_bt_sha256=hashlib.sha256(raw).hexdigest())
