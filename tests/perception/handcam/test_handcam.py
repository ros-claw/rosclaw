"""PR-PE-3 tests: camera pose contract + marker-free visual hand state."""

from __future__ import annotations

import numpy as np

from rosclaw.perception.handcam import (
    CameraIntrinsics,
    CameraPoseContract,
    VisualHandState,
    contact_distance_m,
    estimate_hand_state,
    measure_visual_stability,
)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(width=640, height=480, fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)


def _contract(**kw) -> CameraPoseContract:
    return CameraPoseContract(
        camera_pose_id=kw.get("camera_pose_id", "front_v1"),
        camera_id="d435i_test",
        intrinsics=_intrinsics(),
        roi=kw.get("roi", {}),
    )


def _depth_with_blob(
    center_uv: tuple[int, int], depth_at_blob_mm: int = 350, background_mm: int = 3000
) -> np.ndarray:
    depth = np.full((480, 640), background_mm, dtype=np.uint16)
    u, v = center_uv
    depth[v - 30 : v + 30, u - 30 : u + 30] = depth_at_blob_mm
    return depth


def test_pose_hash_changes_on_any_component() -> None:
    base = _contract()
    moved = _contract(camera_pose_id="front_v2")
    assert base.camera_pose_hash != moved.camera_pose_hash
    changed = base.validate_against(moved)
    # Same intrinsics/mount/roi/lighting — only the declared id changed.
    assert changed == []

    roi_changed = _contract(roi={"x0": 0, "y0": 0, "x1": 320, "y1": 480})
    assert "roi" in base.validate_against(roi_changed)
    assert base.camera_pose_hash != roi_changed.camera_pose_hash

    other_intr = CameraPoseContract(
        camera_pose_id="front_v1",
        camera_id="d435i_test",
        intrinsics=CameraIntrinsics(
            width=848, height=480, fx=420.0, fy=420.0, ppx=424.0, ppy=240.0
        ),
    )
    assert "intrinsics" in base.validate_against(other_intr)


def test_intrinsics_hash_deterministic_and_deproject() -> None:
    intr = _intrinsics()
    assert intr.intrinsic_hash == _intrinsics().intrinsic_hash
    x, y, z = intr.deproject(320.0, 240.0, 1.0)
    assert abs(x) < 1e-9 and abs(y) < 1e-9 and z == 1.0
    x, _, _ = intr.deproject(620.0, 240.0, 0.5)
    assert abs(x - 0.25) < 1e-9  # (620-320)*0.5/600


def test_hand_state_ok_on_clear_blob() -> None:
    depth = _depth_with_blob((320, 240), depth_at_blob_mm=350)
    state = estimate_hand_state(depth, _contract())
    assert state.state == "ok"
    assert state.centroid_3d is not None
    cx, cy, cz = state.centroid_3d
    assert abs(cx) < 0.05 and abs(cy) < 0.05 and 0.3 < cz < 0.45
    assert state.fingertip_3d is not None
    assert state.cluster is not None and state.cluster.ok


def test_hand_state_unknown_when_no_hand_or_scene() -> None:
    # No near blob: flat far scene.
    flat = np.full((480, 640), 3000, dtype=np.uint16)
    state = estimate_hand_state(flat, _contract())
    assert state.state == "unknown"
    assert state.centroid_3d is None

    # Whole-foreground scene (too dispersed → not a hand).
    near_scene = np.full((480, 640), 300, dtype=np.uint16)
    state2 = estimate_hand_state(near_scene, _contract())
    assert state2.state == "unknown"
    assert (
        "dispersed" in state2.reason or "too small" in state2.reason or state2.cluster is not None
    )

    # Empty depth.
    empty = np.zeros((480, 640), dtype=np.uint16)
    state3 = estimate_hand_state(empty, _contract())
    assert state3.state == "unknown"


def test_hand_state_respects_roi() -> None:
    depth = _depth_with_blob((500, 240), depth_at_blob_mm=350)
    roi_left = {"x0": 0, "y0": 0, "x1": 320, "y1": 480}
    state = estimate_hand_state(depth, _contract(roi=roi_left))
    # Blob is in the RIGHT half; left ROI must honestly see nothing.
    assert state.state == "unknown"
    roi_right = {"x0": 320, "y0": 0, "x1": 640, "y1": 480}
    state2 = estimate_hand_state(depth, _contract(roi=roi_right))
    assert state2.state == "ok"
    assert state2.centroid_3d[0] > 0  # right of principal point


def test_visual_stability_onset_and_settle() -> None:
    moving = [(0.01 * i, 0.0, 0.35) for i in range(10)]  # moving away in x
    still = [(0.1, 0.0, 0.35)] * 20
    trajectory = moving + still
    states = [
        VisualHandState("ok", "ok", c, c, None, 0.9)  # type: ignore[arg-type]
        for c in trajectory
    ]
    ts = [i * 0.033 for i in range(len(states))]
    result = measure_visual_stability(states, ts, settle_window_s=0.2, settle_threshold_m=0.005)
    assert result.settled
    assert result.onset_time_s is not None
    assert result.max_post_settle_displacement_m < 0.005

    # Unknown frames break settle.
    states_broken = states[:15]
    states_broken[12] = VisualHandState("unknown", "no cluster", None, None, None, 0.0)
    result2 = measure_visual_stability(states_broken, ts[:15], settle_window_s=0.2)
    assert result2.unknown_frames == 1


def test_contact_distance_requires_both_hands() -> None:
    a = VisualHandState("ok", "ok", (0.0, 0.0, 0.35), (0.0, 0.0, 0.30), None, 0.9)  # type: ignore[arg-type]
    b = VisualHandState("ok", "ok", (0.1, 0.0, 0.35), (0.08, 0.0, 0.30), None, 0.9)  # type: ignore[arg-type]
    assert abs(contact_distance_m(a, b) - 0.08) < 1e-9  # type: ignore[operator]
    unknown = VisualHandState("unknown", "no cluster", None, None, None, 0.0)
    assert contact_distance_m(a, unknown) is None
