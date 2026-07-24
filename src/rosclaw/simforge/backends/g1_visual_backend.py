"""Optional GoalForge trajectory rendering metadata.

Rendering is intentionally downstream of evidence generation: this module does
not produce task labels and cannot make a candidate promotable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def trajectory_overlay(trajectory: dict[str, np.ndarray]) -> dict[str, Any]:
    ball = np.asarray(trajectory.get("ball_pose", np.empty((0, 7))))
    com = np.asarray(trajectory.get("com", np.empty((0, 3))))
    slip = np.asarray(trajectory.get("support_foot_slip", np.empty((0,))))
    return {
        "schema_version": "rosclaw.g1_goalforge.visual_overlay.v1",
        "frame_count": int(ball.shape[0]),
        "ball_xyz": ball[:, :3].tolist() if ball.ndim == 2 else [],
        "com_xyz": com[:, :3].tolist() if com.ndim == 2 else [],
        "support_slip_m": slip.tolist(),
        "label_source": "visualization_only",
    }


__all__ = ["trajectory_overlay"]
