"""Dual-hand and D435i self adapters (v3 §8.2, PR-PE-4)."""

from __future__ import annotations

from typing import Any

from rosclaw.practice.physical_observation import canonical_hash

from ..protocols import SelfStateAdapter
from .rh56 import RH56HandSelfAdapter


class DualHandSelfAdapter(SelfStateAdapter):
    """Both RH56 hands as ONE operational body (bimanual tasks: sync,
    leader-follower, near-contact).  Identity includes BOTH hand hashes
    plus the pairing — swapping one hand changes the body."""

    def __init__(
        self, left: RH56HandSelfAdapter, right: RH56HandSelfAdapter, *, body_id: str
    ) -> None:
        self._left = left
        self._right = right
        self._body_id = body_id

    def body_id(self) -> str:
        return self._body_id

    @property
    def left(self) -> RH56HandSelfAdapter:
        return self._left

    @property
    def right(self) -> RH56HandSelfAdapter:
        return self._right

    def body_hash(self) -> str:
        return canonical_hash(
            {
                "kind": "dual_rh56",
                "body_id": self._body_id,
                "left": self._left.body_hash(),
                "right": self._right.body_hash(),
            },
            prefix="body",
        )

    def current_state(self) -> dict[str, Any]:
        return {
            "body_id": self._body_id,
            "left": self._left.current_state(),
            "right": self._right.current_state(),
        }

    def health_channels(self) -> dict[str, float]:
        return {}

    def skew_channels(
        self,
        left_telemetry: dict[str, Any],
        right_telemetry: dict[str, Any],
    ) -> dict[str, float]:
        """Bimanual supervision channels (§10 T2): per-joint state skew
        between hands for mirrored gestures + temperature asymmetry."""
        channels: dict[str, float] = {}
        left_angles = left_telemetry.get("angle_actual") or {}
        right_angles = right_telemetry.get("angle_actual") or {}
        for joint in set(left_angles) & set(right_angles):
            channels[f"mirror_skew_{joint}"] = abs(
                float(left_angles[joint]) - float(right_angles[joint])
            )
        left_temps = [
            v
            for v in (left_telemetry.get("temperature_c") or {}).values()
            if isinstance(v, (int, float)) and v > 0
        ]
        right_temps = [
            v
            for v in (right_telemetry.get("temperature_c") or {}).values()
            if isinstance(v, (int, float)) and v > 0
        ]
        if left_temps and right_temps:
            channels["temperature_asymmetry"] = abs(max(left_temps) - max(right_temps))
        return channels


class D435iSensorSelfAdapter(SelfStateAdapter):
    """The D435i as a sensor-body: freshness state machine + observation
    reliability belief (v3 §8.4 camera freshness state machine)."""

    def __init__(self, camera_id: str, *, camera_pose_hash: str = "unmeasured") -> None:
        self._camera_id = camera_id
        self._pose_hash = camera_pose_hash

    def body_id(self) -> str:
        return self._camera_id

    def body_hash(self) -> str:
        return canonical_hash(
            {
                "kind": "d435i_sensor",
                "camera_id": self._camera_id,
                "camera_pose_hash": self._pose_hash,
            },
            prefix="body",
        )

    def current_state(self) -> dict[str, Any]:
        return {"camera_id": self._camera_id, "camera_pose_hash": self._pose_hash}

    def health_channels(self) -> dict[str, float]:
        return {}

    def freshness_state(
        self,
        *,
        frame_age_ms: float | None,
        rgb_depth_skew_ms: float | None,
        consecutive_missing: int,
        max_frame_age_ms: float = 500.0,
        max_rgb_depth_skew_ms: float = 50.0,
    ) -> dict[str, Any]:
        """Camera freshness state machine → reliability belief in [0, 1].

        Unknown inputs DEGRADE the belief (unknown ≠ fresh), and three
        consecutive missing frames flip the state to STALE regardless of
        the last measured age."""
        state = "FRESH"
        reasons: list[str] = []
        if consecutive_missing >= 3:
            state = "STALE"
            reasons.append(f"{consecutive_missing} consecutive missing frames")
        if frame_age_ms is None:
            if state == "FRESH":
                state = "DEGRADED"
            reasons.append("frame age unknown")
        elif frame_age_ms > max_frame_age_ms:
            state = "STALE"
            reasons.append(f"frame age {frame_age_ms:.0f}ms > {max_frame_age_ms:.0f}ms")
        if rgb_depth_skew_ms is None:
            reasons.append("rgb/depth skew unknown")
        elif rgb_depth_skew_ms > max_rgb_depth_skew_ms:
            if state == "FRESH":
                state = "DEGRADED"
            reasons.append(
                f"rgb/depth skew {rgb_depth_skew_ms:.0f}ms > {max_rgb_depth_skew_ms:.0f}ms"
            )
        reliability = {"FRESH": 1.0, "DEGRADED": 0.5, "STALE": 0.0}[state]
        if "rgb/depth skew unknown" in reasons and reliability == 1.0:
            reliability = 0.8  # honest haircut for unmeasurable skew
        return {
            "state": state,
            "reliability": reliability,
            "reasons": reasons,
            "consecutive_missing": consecutive_missing,
        }
