"""Camera pose contract + intrinsics binding (Physical Evolution Lab §4.2, PR-PE-3).

Every visual conclusion is valid only under one camera pose.  The
contract binds:

* ``camera_pose_id`` — operator-named pose (e.g. ``front_v1``);
* ``intrinsic_hash`` — from the DEVICE's real intrinsics (fx/fy/ppx/ppy
  + distortion model), read off the D435i — never hand-copied;
* ``extrinsic_hash`` / ``mount_hash`` — "unmeasured" until an operator
  measures them (disclosed, and still hashed so a changed declaration
  is a changed hash);
* ``roi_hash`` / ``lighting_profile_id``.

Moving the camera invalidates the contract: ``camera_pose_hash``
changes and every visual calibration produced under the old hash stops
being valid evidence (v3 §4.2: 移动相机后必须重新生成 Camera Pose
Hash，旧视觉标定不得继续作为有效证据).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rosclaw.practice.physical_observation import canonical_hash

CONTRACT_VERSION = "rosclaw.camera_pose.v1"


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    distortion_model: str = "unknown"
    coeffs: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
            "distortion_model": self.distortion_model,
            "coeffs": list(self.coeffs),
        }

    @property
    def intrinsic_hash(self) -> str:
        return canonical_hash(self.to_dict(), prefix="intr")

    def deproject(self, u: float, v: float, depth_m: float) -> tuple[float, float, float]:
        """Pinhole deprojection (no distortion correction — disclosed;
        D435i depth is factory-rectified so this is the honest model)."""
        x = (u - self.ppx) * depth_m / self.fx
        y = (v - self.ppy) * depth_m / self.fy
        return (x, y, depth_m)


@dataclass(frozen=True)
class CameraPoseContract:
    camera_pose_id: str
    camera_id: str
    intrinsics: CameraIntrinsics
    extrinsic: dict[str, Any] = field(default_factory=dict)
    mount: dict[str, Any] = field(default_factory=dict)
    roi: dict[str, int] = field(default_factory=dict)  # x0,y0,x1,y1 in pixels
    lighting_profile_id: str = "unmeasured"

    @property
    def extrinsic_hash(self) -> str:
        return canonical_hash(self.extrinsic or {"status": "unmeasured"}, prefix="extr")

    @property
    def mount_hash(self) -> str:
        return canonical_hash(self.mount or {"status": "unmeasured"}, prefix="mount")

    @property
    def roi_hash(self) -> str:
        return canonical_hash(self.roi or {"status": "full_frame"}, prefix="roi")

    @property
    def camera_pose_hash(self) -> str:
        return canonical_hash(
            {
                "camera_pose_id": self.camera_pose_id,
                "camera_id": self.camera_id,
                "intrinsic_hash": self.intrinsics.intrinsic_hash,
                "extrinsic_hash": self.extrinsic_hash,
                "mount_hash": self.mount_hash,
                "roi_hash": self.roi_hash,
                "lighting_profile_id": self.lighting_profile_id,
            },
            prefix="campose",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "camera_pose_id": self.camera_pose_id,
            "camera_id": self.camera_id,
            "intrinsics": self.intrinsics.to_dict(),
            "intrinsic_hash": self.intrinsics.intrinsic_hash,
            "extrinsic": self.extrinsic or {"status": "unmeasured"},
            "extrinsic_hash": self.extrinsic_hash,
            "mount": self.mount or {"status": "unmeasured"},
            "mount_hash": self.mount_hash,
            "roi": self.roi or {"status": "full_frame"},
            "roi_hash": self.roi_hash,
            "lighting_profile_id": self.lighting_profile_id,
            "camera_pose_hash": self.camera_pose_hash,
        }

    def validate_against(self, other: CameraPoseContract) -> list[str]:
        """What changed between two poses (evidence invalidation report)."""
        changed: list[str] = []
        if self.intrinsics.intrinsic_hash != other.intrinsics.intrinsic_hash:
            changed.append("intrinsics")
        if self.extrinsic_hash != other.extrinsic_hash:
            changed.append("extrinsic")
        if self.mount_hash != other.mount_hash:
            changed.append("mount")
        if self.roi_hash != other.roi_hash:
            changed.append("roi")
        if self.lighting_profile_id != other.lighting_profile_id:
            changed.append("lighting_profile")
        return changed

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CameraPoseContract:
        intr = raw.get("intrinsics") or {}
        return cls(
            camera_pose_id=str(raw.get("camera_pose_id") or ""),
            camera_id=str(raw.get("camera_id") or ""),
            intrinsics=CameraIntrinsics(
                width=int(intr.get("width") or 0),
                height=int(intr.get("height") or 0),
                fx=float(intr.get("fx") or 0.0),
                fy=float(intr.get("fy") or 0.0),
                ppx=float(intr.get("ppx") or 0.0),
                ppy=float(intr.get("ppy") or 0.0),
                distortion_model=str(intr.get("distortion_model") or "unknown"),
                coeffs=tuple(float(c) for c in (intr.get("coeffs") or ())),
            ),
            extrinsic=dict(raw.get("extrinsic") or {}),
            mount=dict(raw.get("mount") or {}),
            roi={k: int(v) for k, v in (raw.get("roi") or {}).items()} if raw.get("roi") else {},
            lighting_profile_id=str(raw.get("lighting_profile_id") or "unmeasured"),
        )


def intrinsics_from_pyrealsense(profile: Any, stream_type: Any) -> CameraIntrinsics:
    """Read REAL intrinsics off a running pyrealsense2 pipeline profile."""
    stream = profile.get_stream(stream_type).as_video_stream_profile()
    intr = stream.get_intrinsics()
    return CameraIntrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        ppx=float(intr.ppx),
        ppy=float(intr.ppy),
        distortion_model=str(intr.model).split(".")[-1],
        coeffs=tuple(float(c) for c in intr.coeffs),
    )


def intrinsics_from_json(path: str) -> CameraIntrinsics:
    """Load device intrinsics captured earlier by the camera-env probe
    (the repo venv has no pyrealsense2 — probes run in the task env)."""
    with open(path, encoding="utf-8") as handle:
        raw = json.loads(handle.read())
    return CameraIntrinsics(
        width=int(raw["width"]),
        height=int(raw["height"]),
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        ppx=float(raw["ppx"]),
        ppy=float(raw["ppy"]),
        distortion_model=str(raw.get("distortion_model") or "unknown"),
        coeffs=tuple(float(c) for c in (raw.get("coeffs") or ())),
    )
