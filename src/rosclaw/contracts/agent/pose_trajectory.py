"""PoseTrajectorySpecV1（WP-5，0823 审计冻结序列 §五）。

轨迹不再是纯位置点列：每个航点是 SE(3) 位姿（position_m +
orientation_xyzw 单位四元数）+ 语义 kind（transit/approach/
contact/lift）。contact_plane 与 tool_frame 是契约的一等字段——
接触平面法向与工具坐标系不再是隐含约定。approach/lift 段显式
建模在规格里，执行器不得临时拼接。
"""

from __future__ import annotations

import math
from typing import ClassVar, Literal

from pydantic import Field, field_validator

from rosclaw.contracts.common import ContractModel

#: 航点语义类别（冻结集合——执行器按 kind 区分跟踪/转场行为）。
WAYPOINT_KINDS = ("transit", "approach", "contact", "lift")


class ContactPlaneV1(ContractModel):
    """接触平面：法向 + 沿法向的偏移（米）。"""

    SCHEMA: ClassVar[str] = "rosclaw.contact_plane.v1"

    schema_version: Literal["rosclaw.contact_plane.v1"] = "rosclaw.contact_plane.v1"
    normal_xyz: list[float] = Field(min_length=3, max_length=3)
    offset_m: float = 0.0

    @field_validator("normal_xyz")
    @classmethod
    def _unit_normal(cls, value: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in value))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"contact plane normal 非单位向量: {value}")
        return value


class PoseWaypointV1(ContractModel):
    """SE(3) 航点：位置 + 朝向（xyzw 单位四元数）+ 语义 kind。"""

    SCHEMA: ClassVar[str] = "rosclaw.pose_waypoint.v1"

    schema_version: Literal["rosclaw.pose_waypoint.v1"] = "rosclaw.pose_waypoint.v1"
    position_m: list[float] = Field(min_length=3, max_length=3)
    orientation_xyzw: list[float] = Field(min_length=4, max_length=4)
    kind: str = Field(min_length=1)

    @field_validator("orientation_xyzw")
    @classmethod
    def _unit_quaternion(cls, value: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in value))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"orientation 非单位四元数: {value}")
        return value

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in WAYPOINT_KINDS:
            raise ValueError(
                f"unknown waypoint kind {value!r} (frozen: {WAYPOINT_KINDS})"
            )
        return value


class PoseTrajectorySpecV1(ContractModel):
    """SE(3) 位姿轨迹规格（内容寻址，digest 可反查）。"""

    SCHEMA: ClassVar[str] = "rosclaw.pose_trajectory_spec.v1"
    HASH_PREFIX: ClassVar[str] = "pose_trajectory_spec"

    schema_version: Literal["rosclaw.pose_trajectory_spec.v1"] = (
        "rosclaw.pose_trajectory_spec.v1"
    )
    #: 位置所在的世界/基座坐标系（如 "world"）。
    frame_id: str = Field(min_length=1)
    #: 工具坐标系（MJCF site 名，如 "attachment_site"）。
    tool_frame: str = Field(min_length=1)
    contact_plane: ContactPlaneV1
    waypoints: list[PoseWaypointV1] = Field(min_length=1)
    #: 内容寻址 digest（sha256:…，对 waypoints 规范化 JSON）。
    digest: str = Field(min_length=1)


__all__ = [
    "WAYPOINT_KINDS",
    "ContactPlaneV1",
    "PoseTrajectorySpecV1",
    "PoseWaypointV1",
]
