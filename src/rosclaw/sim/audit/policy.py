"""Audit 阈值策略（PR-MH4，规格 §17.3）。

阈值必须是 named policy，不散落 magic number。数值继承
Text2Mujoco 从真实失败补出的经验（showcase/model_audit.py L61-70）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AuditPolicy:
    """审计阈值集合（strict 默认）。"""

    #: 初始位姿穿透阈值（0.1 mm，solver 尚未介入）。
    static_penetration_m: float = -1e-4
    #: 运行期穿透阈值（1 mm，软接触容差）。
    run_penetration_m: float = -1e-3
    #: 自重叠阈值（1 mm）。
    self_overlap_m: float = -1e-3
    #: 连杆缝隙可见阈值（5 mm）。
    continuity_gap_m: float = 5e-3
    #: 滑轨伺服下垂阈值（2 mm）。
    servo_sag_m: float = 2e-3
    #: 铰链伺服下垂阈值（1 度）。
    servo_sag_rad: float = math.radians(1.0)
    #: marker 悬浮上限（3 mm）。
    marker_clearance_m: float = 3e-3
    #: marker 埋入上限（1 mm）。
    marker_burial_m: float = 1e-3
    #: 可见交互 marker 的 site group（Text2Mujoco 约定）。
    marker_group: int = 2
    #: 伺服保持时长。
    servo_hold_s: float = 2.0
    #: 扫描时长（continuity/sequence 默认窗口）。
    sweep_s: float = 1.0
    #: 速度发散阈值。
    max_qvel: float = 1e3
    #: 能量爆炸倍数阈值。
    energy_explosion_factor: float = 100.0
    #: 能量爆炸绝对阈值（防零除）。
    energy_explosion_abs: float = 1.0
    #: 执行器饱和步数占比阈值（A09）。
    saturation_ratio: float = 0.2
    #: solver/timestep 敏感性相对偏差阈值（A23/A24）。
    sensitivity_rel: float = 1e-3


STRICT_POLICY = AuditPolicy()
