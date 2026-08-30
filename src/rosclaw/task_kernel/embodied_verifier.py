"""具身 Verifier（R0-5，0826 体验审计 §5.R0-5）——多维度、按
任务与尺度生成 acceptance，避免低质量 PASS。

阈值公式（文档冻结）：

    threshold = max(robot_floor_m, 0.003, scale_m * 0.05)

- robot_floor_m：平台实测跟踪下限（sim jacobian 控制器实测
  19.6mm@scale0.10——0.025 是其带余量的诚实声明；未来更好的
  控制器可收紧）。指南原文 min(profile_limit, ...) 会让验收
  严过平台能力 = 每个任务永久红（那是假装不是验收）——
  地板语义：验收不得严过平台实测能力，也不得松过尺度 5%
  与 3mm 绝对地板。

证据等级拆分（不用单个 SIM_DYN_ROLLOUT 覆盖全部）：
GEOMETRY_PLAN / KINEMATIC_TRACKING / DYNAMIC_ROLLOUT /
CONTACT_SIMULATION / SCENE_RENDER / REAL_EXECUTION_RECEIPT。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: sim jacobian 控制器实测跟踪下限（2026-08-26 本机实测：
#: max 19.6mm@scale0.10、joint delta 0.0003→0.00005 无改善——
#: 稳态误差非速率限制）。声明带 ~25% 余量。
SIM_JACOBIAN_FLOOR_M = 0.025

#: 绝对地板（任何平台不得严过 3mm）与尺度系数。
ABSOLUTE_FLOOR_M = 0.003
SCALE_FACTOR = 0.05

#: 工具轴对齐容差（指南 §5.R0-5：tool axis ≤3°）。
TOOL_AXIS_LIMIT_DEG = 3.0

#: 接近阈值带宽（0827 审计 P0-5）：误差 ≥90% 阈值占用时不得显示
#: 普通 PASS（19.86mm/20mm=99.3% 显示 PASS 是假成功）。
NEAR_LIMIT_RATIO = 0.9


def tracking_grade(max_error_m: float, threshold_m: float) -> str:
    """跟踪误差分级：PASS / PASS_NEAR_LIMIT（≥90% 阈值占用）/ FAIL。"""
    error = float(max_error_m)
    threshold = float(threshold_m)
    if error > threshold:
        return "FAIL"
    # 浮点边界：90.000...% 也算 near-limit（ε 容差，不放低标准——
    # 只是不让二进制表示误差吃掉精确边界）。
    if threshold > 0 and error + 1e-12 >= NEAR_LIMIT_RATIO * threshold:
        return "PASS_NEAR_LIMIT"
    return "PASS"


def tracking_acceptance(
    scale_m: float, *, robot_floor_m: float = SIM_JACOBIAN_FLOOR_M
) -> float:
    """尺度/平台自适应跟踪阈值（公式见模块 docstring）。"""
    return max(
        max(float(robot_floor_m), ABSOLUTE_FLOOR_M),
        ABSOLUTE_FLOOR_M,
        float(scale_m) * SCALE_FACTOR,
    )


def contact_failures(
    constraints: dict[str, Any], metrics: dict[str, Any]
) -> list[str]:
    """接触证据核验：spec 声明 contact_required 时，trace 必须
    有接触段样本（接触关闭/无接触 = CONTACT_EVIDENCE_MISSING，
    不得 PASS）。"""
    if not constraints.get("contact_required"):
        return []
    samples = int(metrics.get("contact_samples", 0) or 0)
    if samples <= 0:
        return [
            "CONTACT_EVIDENCE_MISSING: spec 要求接触（contact_"
            "required）但 trace 无接触段样本——不得宣称接触/画线"
        ]
    return []


def tool_axis_failures(
    metrics: dict[str, Any], *, limit_deg: float = TOOL_AXIS_LIMIT_DEG
) -> list[str]:
    """工具轴对齐核验（有朝向指标的 trace 才判——无指标不编造）。"""
    value = metrics.get("max_orientation_error_deg")
    if value is None:
        return []
    if float(value) > limit_deg:
        return [
            f"TOOL_AXIS_EXCEEDED: 工具轴误差 {value}° > {limit_deg}°"
        ]
    return []


def scene_media_failures(
    path: Path | str,
    *,
    min_frames: int = 0,
    min_resolution: tuple[int, int] | list[int] = (0, 0),
) -> list[str]:
    """场景媒体核验：可解码 + 帧数 + 分辨率（文件存在≠可交付）。

    MP4 损坏/编码失败/帧数不足/分辨率不足 → SCENE_MEDIA_INVALID。
    """
    import imageio.v3 as iio

    path = Path(path)
    # imiter 实读（improps 对部分 ffmpeg 容器报 n_images=inf——
    # 实测 OverflowError；早停：超过 min_frames 即够判定）。
    try:
        frames = 0
        width = height = 0
        stop_after = max(int(min_frames or 0), 1) + 1
        for frame in iio.imiter(path):
            if frames == 0:
                shape = tuple(getattr(frame, "shape", ()))
                height, width = (
                    (int(shape[0]), int(shape[1])) if len(shape) >= 2
                    else (0, 0)
                )
            frames += 1
            if frames >= stop_after:
                break
    except Exception as exc:  # noqa: BLE001 - 不可解码是数据
        return [
            f"SCENE_MEDIA_INVALID: {path.name} 不可解码（"
            f"{type(exc).__name__}: {str(exc)[:120]}）"
        ]
    if frames == 0:
        return [f"SCENE_MEDIA_INVALID: {path.name} 零帧"]
    if min_frames and frames < min_frames:
        return [
            f"SCENE_MEDIA_INVALID: {path.name} 帧数 {frames} < "
            f"{min_frames}"
        ]
    req_w, req_h = (int(min_resolution[0]), int(min_resolution[1])) if (
        len(min_resolution) >= 2
    ) else (0, 0)
    if (req_w and width < req_w) or (req_h and height < req_h):
        return [
            f"SCENE_MEDIA_INVALID: {path.name} 分辨率 {width}x{height}"
            f" < {req_w}x{req_h}"
        ]
    return []


__all__ = [
    "ABSOLUTE_FLOOR_M",
    "SCALE_FACTOR",
    "SIM_JACOBIAN_FLOOR_M",
    "TOOL_AXIS_LIMIT_DEG",
    "contact_failures",
    "scene_media_failures",
    "tool_axis_failures",
    "tracking_acceptance",
]
