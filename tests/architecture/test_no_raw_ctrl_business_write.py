"""ControlSchema 统一路由架构锁定（MH20-B，讨论总纲 §7）。

**所有执行路径只经 Canonical ControlSchema 写 ctrl。** 业务模块
（interaction executors 等）不得直接写 `data.ctrl[...]`——只有
底层 control adapter（rollout plan 应用层 / control mapper /
backend 指定的 ctrl 写入函数）允许 raw 写。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: raw ctrl 写只允许出现在这些 adapter 模块内（底层控制适配层）。
_ALLOWED_MODULES = (
    "src/rosclaw/sim/backends/mujoco/rollout.py",  # plan 应用层（hold/ctrl_series/targets）
    "src/rosclaw/sim/backends/mujoco/control.py",  # ControlMapper（唯一业务通道）
    "src/rosclaw/sim/backends/mujoco/backend.py",  # backend 内部 ctrl 编排
)

#: 扫描的业务模块（执行路径，禁止 raw 写）。
_BUSINESS_MODULES = (
    "src/rosclaw/sim/backends/mujoco/interact.py",
)

_RAW_CTRL_RE = re.compile(r"data\.ctrl\[")


def test_no_raw_ctrl_write_in_business_modules() -> None:
    for rel in _BUSINESS_MODULES:
        text = (REPO / rel).read_text(encoding="utf-8")
        matches = _RAW_CTRL_RE.findall(text)
        assert not matches, (
            f"{rel} 出现 {len(matches)} 处 data.ctrl[...] raw 写——"
            "执行路径必须走 ControlMapper（Canonical ControlSchema）"
        )


def test_allowed_adapter_modules_exist() -> None:
    """白名单模块必须存在（防架构漂移——改名要同步改这里）。"""
    for rel in _ALLOWED_MODULES:
        assert (REPO / rel).is_file(), f"adapter 模块缺失: {rel}"
