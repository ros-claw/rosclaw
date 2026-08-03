"""Shared test constants for agentd suites."""

from __future__ import annotations

import os

#: 与 operator.sock 的 SO_PEERCRED 语义一致的本地主体（CI 运行 uid 可能
#: 不是 1000）。凡走完整审批-验证链路的测试必须使用它而不是硬编码 uid。
LOCAL_PRINCIPAL = f"user:local:{os.getuid()}"
