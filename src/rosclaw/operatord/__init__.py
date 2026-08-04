"""rosclaw-operatord — 独立的 Operator 授权进程（审计 P0-01）。

拓扑与红线：

* operatord 是唯一持有 **operator enrollment key** 的用户态进程；
  agentd、Worker、普通 curl、同机恶意进程都没有这把 key。
* daemon `proposal.decide` 的 ACL：daemon 服务 UID **或** 持有效
  enrollment proof 的调用方——agentd 代码里不再有任何 decide 路径。
* REAL 决定还需要 human-presence 信号（前台 /dev/tty 按键）。
* 同 UID 一体运行只能标记 `DEV_SIM_ONLY`；REAL 启动检查 fail closed。
"""

from rosclaw.operatord.enrollment import (
    DEV_SIM_ONLY_LABEL,
    EnrollmentError,
    OperatorEnrollment,
    load_or_create_enrollment,
    sign_decision_proof,
    verify_decision_proof,
)

__all__ = [
    "DEV_SIM_ONLY_LABEL",
    "EnrollmentError",
    "OperatorEnrollment",
    "load_or_create_enrollment",
    "sign_decision_proof",
    "verify_decision_proof",
]
