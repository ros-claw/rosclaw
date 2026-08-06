"""rosclaw-operatord — 独立的 Operator 授权进程（二次复核 R1/R2）。

拓扑与红线：

* operatord 是唯一持有 **operator Ed25519 私钥** 的用户态进程；
  rosclawd 只保存公钥（持久化 registry），双方不再共享对称秘密。
* daemon `proposal.decide` 的唯一凭证是 Ed25519 签名的
  OperatorDecisionProofV1（内嵌 daemon 签发的同一个 challenge）——
  没有 daemon-UID 直通，同 UID 测试不再绕过 proof。
* REAL/SHADOW 决定需要真实前台终端 Y/N（/dev/tty 显示不可变卡片 +
  请求方前台进程组校验）；默认/超时/EOF 一律 deny。
* 同 UID 一体运行只能标记 `DEV_SIM_ONLY`；REAL 启动检查 fail closed。
"""

from rosclaw.operatord.enrollment import (
    DEV_SIM_ONLY_LABEL,
    EnrollmentError,
    OperatorIdentity,
    load_identity,
    load_or_create_identity,
)

__all__ = [
    "DEV_SIM_ONLY_LABEL",
    "EnrollmentError",
    "OperatorIdentity",
    "load_identity",
    "load_or_create_identity",
]
