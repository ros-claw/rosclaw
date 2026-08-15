"""Worker 控制协议与终止原因（十四审 PR-14.1，总纲 §3.2/§3.4）。

控制请求必须有 ACK：supervisor 在收到 control.ack 前只能显示
PAUSE_REQUESTED，不得乐观落 PAUSED。exit code 只是 Unix 表象，
终态原因唯一权威是 termination.json（TerminationV1）——进程来不及
写时按 SIGNAL_UNKNOWN → INTERRUPTED_RESUMABLE 处理，绝不直接 FAILED。
"""

from __future__ import annotations

from typing import Literal

from rosclaw.contracts.common import ContractModel

#: 权威终止原因枚举（总纲 §3.4）——Native Agent 只许展示 cause+证据，
#: 不得根据日志自由推测。
TERMINATION_CAUSES = (
    "COMPLETED",
    "USER_CANCELLED",
    "USER_PAUSED",
    "BUDGET_HARD_PAUSED",
    "AGENTD_SHUTDOWN",
    "PROVIDER_TRANSIENT",
    "PROVIDER_FATAL",
    "TOOL_FAILED",
    "DELIVERABLE_REJECTED",
    "WORKER_CRASH",
    "SIGNAL_UNKNOWN",
)

#: attempt_failed error_code → termination cause 的映射（worker 事件
#: 是次权威——termination.json 缺失时用于归类，绝不落回 "worker exited"）。
ERROR_CODE_CAUSES = {
    "MODEL_ERROR": "PROVIDER_FATAL",
    "MODEL_UNAVAILABLE": "PROVIDER_FATAL",
    "PROVIDER_TIMEOUT": "PROVIDER_TRANSIENT",
    "ADAPTER_PROTOCOL_ERROR": "WORKER_CRASH",
    "TOOL_CONTRACT_MISMATCH": "TOOL_FAILED",
    "DELIVERABLE_FAILED": "DELIVERABLE_REJECTED",
    "BLOCKED_PREFLIGHT": "TOOL_FAILED",
}

#: 控制动作（pause=安全暂停等 resume；cancel=用户取消唯一 CANCELLED 来源；
#: resume=同一会话继续）。
CONTROL_ACTIONS = ("pause", "resume", "cancel")


class ControlRequestV1(ContractModel):
    """supervisor → worker（stdin JSONL）。control_id 用于 ACK 对账。"""

    SCHEMA = "rosclaw.worker_control_request.v1"

    schema_version: Literal["rosclaw.worker_control_request.v1"] = (
        "rosclaw.worker_control_request.v1"
    )
    type: Literal["control.request"] = "control.request"
    control_id: str
    action: Literal["pause", "resume", "cancel"]
    mode: Literal["safe", "immediate"] = "safe"
    reason: str = ""


class ControlAckV1(ContractModel):
    """worker → supervisor（stdout WorkerEvent kind=control.ack）。

    state=PAUSED 表示模型循环已真实停止且进程存活等待 resume——
    只有此时 supervisor 才可落 PAUSED/BUDGET_PAUSED。"""

    SCHEMA = "rosclaw.worker_control_ack.v1"

    schema_version: Literal["rosclaw.worker_control_ack.v1"] = (
        "rosclaw.worker_control_ack.v1"
    )
    control_id: str
    state: Literal["PAUSE_REQUESTED", "PAUSED", "RUNNING", "CANCELLED"]
    session_id: str = ""
    detail: str = ""


class TerminationV1(ContractModel):
    """termination.json——worker 退出前原子落盘的权威终止原因。"""

    SCHEMA = "rosclaw.worker_termination.v1"

    schema_version: Literal["rosclaw.worker_termination.v1"] = (
        "rosclaw.worker_termination.v1"
    )
    cause: Literal[
        "COMPLETED",
        "USER_CANCELLED",
        "USER_PAUSED",
        "BUDGET_HARD_PAUSED",
        "AGENTD_SHUTDOWN",
        "PROVIDER_TRANSIENT",
        "PROVIDER_FATAL",
        "TOOL_FAILED",
        "DELIVERABLE_REJECTED",
        "WORKER_CRASH",
        "SIGNAL_UNKNOWN",
    ]
    detail: str = ""
    exit_code: int = 0
    session_file: str = ""
    at: str = ""
