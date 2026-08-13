"""仿真执行回执（十二审 PR-12.6，总纲 §6.2）。

`simulation_run` 任务的标准 receipt——动画只是 trace 的派生物；
只有带真实后端 trace + receipt 的任务才能标 SIMULATED。
"""

from __future__ import annotations

from typing import Any, Literal

from rosclaw.contracts.common import ContractModel


class SimRunReceiptV1(ContractModel):
    """仿真运行回执：后端、轨迹 hash、步数、验证结论。"""

    SCHEMA = "rosclaw.sim_run_receipt.v1"
    schema_version: Literal["rosclaw.sim_run_receipt.v1"] = "rosclaw.sim_run_receipt.v1"
    body_id: str
    backend: str  # mujoco / command-replay / ...
    scene: str = ""
    trace_hash: str = ""
    trace_steps: int = 0
    started_at: str = ""
    finished_at: str = ""
    validation: dict[str, Any] = {}  # verifier 结果（端点误差/RMSE/…）
    # 三层证据语言（Gate H）：command-replay 的沙盒只能 COMMAND_REPLAY。
    evidence_level: str = "COMMAND_REPLAY"  # COMMAND_REPLAY | SIMULATED
    artifacts: list[str] = []
