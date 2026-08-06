"""Pi Session ↔ Mission 绑定合约（重构规格 §12，PR-PNA-1）。

Pi Session 是认知与对话会话；ROSClaw Mission 是具身任务与物理事实
容器。绑定记录谁是谁的认知前端；lease 保证一个 Mission 同时只有一个
主认知 writer（防双写）。
"""

from __future__ import annotations

from typing import Literal

from rosclaw.contracts.common import ContractModel


class PiSessionBindingV1(ContractModel):
    SCHEMA = "rosclaw.pi_session_binding.v1"

    schema_version: Literal["rosclaw.pi_session_binding.v1"] = "rosclaw.pi_session_binding.v1"
    binding_id: str
    pi_session_id: str
    pi_session_path: str = ""
    mission_id: str
    body_id: str = ""
    execution_mode: Literal["SIMULATION", "SHADOW", "REAL"] = "SIMULATION"
    created_at: str
    created_by: str
    parent_binding_id: str | None = None
    source_mission_id: str | None = None
    status: Literal["ACTIVE", "DETACHED", "NEEDS_BINDING"] = "ACTIVE"
    binding_revision: int = 1


class PiSessionLeaseV1(ContractModel):
    SCHEMA = "rosclaw.pi_session_lease.v1"

    schema_version: Literal["rosclaw.pi_session_lease.v1"] = "rosclaw.pi_session_lease.v1"
    lease_id: str
    mission_id: str
    pi_session_id: str
    owner_pid: int
    owner_uid: int
    host_id: str = ""
    lease_token_hash: str
    issued_at: str
    expires_at: str
    heartbeat_at: str
