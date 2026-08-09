"""SimActionChannel（PR-12）：SIMULATION 模式下的 SIM 物理权威。

与 rosclawd（REAL 唯一物理权威）镜像的 SIM 执行通道：
- 只在 EXACT_ACTION grant 被 Broker 验证消费之后调用；
- 直接以独立 MCP 客户端调用 SIM 身体的动作工具——此路径不属于
  ToolCatalog，模型永远无法触达；
- 产出 SIMULATED receipt：evidence_domain=simulation、
  usable_for_real_execution=False、acoustic/physical 观测缺失时诚实声明
  只证明驱动执行。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rosclaw.contracts.common import ValidationError, new_id


class SimActionError(ValidationError):
    pass


@dataclass
class SimActionOutcome:
    action_id: str
    final_state: str  # COMPLETED | FAILED
    receipt: dict[str, Any] = field(default_factory=dict)

    @property
    def receipt_id(self) -> str:
        return str(self.receipt.get("receipt_id") or "")


class SimActionChannel:
    """对一个 SIM MCP server 的动作执行通道（每次调用独立 stdio 会话）。"""

    def __init__(
        self,
        *,
        command: str,
        args: tuple[str, ...],
        name: str = "limo-sim",
        client=None,
    ) -> None:
        self._command = command
        self._args = args
        self._name = name
        #: 与观测路径共享的持久会话（有状态 SIM 身体必须同进程）。
        self._client = client

    async def execute(
        self,
        *,
        capability_id: str,
        arguments: dict[str, Any],
        grant_id: str,
        mode: str = "SIMULATION",
    ) -> SimActionOutcome:
        if mode != "SIMULATION":
            raise SimActionError(f"SimActionChannel only executes SIMULATION, got {mode}")
        tool_name = capability_id
        try:
            result_text = await self._call_tool(tool_name, arguments)
            result = json.loads(result_text)
        except Exception as exc:  # noqa: BLE001 - 诚实失败
            raise SimActionError(
                f"sim executor {self._name} failed for {tool_name}: {type(exc).__name__}: {exc}"
            ) from exc
        action_id = new_id("act")
        # 五审 P0-5E：receipt 是独立证据对象——独立 receipt_id，
        # 不得与 action_id 同值（否则"receipt 存在"无法独立于
        # "action 被派发"被验证）。
        receipt_id = new_id("rcpt")
        # 五审 P0-5E：domain 级失败（ok=false / driver=failed / error 字段）
        # 不得报 COMPLETED——transport 没错 ≠ 动作成功。success predicate：
        # 只有明确 ok=true 或无否定字段的结果才算完成。
        domain_failed = bool(
            result.get("ok") is False
            or result.get("driver") == "failed"
            or result.get("error")
            or result.get("status") in ("failed", "error", "FAILED", "ERROR")
        )
        if domain_failed:
            raise SimActionError(
                f"sim executor {self._name} domain failure for {tool_name}: "
                f"{json.dumps(result, ensure_ascii=False)[:300]}"
            )
        receipt = {
            "receipt_id": receipt_id,
            "action_id": action_id,
            "capability_id": capability_id,
            "arguments": arguments,
            "grant_id": grant_id,
            "final_state": "COMPLETED",
            "trust_level": "SIMULATED",
            "evidence_domain": "simulation",
            "evidence_level": "DRIVER_EXECUTED",
            "usable_for_real_execution": False,
            "executor": f"mcp:{self._name}",
            "executor_result": result,
            # §18.4 诚实验证：无声学/物理观测时只证明驱动执行。
            "physical_effect_proven": bool(result.get("acoustic_observation")),
        }
        return SimActionOutcome(action_id=action_id, final_state="COMPLETED", receipt=receipt)

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self._client is not None:
            return await self._client.call_tool(tool_name, arguments)
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self._command, args=list(self._args))
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
        if result.isError:
            text = " ".join(getattr(b, "text", "") for b in result.content).strip()
            raise SimActionError(f"sim tool {tool_name} error: {text or 'unknown'}")
        return "".join(getattr(b, "text", "") for b in result.content)
