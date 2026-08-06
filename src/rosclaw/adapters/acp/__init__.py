"""ACP adapter（批次 G §9.1）：让成熟编辑器客户端接入同一个 Native Agent。

边界（§12.5）：
- ACP client 是 UI client，不是 Agent Engine——Native Agent 仍运行
  AgentLoop；
- ACP generic permission 不等于 rosclawd Permit——物理授权仍由
  Operator Broker + rosclawd 验证；客户端不支持精确卡片时只回
  approval id/URL，绝不退化为聊天口令或自动批准。
"""

from rosclaw.adapters.acp.mapper import event_to_session_update

__all__ = ["event_to_session_update"]
