"""Error Recovery Registry（PR-N6C，调整方案 §五.N6C）——结构化恢复梯度。

稳定错误码 → 默认恢复动作的唯一事实源。envelope 的 recovery 字段、
提示词恢复指引、TUI 错误卡都从本注册表投影——不在各处临时拼。

未知码 → 空字符串（诚实无建议，不编造）。
"""

from __future__ import annotations

#: 方案 §五.N6C 表格（含本仓既有错误码的对应）。
RECOVERY_REGISTRY: dict[str, str] = {
    "INVALID_ARGUMENTS": "读取该能力的 input_schema 后修正参数一次",
    "CAPABILITY_UNKNOWN": "inspect capability registry 查精确 ID——不猜名称",
    "EFFECT_UNRESOLVABLE": "inspect capability registry 查精确 ID——不猜名称",
    "CAPABILITY_NOT_FOUND": "inspect registry——不猜名称",
    "CAPABILITY_SNAPSHOT_CHANGED": "重新获取 pi.capability.snapshot 并按新工具面重新规划一次",
    "RESOURCE_PROVENANCE_MISSING": "经 Resource Resolver 解析权威资源——不从 / 全盘搜索",
    "RESOURCE_ID_MISMATCH": "经 Resource Resolver 解析权威资源——不从 / 全盘搜索",
    "RESOURCE_DIGEST_MISMATCH": "经 Resource Resolver 解析权威资源——不从 / 全盘搜索",
    "NON_CANONICAL_RESOURCE": "经 Resource Resolver 解析权威资源——不从 / 全盘搜索",
    "RESOURCE_NOT_FOUND": "经 Resource Resolver 解析——不从 / 全盘搜索",
    "RUNTIME_NOT_READY": "Runtime Manager 自动准备中——等待或诚实阻塞并说明缺什么",
    "TRANSPORT_UNREACHABLE": "检查官方本地执行器/降级路径——不重试同一通道",
    "INVALID_CAPABILITY_OUTPUT": "能力实现故障——进入开发修复（读实现、修、测），不重试同一调用",
    "OUTPUT_SCHEMA_INVALID": "能力实现故障——进入开发修复（读实现、修、测），不重试同一调用",
    "ACCEPTANCE_FAILED": "按结构化失败项修复同一 revision（不新开任务）",
    "SAFETY_DENIED": "不重复调用——向用户给出原因",
    "WAITING_APPROVAL": "让出回合——等待审批事件恢复，不轮询",
}


def recovery_for(code: str) -> str:
    """稳定错误码 → 默认恢复动作；未知码 → 空（诚实）。"""
    return RECOVERY_REGISTRY.get(code, "")


__all__ = ["RECOVERY_REGISTRY", "recovery_for"]
