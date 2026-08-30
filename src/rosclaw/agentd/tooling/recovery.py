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
    # R0-9（0826 体验审计 §5.R0-9）：渲染/资产类基础设施错误——
    # 模型重试预算为 0（换参数重试不会成功）。
    "RENDER_INPUT_MISSING": "渲染输入缺失——先完成轨迹 rollout（trace 不存在/无 states），不重试渲染",
    "RENDER_INPUT_DIGEST_MISMATCH": "trace 被改写——重跑 rollout 生成新 trace，不重试渲染",
    "RENDER_BACKEND_UNAVAILABLE": "渲染后端不可用（EGL/OSMesa/Xvfb）——安装/修复离屏后端，模型不重试",
    "RENDER_RESULT_MISSING": "渲染子进程未产出结果——内核已降级一次；检查渲染后端日志，模型不重试",
    "RENDER_RESULT_CORRUPT": "渲染结果损坏——内核侧故障，模型不重试",
    "RENDER_RESULT_INCOMPLETE": "渲染结果不完整——内核侧故障，模型不重试",
    "RENDER_FAILED": "渲染子进程失败——内核已降级一次；检查后端/依赖，模型不重试",
    "WORLD_ASSET_MISSING": "场景 world 资产不存在——换已支持的 world 或安装资产，模型不重试",
    "TOOL_ASSET_MISSING": "工具资产不存在——不得假装持笔；安装工具资产或去掉工具声明",
    # P0-6（0827 审计）：引用类稳定码——跨进程引用失败必须带码
    # 透传（不得包成 EXECUTOR_ERROR 丢语义）。
    "REF_NOT_FOUND": "引用不在共享 PlanStore——重新规划生成新引用，不重试同一 id",
    "REF_FORMAT_UNKNOWN": "引用格式不可解码（生产者/消费者 schema 不兼容）——用生产者当前 schema 重新生成",
    "REF_CONFORMANCE_FAILED": "PlanRef 生产者/消费者不共享存储——工具对已退出模型面，改用别的规划链",
}


def recovery_for(code: str) -> str:
    """稳定错误码 → 默认恢复动作；未知码 → 空（诚实）。"""
    return RECOVERY_REGISTRY.get(code, "")


__all__ = ["RECOVERY_REGISTRY", "recovery_for"]
