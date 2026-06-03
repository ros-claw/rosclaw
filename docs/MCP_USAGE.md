# ROSClaw MCP 使用指南

## 什么是 MCP？

MCP (Model Context Protocol) 是 ROSClaw 与 Claude Code 等 AI 助手的标准通信协议。通过 MCP，Claude Code 可以直接调用 ROSClaw 的物理 AI 能力。

## 查看已注册的 MCP Tools

```bash
rosclaw provider list
```

已注册的 providers 自动暴露为 MCP tools：
- `llm` — 文本生成和任务规划
- `vlm` — 视觉感知和物体定位
- `vla` — 端到端控制
- `vln` — 移动机器人路径规划
- `skill` — 机器人动作执行
- `critic` — 成功/失败判断

## 通过 Claude Code 调用

在 Claude Code 中，直接描述你的物理 AI 任务：

```
让 UR5e 机械臂执行一个安全的 reach 动作到 [0.3, 0.2, 0.4]
```

Claude Code 会自动：
1. 查询 robot inspect 获取 UR5e 能力
2. 调用 firewall check 验证安全性
3. 调用 skill provider 执行动作
4. 通过 practice 记录 episode

## 生成新的 MCP Bundle

使用 Forge 将 SDK 文档转换为 MCP Server：

```bash
rosclaw forge sdk-to-mcp \
  --name my_sensor \
  --sdk-docs docs/my_sensor_sdk.md \
  --output bundles/my_sensor
```

生成内容：
- `mcp_server.py` — MCP Server stub
- `skill_manifest.json` — Skill 注册清单
- `provider_manifest.json` — Provider 路由清单
- `tests/` — 单元测试
- `README.md` — 使用文档

## 验证和安装 Bundle

```bash
rosclaw forge validate bundles/my_sensor
rosclaw forge install bundles/my_sensor --staging
```

## MCP Server 架构

```
Claude Code
    ↓ MCP Protocol
ROSClaw MCPHub
    ↓ EventBus
Providers / Skills / Sandbox / Runtime
    ↓
Physical Robot (ROS2 / MuJoCo / Mock)
```

## 故障排除

| 问题 | 解决 |
|------|------|
| MCP tools 不显示 | 确认 providers 已注册：`rosclaw provider list` |
| 调用失败 | 检查 firewall：`rosclaw firewall check --robot <robot> --action <action>` |
| 无 episode 记录 | 确认 practice recorder 健康：`rosclaw status` |
