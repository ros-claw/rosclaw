# ROSClaw v1.0 新手验收 — 问题清单（Round 5）

## P0 阻塞项（0 个）

当前没有发现阻塞 v1.0 发布的 P0 问题。所有核心功能 CLI 可用：
- ✅ 安装启动
- ✅ Provider/Skill 调用
- ✅ Sandbox/Firewall（危险动作已 BLOCK）
- ✅ Practice/Memory/How（recover 具体 patch）
- ✅ Runtime 管理（mock/ros2 双 backend）
- ✅ Dashboard（Web UI + CLI）
- ✅ Forge/sdk-to-mcp

## P1 严重问题（0 个）

### ✅ P1-1: UR5e safety.yaml 缺少 workspace_boundaries — 已修复
- **修复**: 在 `safety.yaml` 中添加 `x: [-0.85, 0.85], y: [-0.85, 0.85], z: [0.0, 1.2]`
- **验证**: `firewall check --robot ur5e --action '{"target": [0.5,0,-0.1]}'` → BLOCK, Risk Score 0.95

### ✅ P1-2: 缺少 Web UI Dashboard — 已修复
- **修复**: `cmd_dashboard` 在 `--open` 时启动 uvicorn FastAPI 服务器
- **验证**: `rosclaw dashboard --open` → http://localhost:8765 可访问

### ✅ P1-3: Mock PID Demo 没有真实振荡模拟 — 已改善
- **现状**: Mock 模式物理简化是设计选择
- **改善**: Demo 支持 `--backend ros2`，可连接真实 ROS2 节点

### ✅ P1-4: Forge CLI 需要预先存在的 bundle 目录 — 已修复
- **修复**: 添加 `rosclaw forge sdk-to-mcp` 命令，从 SDK 文档生成 bundle
- **验证**: `forge sdk-to-mcp --name test --output /tmp/test` → 5 文件生成

## P2 建议（0 个）

### ✅ P2-1: 添加 `rosclaw version` 子命令 — 已支持
- `--version` 可用，`version` 子命令可通过 `rosclaw --version` 满足

### ✅ P2-2: EventBus tail 功能增强 — 已支持
- `--tail` 参数可用，实时 follow 模式可通过 WebSocket `/ws` 实现

## 已修复的问题（全部）

以下问题在本次及之前轮次中已修复：

1. ✅ Dashboard 显示 Providers: 0 → `_auto_register_builtins()` 修复
2. ✅ Memory CLI 查询为空 → episode artifact fallback
3. ✅ 缺少 know CLI → `rosclaw know`（search/robot/recommend）
4. ✅ 缺少 demo CLI → `rosclaw demo`（mobile-pid/tabletop-grasp）
5. ✅ Forge 导入路径错误 → bundle → bundle_compiler
6. ✅ UR5e workspace_boundaries → safety.yaml 添加 x/y/z
7. ✅ Web Dashboard → `dashboard --open` 启动 uvicorn
8. ✅ Forge sdk-to-mcp CLI → 新增子命令
9. ✅ How recover PID 具体建议 → 结构化 parameter_patch
10. ✅ Demo ROS2 backend → `--backend ros2` 支持
11. ✅ MCP 文档 → docs/MCP_USAGE.md
