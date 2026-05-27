# ROSClaw v1.0 端到端测试发现问题

## 测试概况
- 测试时间: 2025-05-27
- 测试方式: 从用户角度从零开始使用 ROSClaw
- 测试结果: 8/8 通过（修复后）
- 发现问题: 7 个 API 不一致/文档缺失问题

## 发现的问题

### 1. AgentRuntime 类名不存在
- **位置**: `rosclaw.agent_runtime.__init__`
- **问题**: 文档提到 `AgentRuntime`，实际导出的是 `AgentContext`
- **影响**: 用户按文档导入会失败
- **修复**: 统一类名或更新文档

### 2. EUrdfParser vs EURDFParser 大小写不一致
- **位置**: `rosclaw.e_urdf.__init__`
- **问题**: 文档用 `EUrdfParser`，实际是 `EURDFParser`
- **影响**: 用户按文档导入会失败
- **修复**: 统一命名

### 3. SQLiteSeekDB 类名不存在
- **位置**: `rosclaw.memory.__init__`
- **问题**: 文档提到 `SQLiteSeekDB`，实际是 `SeekDBSQLiteClient`
- **影响**: 用户按文档导入会失败
- **修复**: 统一类名

### 4. PraxisEventType 枚举不存在
- **位置**: `rosclaw.core.types`
- **问题**: 测试期望 `PraxisEventType.MOVE`，实际只有字符串 `event_type`
- **影响**: API 类型安全缺失
- **修复**: 添加 PraxisEventType 枚举或更新文档

### 5. MCPHub 构造函数需要 event_bus
- **位置**: `rosclaw.agent_runtime.mcp_hub`
- **问题**: 文档未说明构造函数需要 `event_bus` 参数
- **影响**: 用户直接 `MCPHub()` 会失败
- **修复**: 更新文档或添加默认参数

### 6. FirewallValidator 构造函数参数未文档化
- **位置**: `rosclaw.firewall.validator`
- **问题**: 需要 `robot_model`, `event_bus`, `safety_level` 三个参数
- **影响**: 用户不知道需要这些依赖
- **修复**: 更新文档和示例

### 7. UnifiedTimeline 构造函数参数未文档化
- **位置**: `rosclaw.practice.timeline`
- **问题**: 需要 `robot_id`, `event_bus` 参数
- **影响**: 用户直接 `UnifiedTimeline()` 会失败
- **修复**: 更新文档和示例

### 8. SkillRegistry.register() 参数类型不匹配
- **位置**: `rosclaw.skill_manager.registry`
- **问题**: 需要 `SkillEntry` 对象，不是字符串+lambda
- **影响**: 用户按直觉使用会失败
- **修复**: 更新文档和示例

### 9. SeekDB API 与文档不符
- **位置**: `rosclaw.memory.seekdb_client`
- **问题**: 文档提到 `store_experience`/`search_similar`，实际是 `insert`/`query`
- **影响**: 用户按文档使用会失败
- **修复**: 统一 API 或更新文档

## 建议
1. 创建统一的 API 文档（docs/API_REFERENCE.md）
2. 所有公共 API 添加类型提示和文档字符串
3. 提供完整的用户使用示例
4. 端到端测试纳入 CI
