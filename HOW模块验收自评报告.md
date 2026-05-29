# ROSClaw v1.0 模块验收自评 — HOW（在线死锁突破器）

## 评估日期: 2026-05-29
## 负责人: HOW Module Owner
## 当前Commit: bf75095 + 本地 Recovery Loop / Integration API 实现

---

## 一、P0 阻塞项自评

| P0项 | 当前状态 | 证据/说明 |
|------|----------|-----------|
| L0: 从零安装启动 | □ 通过 □ 未通过 ☑ 部分 | `rosclaw doctor` 能运行，但缺少 `install.sh`，e-URDF-Zoo 路径需手动软链接。非 HOW 负责。 |
| L1: Claude Code MCP接入 | □ 通过 □ 未通过 ☑ 部分 | MCPHub 已注册 `get_recovery_strategy` tool，Claude Code 可查 heuristic 规则。但**真实失败场景调用 How 恢复未闭环验证**。 |
| L2: Event Bus真实工作 | □ 通过 □ 未通过 ☑ 部分 | Runtime 已订阅 3 个 failure topic（`rosclaw.sandbox.episode.failed`, `rosclaw.sandbox.action.blocked`, `rosclaw.runtime.execution.failed`），事件发布到 `rosclaw.how.recovery_hint.generated`。但 handler 内调用 RecoveryEngine 为 sync→async 边界（ThreadPoolExecutor），非纯事件驱动。 |
| L3: Practice记录全过程 | □ 通过 □ 未通过 ☑ 部分 | Practice 有 `rosclaw practice list/show/replay` CLI，但 How recovery hint 未写入 episode 元数据（缺少"恢复前/恢复后"对比）。 |
| L4: Memory回答真实问题 | □ 通过 □ 未通过 ☑ 部分 | Memory BM25 搜索正常，但 How 未主动查询 Memory 的历史失败模式做 analogy fallback（当前 knowledge fallback 是 stub）。 |
| L5: How给出恢复策略 | □ 通过 □ 未通过 ☑ 部分 | **RecoveryEngine 已实现**：`generate_recovery_hint()` 生成含 confidence 的 RecoveryHint，`build_retry_plan()` 输出 parameter_patch，`format_for_eventbus()` 生成 EventBus payload。22 条规则覆盖常见失败。但**真实失败场景下的"失败→How恢复→重试→对比提升"闭环未验证**。 |

---

## 二、场景验收自评

| 场景 | 当前状态 | 证据/说明 |
|------|----------|-----------|
| A: 小车PID运动控制 | □ 通过 □ 未通过 ☑ 部分 | 无 How 相关失败恢复测试。Kp 过大导致振荡时，How 规则有 "velocity exceeds limit"→"Add output saturation clamp"，但未在真实 PID 场景中验证。 |
| B: 机械臂reach | □ 通过 □ 未通过 ☑ 部分 | 有 "joint limit exceeded"、"collision detected" 规则，UR5e MuJoCo 仿真可达。但**sandbox firewall BLOCK 后 How 恢复→重试未闭环验证**。 |
| C: 机械臂抓取红杯子 | □ 通过 □ 未通过 ☑ 部分 | 有 "grasp slippage"、"unstable grasp"、"force exceeded" 规则及对应 parameter_patch（gripper_force_offset, approach_offset_z）。但**无 VLM 感知、无 grasp skill、无 critic 判断**，How 仅在 mock 测试中验证。 |
| D: Unitree巡检 | □ 通过 ☑ 未通过 □ 部分 | 无巡检任务代码，How 无相关规则。 |
| E: G1人形行走 | □ 通过 □ 未通过 ☑ 部分 | G1 sit-to-stand demo 能跑（固定底座）。How 有 "joint overload"、"sensor failure" 规则，但**fall detection 未实现**，无 free-floating 行走失败恢复。 |
| F: Forge自扩展 | □ 通过 □ 未通过 ☑ 部分 | AssetCompiler/ManifestBuilder 存在，但**Forge 生成的 bundle 中 How 规则未通过 Critic validation**。 |

---

## 三、HOW 模块详细清单

| 组件 | 状态 | 测试 | 说明 |
|------|------|------|------|
| HeuristicEngine | ✅ 完成 | 42/42 pass | 规则缓存、exact/substring 匹配、outcome 记录、stats |
| RuleManager | ✅ 完成 | ✅ pass | CRUD 操作 |
| RecoveryEngine | ✅ 完成 | ✅ pass | RecoveryHint 生成、build_retry_plan、format_for_eventbus |
| RecoveryFormatter | ✅ 完成 | ✅ pass | to_event_payload、apply_trajectory_adjustment、format_recovery_suggestion |
| 默认规则 | ✅ 22 条 | ✅ pass | 覆盖 joint limit、collision、timeout、gripper、force、sensor、communication 等 |
| Runtime EventBus 订阅 | ✅ 3 个 topic | ✅ pass | sandbox.episode.failed、action.blocked、execution.failed → how.recovery_hint.generated |
| Runtime 集成 APIs | ✅ 4 个 API | ✅ pass | capability_invoke、plan_action、sandbox_check、execute |
| MCP Tool | ✅ 已注册 | ✅ pass | `get_recovery_strategy` 可通过 MCP 调用 |
| Confidence 动态评分 | ⚠️ stub | N/A | 基于历史 success/failure 的静态评分，无时间衰减、无场景加权 |
| Knowledge Analogy Fallback | ⚠️ stub | N/A | 调用 Knowledge 模块做 analogy 查找，当前为占位实现 |
| Practice Episode 关联 | ❌ 缺失 | N/A | RecoveryHint 未写入 episode 元数据，无"恢复前/恢复后"对比 |
| Memory 历史查询 | ❌ 缺失 | N/A | How 未查询 Memory 的相似失败历史 |
| Dashboard 可视化 | ❌ 缺失 | N/A | 无 How recovery 专用面板 |
| 规则学习/自动扩展 | ❌ 缺失 | N/A | 规则靠人工 seed，无从失败中自动学习新规则 |

---

## 四、已知缺口（必须诚实填写）

1. **真实闭环验证缺失**：当前所有 How 测试都是单元测试/mock 测试。没有"真实失败 → How 生成 recovery hint → 下一轮任务应用 parameter_patch → Practice 记录对比"的端到端验证。

2. **Practice episode 未关联 How**：`RecoveryEngine.format_for_eventbus()` 生成了事件 payload，但 Practice 的 episode 元数据中未包含 recovery hint 信息。无法通过 `rosclaw practice replay` 看到"这次失败 How 给了什么建议"。

3. **Memory 联动缺失**：`HeuristicEngine._knowledge_fallback()` 是 stub，未真实调用 Knowledge/SeekDB 查询相似历史失败。How 只能基于预设规则，不能从记忆中学习。

4. **Confidence 评分过简单**：当前 confidence = success_count / (success_count + failure_count)，无时间衰减、无场景加权、无置信区间。一条很久以前成功的规则和新规则分数相同。

5. **参数 patch 解析硬编码**：`RecoveryEngine.build_retry_plan()` 通过关键字匹配 action 文本生成 parameter_patch（如 "increase grip force" → gripper_force_offset=0.15）。不够灵活，新增规则需要同步修改解析逻辑。

6. **Dashboard 无 How 面板**：Dashboard 有 Runtime Overview、Provider Health 等，但无 How recovery 历史、规则 efficacy 趋势图。

7. **规则覆盖率有限**：22 条规则覆盖常见机器人失败，但缺少传感器噪声、地形不平、网络分区、电池低电量、电机过热等场景。

---

## 五、预计修复工时

| 任务 | 工时 | 优先级 | 阻塞项？ |
|------|------|--------|---------|
| 真实闭环验证（写一个端到端测试：失败→How恢复→重试→对比） | 6h | P0 | **是** |
| Practice episode 关联（recovery hint 写入 episode 元数据） | 4h | P0 | **是** |
| Memory 历史查询联动（knowledge_fallback 实现） | 4h | P1 | 否 |
| Confidence 动态评分改进（时间衰减 + 场景加权） | 3h | P1 | 否 |
| 参数 patch 解析引擎（从 action 文本自动生成 patch，非硬编码） | 4h | P1 | 否 |
| Dashboard How 面板（规则 efficacy、recovery 历史） | 3h | P2 | 否 |
| 规则扩展（传感器噪声、电池低、电机过热等） | 2h | P2 | 否 |
| 验收文档 + 证据收集 | 2h | P0 | **是** |

**HOW 模块总计: 约 28h（约 3.5 人天）**

其中 **P0 阻塞项: 12h**（闭环验证 6h + Practice 关联 4h + 文档 2h）

---

## 六、需要其他模块配合的事项

1. **Practice 模块**：需要在 episode metadata 中预留 `recovery_hint` 字段，How 才能写入恢复建议。需要 Practice owner 确认 metadata schema。

2. **Memory 模块**：需要 Knowledge/SeekDB 提供 `find_analogy(error_log)` API 接口，How 的 knowledge fallback 才能真实工作。当前接口已定义但未实现。

3. **Sandbox 模块**：需要 Sandbox 在 BLOCK 时提供结构化 `failure_type`（而非自由文本），How 才能准确匹配规则。当前 BLOCK reason 是字符串，匹配靠 substring。

4. **Dashboard 模块**：需要 Dashboard 提供事件订阅接口或 WebSocket，How 的 recovery 历史才能实时展示。

5. **Runtime 模块**：EventBus handler 中 sync→async 边界当前用 ThreadPoolExecutor，建议统一改为 async EventBus 或提供 `async_publish` API。

---

## 七、HOW 模块评分（自评）

| 维度 | 评分 | 说明 |
|------|------|------|
| 代码完成度 | 7/10 | RecoveryEngine、规则引擎、EventBus 集成、MCP Tool、Runtime API 均已完成。缺 Practice/Memory/Dashboard 联动。 |
| 测试覆盖 | 8/10 | 42 个测试全部通过，覆盖单元测试、集成测试、MCP Tool 测试。缺端到端闭环测试。 |
| 文档 | 5/10 | 代码有 docstring，但缺少 HOW 模块 README、用户使用指南、规则扩展指南。 |
| 用户闭环 | 4/10 | MCP Tool 可调用，CLI 有 `rosclaw doctor`。但**真实失败场景下的恢复→重试→对比未验证**。 |

**HOW 模块总分: 24/40 → 换算后 6/10**

对比主管初评 2/10，HOW 模块已大幅提升（RecoveryEngine + 规则扩展 + EventBus + Runtime API），但**距 v1.0 P0 标准还差一个端到端闭环验证**。

---

## 八、结论

**HOW 模块当前状态：**
- ✅ Recovery Loop 核心链路已实现（HeuristicEngine → RecoveryEngine → EventBus → Runtime API）
- ✅ 22 条规则覆盖常见机器人失败场景
- ✅ 42 个测试全部通过
- ❌ 真实闭环验证未做（失败→How恢复→重试→对比提升）
- ❌ Practice/Memory/Dashboard 联动缺失

**是否满足 v1.0 P0 标准：**
- **部分满足**。How 能生成 RecoveryHint 和 retry plan，但缺少"应用建议→重试→记录对比"的完整证据。

**建议：**
1. 优先完成 P0 阻塞项（12h）：端到端闭环验证 + Practice 关联 + 文档
2. 完成后 HOW 模块可从 6/10 提升到 8/10
3. 配合主管路线图 Phase 1（1周修复 P0），HOW 模块可在 1.5 人天内达标
