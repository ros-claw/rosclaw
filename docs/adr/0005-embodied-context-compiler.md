# ADR-0005：Embodied Context Compiler（身体认知的可测试编译管线）

- 状态：Accepted
- 日期：2026-08-01
- 依据：实施总纲 §5、§6

## 背景

静态系统提示词会腐化：校准变化、传感器故障、换手爪、队友离线、权限到期后，旧 prompt 仍声称能力可用。Native Agent 的“身体认知”必须由可测试组件从可信源编译，而不是拼 prompt 字符串。

## 决策

1. 新增 `rosclaw.agentd.context.ContextCompiler`：可信源 + 动态源 + 任务源 + 组织源 + 策略 → 解析/验证/freshness/冲突检查 → 版本化 `EmbodiedContextBundleV1`（`rosclaw.embodied_context.v1`）→ 按预算裁剪后送模型。
2. 上下文分九层（L0 Constitution、L1 Identity&Body、L2 Dynamic Self、L3 Capabilities&Tools、L4 Mission&TaskGraph、L5 Memory&Knowledge、L6 Workers&Team、L7 Consent&Safety、L8 Conversation&Artifacts）。L8 是唯一可被用户/网页文本影响的层，且永远标记不可信边界；L0–L3、L7 不可被外部文本覆盖。
3. bundle 带 `context_id/context_revision/bundle_hash` 及 body/self/team/authorization 四个 binding hash；模型每个 Decision 必须绑定 context revision，过期决策拒绝执行并重新规划。
4. 预算策略：优先保留 L0/L1/L2/L7、当前任务与异常；历史对话与低相关记忆先摘要/引用化。工具两阶段加载：先 `search_capabilities` 候选摘要，再注入 5–12 个最相关严格 schema（`additionalProperties:false`）。超长上下文是上限不是数据治理替代品。
5. 编译确定性：同一输入产生同一 bundle hash；稳定层（L0/L1）单独序列化支持 provider prompt caching；保存 token 用量、各层占比、命中/裁剪记录作为评测指标。
6. 重编译触发器：body/calibration/maintenance hash 变化、SelfSnapshot sequence 变化或过期、daemon health/mode/capability 变化、Mission/TaskGraph revision 变化、Worker 变动、Team epoch/role/world revision 变化、grant 生效/撤销/过期、工具返回新证据、模型请求未加载工具类别、人类纠正关键事实。
7. 系统提示词（`agentd/context/prompts/native_agent_v1.md`）只表达稳定宪法与交互协议，英文为 canonical source；prompt 有 `prompt_id`/semver/内容 hash/发布日期，修改稳定规则需 ADR + 回归评测；自动生成的 prompt patch 只能进候选区，经固定测试集多 seed 评测后才可晋升。
8. 模型输出统一为 `DecisionV1`（`rosclaw.decision.v1`）：`next_intent ∈ {ANSWER, OBSERVE, PLAN_PATCH, HIRE_WORKER, TEAM_COORDINATE, REQUEST_APPROVAL, REQUEST_ACTION, VERIFY, WAIT, PAUSE, FAIL_SAFE}`，含 assumptions/evidence_refs/uncertainty/verification/on_failure/public_rationale。系统记录 public rationale 与证据，不索取、不记录私有思维链。

## 后果

- Truth before intelligence 可机械验证：未观测/过期/仿真/推断在 bundle 中显式标记；编译器单测覆盖 fail-closed 路径。
- 当前 `context.snapshot.json`（静态、面向外部 Coding Agent）保留原用途，不再是 Native Agent 的上下文来源。
