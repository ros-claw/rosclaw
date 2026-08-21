# ADR-0013：Harness Backend 默认与认证路线——用户无 engine 面

（即调整方案 §十二.3 所称"ADR-0012A"；按 ADR-0000 命名规范编为 0013）

- 状态：Accepted（2026-08-21，PR-ADR-0012A）
- 基线：main=`090da56`
- 依据：`ROSClaw_最强Native_Agent_认知与执行内核重构实施方案_2026-08-21调整.md` §八/§九/§十/§十二
- Amends：ADR-0012（不推翻；细化 Backend 策略与删除清单）

## 决定

1. **Pi 是当前唯一默认 Harness Backend，且用户没有任何 engine 选择面。**
   不存在也不许新增：`--engine pi|codex`、`--backend <harness>`、
   `--harness`、配置键 `harness.engine`、TUI `/engine` 命令、UI 中的
   引擎名称。CLI 中合法的 `--engine`/`--backend` 仅指**仿真/运行时
   引擎**（choices ∈ mujoco/isaac/mock/fixture/ros2），语义与
   Harness Backend 无关；任何 choices 混入 Harness 名称（pi/codex/
   app-server/claude-code）即违规。Backend 切换是 ROSClaw 内部的
   认证与装配决策，不是用户决策。
2. **Provider ≠ Harness。** ModelProfile.provider 词表只含模型 API
   提供方（kimi_cn/kimi_code/mock/…）。OpenAI 是 provider；
   Codex app-server 是 Harness Backend 候选；Pi 是 Harness Backend。
   Codex/Pi 永远不以 provider 身份出现。
3. **Codex 是第二认证 Harness 候选，不立即替换 Pi。** 前置条件
   （调整方案 §九）：N5（能力/工具/输出协议）、HP1（NativeEventV2）、
   HP2（Pi Backend 真正迁移）、N6–N8 全部通过后，才实现 Codex
   app-server 原型（`codex app-server` stdio/Unix socket、
   Thread/Turn/Item 映射、固定版本协议 Schema、Tool Catalog 投影为
   本地 MCP、内建文件/Shell 限 Workspace、Thread ID 只在内部
   binding、不开 subagent）。只有通过 HP3 Backend Conformance 与
   ROSClaw 任务基准后，才决定是否在 OpenAI 模型配置下成为默认
   Backend。
4. **删除孤儿 Worker 形态 Harness 驱动。** `agentd/codex_driver.py`
   （十五审 RF-5）与 `agentd/acp_driver.py`（十五审 RF-3）在 H9 删除
   Worker 默认链后已无任何生产引用（仅各自测试引用）。它们是"Codex
   当 Worker"的形态——调整方案 §十 明确禁止复活此形态。未来 Codex
   从 `NativeHarnessBackend` SPI + HP3 conformance 重写，不从这两个
   文件演化。
5. **DeepSeek：只借鉴模式，不引入依赖。** append-only session log、
   canonical tool output、presentCall/presentResult、guarded
   pre/execute/post pipeline、Code Mode 纳入 N5/N6 设计；Embodiment
   Code Mode 等强类型能力工具稳定后再评估，且每个 SDK 子调用仍经
   ToolGateway、Effect Policy 与 Evidence Pipeline。
6. **Worker 保持关闭。** 普通任务 Worker 数量 = 0；长运行用
   Operation 而非 Worker；Codex 若成为主 Harness 则不是 Worker。
   Worker 只有在 Native 基准通过后才作为显式隔离承包商重新设计，
   且仍不得持有物理工具、ROS/DDS、设备 SDK、permit 或 operator
   token。`rosclaw setup worker` 仅为探测，不构成启用。

## 为什么

当前最重要的不是把 Codex 塞进来，而是先用 N5 把能力、工具、输出、
安全和展示做成统一协议；否则换任何 Harness 都会复刻"能力参数靠猜、
执行结果靠文本、TUI 无法重放、Verifier 不知道实际加载了什么资产"。
用户面对 engine 选择只会得到品牌泄漏与配置分裂；Backend 认证是
ROSClaw 的质量门槛，不是用户负担。

## 后果

- 新增：`tests/agentd/test_adr0012a_no_engine_surface.py` 结构守门
  （无 Harness choices / 无 engine 选项与命令 / provider 词表纯净 /
  孤儿驱动不得复活）。
- 删除：`agentd/codex_driver.py`、`agentd/acp_driver.py` 及各自测试
  （test_fifteen_rf4/rf5）。`adapters/acp/`（ACP server 对外适配面）
  不受影响。
- 修正：`packages/rosclaw-agent/src/main.ts` 陈旧注释（曾引用不存在
  的 `rosclaw chat --engine pi`）。
- 不变：rosclawd 是 REAL 唯一执行权威；E-Stop 不依赖 LLM；REAL 门禁
  保持关闭；ModelService 单轨（ADR-0012 #6）。
