# Pi Coding-Agent SDK Embedding Audit（PR-PNA-0 前置强制审计）

- 日期：2026-08-05
- 上游基线：`earendil-works/pi` commit `588915ec71714688cee8b7153339e8bdebb3e82e`
- 包：`@earendil-works/pi-coding-agent@0.83.0`（engines `node >=22.19.0`，纯 ESM，npm 包**自带预构建 dist**，消费者无需 TS build）
- 方法：sparse clone（coding-agent/agent/ai/tui/protocol/client）+ 源码精读；所有结论附 file:line。

## 0. 总结论

重构规格 §9 假设的 SDK 入口**全部真实存在且从包根导出**：
`createAgentSession` / `createAgentSessionFromServices` / `createAgentSessionServices` /
`createAgentSessionRuntime` / `InteractiveMode` / `runPrintMode` / `runRpcMode` /
`SettingsManager` / `SessionManager` / `ModelRuntime` / `DefaultResourceLoader`
（src/index.ts:152-261、src/core/sdk.ts:169、agent-session-services.ts:136/195、
agent-session-runtime.ts:414）。

**PR-PNA-0 可以纯 extension + SDK 组装完成，不需要 fork。**
fork 只在产品化阶段为品牌残留（APP_NAME、内建命令表、`/resume` 选择器流、`.pi` 目录名）准备，见 §10。

## 1. 包根正式可 import 的能力

exports map 只有 `"."` / `"./rpc-entry"` / `"./client"` 三个子路径（package.json）。
SDK 全部走根 `"."`。关键导出（src/index.ts）：

- session/services/runtime：`createAgentSession`、`createAgentSessionFromServices`
  （options: services/sessionManager/sessionStartEvent/model/thinkingLevel/scopedModels/
  tools/excludeTools/noTools/customTools —— agent-session-services.ts:51-64）、
  `createAgentSessionServices`（cwd 必填 + agentDir/settingsManager/modelRuntime/
  resourceLoaderOptions 等 —— :35-44）、`createAgentSessionRuntime`（:414-432）。
- `AgentSessionRuntime`：`switchSession/newSession/fork/importFromJsonl/dispose/
  setRebindSession/setBeforeSessionInvalidate`。
- `SettingsManager.create(cwd, agentDir?, options?)`（settings-manager.ts:311）。
- `ModelRuntime.create({ credentials?, authPath?, modelsPath?, ... })`
  （model-runtime.ts:66-80,171-173）。
- `DefaultResourceLoader` + `ResourceLoader` 接口 + `loadProjectContextFiles`。
- 工具工厂：`createCodingTools`、`createReadOnlyTools`、`createReadTool` 等（index.ts:214-223）。
- 扩展类型全家桶（type-only）+ `defineTool`、`wrapRegisteredTools`、`ExtensionRunner`。
- UI 组件：`FooterComponent`、`CustomEditor`、`ToolExecutionComponent`、selector、theme。

## 2. 仅存在于内部源码的能力

- `APP_NAME`/`APP_TITLE`（src/config.ts:403-404）——从 pi 自身 package.json 的
  `piConfig.name` 在模块加载时读取；不导出、无运行时覆盖。用于内建 header logo
  （interactive-mode.ts:888）、更新提示（:4072,4101）、`/quit` 文案。
- `BUILTIN_SLASH_COMMANDS`（src/core/slash-commands.ts:20-43）不导出。
- `AuthStorage` / `FileAuthStorageBackend` / `InMemoryAuthStorageBackend`
  （auth-storage.ts:50,202,237）不导出（但 `ModelRuntime.create` 接受自定义
  `CredentialStore`，见 §7）。
- `getAuthPath`/`getModelsPath`/`getSessionsDir`（config.ts）不导出。

## 3. InteractiveMode 可由 extension 定制的部分

构造：`new InteractiveMode(runtimeHost, options)`（interactive-mode.ts:508）；
options 只有 migratedProviders/modelFallbackMessage/autoTrustOnReloadCwd/
initialMessage(s)/verbose/uiMode（:314-331）——**无品牌选项**。

extension `ctx.ui` 可定制（extensions/types.ts:131-282）：

| 部位 | 接口 | 覆盖方式 |
|---|---|---|
| header | `setHeader` | ✅ extension |
| footer | `setFooter`（接 TUI/Theme/FooterDataProvider） | ✅ extension |
| working 动画 | `setWorkingIndicator({frames, intervalMs})` / `setWorkingMessage` / `setWorkingVisible` | ✅ extension |
| widget | `setWidget`（编辑器上/下） | ✅ extension |
| 终端标题 | `setTitle` | ✅ extension |
| 编辑器 | `setEditorComponent` / `addAutocompleteProvider` | ✅ extension |
| 主题 | `setTheme` | ✅ extension |
| 内建 header 内容（logo 文本） | — | ❌ 硬编码（需 fork 或 setHeader 全替换） |
| selector 流（/resume /tree 对话框流程） | — | ❌ 流程硬编码（组件虽导出） |

## 4. Pi 内建命令与冲突

内建全表（slash-commands.ts:20-43）：
`settings model scoped-models export import share copy name session changelog
hotkeys fork clone tree trust login logout new compact resume reload quit`
（+ 隐藏 `/debug` 等三个彩蛋）。

- **禁用/替换内建命令：无设置、无 hook**。dispatch 在任何 extension 处理之前
  （interactive-mode.ts:2840-2967）；同名 extension 命令永远输给内建（:601-616）。
- **对策（无 fork）**：ROSClaw 新增命令用不冲突的名字（/mission /body /estop
  /delegate…）；`/trust`/`/share`/`/import` 这类必须改语义的，用 `input` 事件
  在到达内建 dispatch 之前拦截——注意 `input` 对内建命令**不触发**（它们不进
  session.prompt），因此 ROBOT profile 下必须靠 `onTerminalInput`/自定义 editor
  或薄 fork 的 BuiltinCommandPolicy。PR-PNA-0 先接受内建语义（developer 桌面），
  PNA-9（Resource Security）再做 ROBOT 拦截。
- `!` bash：**可以功能级关闭**——`user_bash` 事件支持 full replacement
  （返回 `{result}` 则内建执行被跳过，interactive-mode.ts:6248-6276）。
  返回固定 `{output:"bash disabled by ROSClaw policy", exitCode:1, ...}` 即可。

## 5. 关闭项目资源发现

`DefaultResourceLoaderOptions`（resource-loader.ts:158-193）逐项开关：

| 类别 | 开关 |
|---|---|
| `.pi/extensions`、agentDir extensions、npm 扩展包 | `noExtensions: true`（`additionalExtensionPaths`/`extensionFactories` 仍加载——正好用来注入 ROSClaw 内联扩展） |
| skills（含 `~/.agents/skills` 与 git 祖先） | `noSkills: true` |
| `.pi/prompts` | `noPromptTemplates: true` |
| themes | `noThemes: true` |
| AGENTS.md/CLAUDE.md | `noContextFiles: true` |
| system prompt 文件 | `systemPrompt`/`appendSystemPrompt` 直接覆盖 |

## 6. Session 生命周期事件覆盖

`pi.on(...)`（types.ts:1198-1239）：

| 事件 | veto/customize |
|---|---|
| `session_before_switch`（new/resume/import） | ✅ `{cancel:true}` |
| `session_before_fork`（含 /clone——clone 走 fork at-position） | ✅ cancel / skipConversationRestore |
| `session_before_compact` | ✅ cancel / 自定义 CompactionResult |
| `session_before_tree` | ✅ cancel / 替换 summary/instructions |
| `session_start` / `session_shutdown` / `session_compact` / `session_tree` | 观察 |

→ new/resume/fork/tree/clone/compact 全部可被 ROSClaw extension 拦截或否决，
满足 §13 生命周期映射的实现前提。

## 7. Credential store 替换

- `ModelRuntime.create({ credentials: CredentialStore, modelsPath: null })`：
  CredentialStore 接口来自 pi-ai（read/list/withLock 写）。
- ROBOT profile env-only：传"只读 env、写入即拒绝"的自定义 CredentialStore +
  `modelsPath: null`（跳过 models.json）。
- OAuth：provider 级 `pi.registerProvider(name, { oauth: {...} })`；
  `/login` 由 LoginDialogComponent 驱动——可保留（developer）或经 ResourcePolicy
  在 ROBOT profile 下关闭。
- 注意：`createAgentSession` 只有在你**不**传 `modelRuntime` 时才从 agentDir 推导
  authPath/modelsPath（sdk.ts:174-176）——必须显式传 modelRuntime。

## 8. createAgentSession 选项与自定义工具

- `noTools: "all" | "builtin"`（sdk.ts:61）+ `tools` 白名单 + `excludeTools`
  黑名单 + `customTools: ToolDefinition[]`。
- ToolDefinition（types.ts:449-498）：`name/label/description/promptSnippet/
  promptGuidelines/parameters(TypeBox)/execute(toolCallId, params, signal,
  onUpdate, ctx)/renderCall/renderResult`。`defineTool()` 保留类型推导。
- `onUpdate` 流式进度 + `renderCall/renderResult` 自定义组件（ToolExecutionComponent
  可复用）→ 满足 §20.5 tool waiting 原位更新。
- **无内建 permission/approval 框架**——用 `tool_call` 事件 `{block:true}` +
  自渲染 ApprovalCard（§20 的专用卡片不走 ctx.ui.confirm，直接由 ROSClaw
  extension 组件实现）。

## 9. Extension API 面（关键子集）

可 veto/modify：`tool_call`（block/改参）、`tool_result`（改结果）、
`before_agent_start`（换 systemPrompt/注入 custom message）、`context`
（每次 LLM 调用前替换消息列表）、`before_provider_request`（换 payload）、
`before_provider_headers`（改 header）、`message_end`（替换最终消息）、
`input`（transform/handled）、`user_bash`（替换执行）、`project_trust`
（决定信任）、`session_before_*`（见 §6）。
注册面：`registerTool/Command/Shortcut/MessageRenderer/EntryRenderer`、
`sendMessage/sendUserMessage/appendEntry`、`setActiveTools`、`setModel`、
`registerProvider`、`events: EventBus`。
内联扩展（`extensionFactories`）无需落盘文件。

## 10. 需要薄 fork 的点（产品化阶段，非 PNA-0 前提）

| 需求 | 结论 |
|---|---|
| APP_NAME/APP_TITLE 品牌串 | fork：package.json `piConfig.name/configDir` 是上游设计的 rebrand 旋钮（config.ts:385-410） |
| 内建命令表禁用/改名 | fork：BuiltinCommandPolicy（规格 §23.4 已预留接口设计） |
| `/resume` 等 selector 流程替换 | fork：流程硬编码 |
| `.pi` 项目目录名 | fork：`CONFIG_DIR_NAME` 构建期固定 |
| 其余（header/footer/working/widget/资源策略/凭据/工具/生命周期 veto） | **extension 全覆盖** |

## 11. 风险登记

1. **内建命令不可 veto**（最大缺口）：ROBOT profile 的 `/trust`/`/share`
   语义改造在 PNA-9 前必须靠自定义 editor/onTerminalInput 或 fork；
   PNA-0 仅 developer 桌面场景，风险可接受并记录。
2. `user_bash` 替换执行 ≠ UI 层 `!` 提示消失（边框变色仍在）——PNA-9 处理。
3. 升级策略：精确锁 0.83.0 + overrides + pi-upstream.lock.json +
   characterization suite（规格 §29.1）；禁止 `^` 范围。

## 12. 结论

按规格 §36：进入 PR-PNA-0——`packages/rosclaw-agent`，Pi InteractiveMode +
ROSClaw header/footer + working 动画 + `noTools:"all"` + 内联 ROSClaw
extension（user_bash 关闭、noExtensions/noSkills/noPromptTemplates/
noContextFiles）+ 一个 `rosclaw_status` 只读工具 + 不切默认 engine。
