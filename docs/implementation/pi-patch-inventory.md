# Pi 上游补丁盘点（W01，规格 §5.1）

**基线**：`@earendil-works/pi-coding-agent@0.83.0`（package-lock sha256[:16]=`e2c557e386b10500`）
**盘点日期**：2026-09-09
**原则**：公开 SDK/extension hook > 已提交上游的小接口 > 临时版本绑定补丁。保留的补丁必须隔离、锚点硬失败、绑定适用版本并有退出计划。

## 总表

| 补丁 | 目标文件 | 原始需求 | 上游替代 | 状态 | 退出计划 |
|---|---|---|---|---|---|
| patch-01 resume formatter | `dist/modes/interactive/interactive-mode.js` | 退出提示必须引导 `rosclaw continue`（恢复经 kernel/binding/lease，不经外部 pi CLI）；WP-P0-1 不暴露内部 session id | 未见官方 resumeCommandFormatter hook（0.83.0） | **保留** | 上游提供 AppIdentity/resumeCommandFormatter 或等价 hook 时退休 |
| patch-01b retire 旧 hint | 同上 | 清理旧版 patch 残留（先 return 先生效） | — | 保留（清理器） | 随 patch-01 一起退出 |
| patch-02 builtin command policy | `dist/modes/interactive/interactive-mode.js` | `/trust /share /import /reload /update` 在 dispatch 前拦截（input hook 在 dispatch 后拦不住内建命令） | 未见官方 BuiltinCommandPolicy hook | **保留** | 上游提供命令前置 policy hook 时退休 |
| ~~patch-03 strip reasoning on write~~ | `dist/core/session-manager.js` | （旧）TranscriptPolicy：reasoning 不持久化 | **官方机制已足**：`settingsManager.setHideThinkingBlock(true)`（P1-1，UI 不展示）；会话存储本身是受控区域 | **已退役（W01）** | 应用器自动恢复上游原文（retire 条目） |
| ~~patch-04 never replay reasoning~~ | `dist/api/openai-completions.js`（顶层+嵌套两份） | （旧）TranscriptPolicy：绝不回放 reasoning 给 provider | **上游按官方协议管理 continuation**（thinkingSignature 按 provider 条件回放） | **已退役（W01）** | 同上 |

## patch-03/04 退役依据（规格 §5.2）

旧补丁的全局删除是跨 provider 的一刀切：

- Kimi 的部分 thinking 配置要求**按原样保留历史 reasoning 字段**才能 continuation；
- Claude 在工具回合中有 thinking block 协议要求；
- 删除字段不是 0907 问题的根因（0907 是验收/证据语义问题，不是推理字段问题）。

新政策（规格 §5.2 逐条落实）：

1. **UI 默认不展示内部思考** → 官方 `setHideThinkingBlock(true)`（pi-runtime.ts P1-1），无需补丁。
2. **provider continuation 所需字段** → 上游 openai-completions.js 按 provider 条件回放 thinkingSignature（官方协议），不再全局 `if(false)` 杀死。
3. **协议状态只留在访问受控的会话存储** → `~/.rosclaw/agent/sessions/`（目录权限受控）；**不自动进入遥测**——pi.events.batch 镜像只存 hash/元数据（content 字段硬拒绝，FULL_TEXT_FORBIDDEN）。
4. **公开导出不泄漏** → 两道：journey `_assert_reasoning_protocol`（provider 请求里 marker 只允许在官方 reasoning 字段——含 thinkingSignature 协议常量）；独立证据包（sanitized_assertions.json）的 `reasoning_forbidden_field_counts` 改为**泄漏语义**（只统计官方协议字段外的出现，verifier 全零 Gate 不变——官方字段不再误报）。

## 验证

| 实验 | 结果 |
|---|---|
| 退役-恢复机制（fresh node_modules 上 anchor→replacement 还原） | 实测：两份 openai-completions.js + session-manager.js 的 patch 文本计数归零；二次运行幂等（[already retired]） |
| applier 硬失败语义（锚点漂移） | 保留：非 optional 锚点缺失即 exit 1 |
| journey A（安装产物黑盒：多轮工具续接 + /compact + resume + reasoning 协议断言） | 绿（196s）——`_assert_reasoning_protocol` 替代旧 no_reasoning_replay |
| 五轮工具续接+恢复（真实主模型 K3 thinking continuation） | **NOT_RUN**（无 ROSCLAW_KIMI_API_KEY——不合成冒充；operator 带 key 重跑） |
| 改变身体后工具上下文刷新 | 已由 context envelope per-turn 重取 + stale 拒绝机制覆盖（既有测试） |

## 应用器安全性质（本轮新增）

- `retire: true` 条目：仅当已打补丁文本在场时恢复；pristine 环境 no-op；二次运行幂等。
- 修复了空 replacement 时 `source.includes(replacement)` 恒真导致恢复被跳过的问题（第一次实现即实证）。
