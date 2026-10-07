# ADR: PI1.0.4 Provider 进展看门狗与诚实活动区 UX

- case: PI104_UX_COMPLETION_e6722ed6ad3c49a589c92b89b40b80ef（续作；前身 PI104_PROVIDER_UX_a89ba1a990004f3ba5af862da5e28551）
- scope: PI104_PROVIDER_PROGRESS_CLOCK_UI_COMPLETION_SOURCE_ONLY（仅 5 份产品/测试源；无硬件/ROS/NN/全局配置改动）
- 前史边界：上一版整体被 operator 以"raw Bash JSON framing 解码"为由驳回
  （NATIVE_STOPPED，不重评、不回溯计分）。本轮三个缺陷是**独立实测的**
  产品时钟/UI 健壮性问题，阻塞了集成验收；不作任何 historical22 超时
  因果断言。

## 背景（实证缺陷，非历史归因）

`ACTUAL_PI104_BOUNDARY_COUNTER.json` 实证：PI1.0.4 agent loop 会发出重复的
`text_start` 边界事件与空 `text_delta`。旧的已注册 `message_update` 钩子
无条件调用 `stallWatchdog.contentProgress()`，导致看门狗被持续重置——
`empty_boundary_updates` 模式下期望取消 1 次、实际 0 次（COUNTEREXAMPLE）。
本 ADR 只声明这一**当前**兼容性缺陷，不对 original22 超时做任何因果断言。

## 决定

1. **有意义增量分类**（`provider-watchdog.ts` 新增 `isMeaningfulAssistantEvent`）：
   - `text_delta` / `thinking_delta` / `toolcall_delta` 且 `delta` 非空 → 真实进展；
   - `text_start/end`、`thinking_start/end`、`toolcall_start/end` 边界事件与空
     delta → 不续期，不推迟首 token / 流式 idle 取消；
   - 无结构事件 / 未知类型 → 保持旧兼容（视为流动）。
   `message_update` 钩子改为 `assistantEventProgress(assistantMessageEvent)`。
2. **诚实活动区**（`activity.ts` + `index.ts`）：回合期间 250ms 周期把真实
   阶段写入 working message——`等待模型首个响应（已等待 Xs）` /
   `模型输出中（累计等待 Provider Xs）` / `等待用户确认（Provider 计时已暂停…）`；
   工具阶段保持 `调用 <tool>` 文案（Provider 计时暂停，不把工具时间误标为
   Provider 停滞）。thinking 增量只作活性信号，**其文本绝不上屏**。
3. **等待耗时记账**：`providerWaitElapsedMs()` 只累计真实等待 Provider 的
   时间；`pauseForTool`/`pauseForUser` 冻结、resume 续计；`turnStarted`
   归零、`_disarm` 停表。
4. **保留语义**：嵌套工具计数暂停、用户暂停/恢复、终态定时器全解除、
   缺失 abort API 的诚实警告、env 阈值覆盖（1s..1h 边界）、非空长流不杀——
   全部由既有测试与新增 `provider-progress-pi104.test.ts` 覆盖。

## 本轮新增决定（三个实测缺陷的修复）

5. **重叠暂停恢复顺序**（`provider-watchdog.ts` `_resumeWait`）：实测两种
   顺序——user 先恢复（tool 仍忙）与 tool 先恢复（user 仍忙）——都会让
   Provider 等待时钟多计约 90ms 暂停期。修复：`_resumeWait` 只有在
   `!userBusy && toolBusyCount === 0`（全部暂停清空，含嵌套工具）且
   `waitResumedAt == null` 时才恢复计时。工具执行仍不算 Provider 等待。
6. **终态耗时冻结**（`_disarm`）：实测终态后实测等待约 35ms 的
   `providerWaitElapsedMs()` 回退为 0。修复：`_disarm` 先 `_pauseWait()`
   把在计段并入 `waitAccumMs` 再停表——终态保留实测耗时且不再累加空闲
   时间；新回合由 `turnStarted` 自行归零（`waitAccumMs = 0`）。
7. **周期回调 UI 异常隔离**（`index.ts` `refreshProviderActivity`）：实测
   已注册的 250ms 周期回调里迟到的 `setWorkingMessage` 抛错会以
   uncaught exception 崩宿主。修复：try/catch 包裹该 UI 写入——只丢
   这一帧进度文案，看门狗、取消与终态定时器清理语义不变（与既有 M8
   notice/abort 隔离同款）。

## 验证（真实执行，非声明）

- `--mode check`：PASS_NATIVE_PROVIDER_PROGRESS_SOFTWARE_CHECKS。
  已注册 callback 反例修复：silent/empty_boundary/empty_delta 均 abort=1
  （修复前 empty_boundary/empty_delta 为 0）；nonempty/thinking/toolcall
  delta 与 long_tool abort=0；7 种模式 label 均含诚实阶段+耗时文案。
- `--mode regression`：PASS。450 测试，447 pass / 0 fail / 3 skipped
  （skip 为既有跳过项），含全部看门狗、扩展、回归目标。

## 局限

- 验证基于合成 provider + 实际编译的 PI 与扩展工厂 callback（无真实
  RPC/模型/ROS）；不构成对线上 Provider 行为的证明。
- 活动区刷新周期 250ms，耗时显示精度 0.1s；未做跨进程时钟校准。
- 未改动 original22 相关任何历史路径，也不声称修复历史超时。
