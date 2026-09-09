# W06 一次批准与真实隔离（规格 §10）

**分支**：a-w06-isolation（基于 main=8221d27，§17 依赖：W06←W01）
**日期**：2026-09-09

## 现状盘点（保留已有可用实现）

- workspace-pack shell 执行：bwrap 真实 exec 探测（`--ro-bind / /
  true`——非存在性检查）；不可用时 REAL/SHADOW **fail closed**，
  SIM 降级带 `[TOOL_LAYER_ONLY]` 诚实标记（R1-2b）——**正确，保留**。
- 本机实测：bwrap 两种 smoke 均失败（uid map Permission denied /
  RTM_NEWADDR）→ 执行路径自动走降级标记，无谎言。
- REAL 缺 Operator 无法执行：consent channel / action admission
  行为链已有测试覆盖（test_consent_channel 等）——保留不重复。
- evidence 受信区 kernel-only（P0-E）——进程内 provenance 的
  实际边界，本轮补行为锚点。

## 发现的谎言（已修）

会话开始的隔离提示文案宣称「shell 类操作将在会话内**弹确认卡
降级运行**」——该确认卡已被大道至简 R1-2b 退役（SIM 任务沙箱
bash 自动执行），提示在承诺一个不存在的机制（§10.2「不暗中
降级」反面：也不得宣称不存在的批准）。且提示挂在
session_start 上，resume/switch 会重复展示（§10.2「无沙箱
警告仅……显示一次」）。

**修改（packages/rosclaw-agent/src/extension/index.ts）**：

1. 文案如实：「任务代码直接在宿主执行：可访问工作区外文件与
   网络，产物证据为进程内 provenance（非安全隔离、非防篡改）；
   rosclaw doctor 查看结论与修复建议」。
2. `isolationNoticeShown` 一次性标志——resume/switch 不重复。

## 验证（红→绿）

| 测试 | 结果 |
|---|---|
| test_w06_isolation.py（4：ready⇒真实 smoke 不变式/落盘一致/evidence 区模型拒绝+kernel 通过/REAL·SHADOW 永不 POLICY_AUTO） | 绿 4/4（本机 bwrap broken 实测不就绪） |
| a0902-r1c TS（提示一次含两次 session_start/文案无「确认卡」/隔离可用无提示/doctor 落盘单源） | 绿 3/3 |
| TS 全量（232 例） | 229 pass / 0 fail / 3 skipped（6 个初始失败=stale 资产构建顺序，重构建后全绿，与本轮无关） |
| ruff check src tests | 全过 |

## 边界说明

- §10.1「真实 OS 隔离」：本机 bwrap broken——无法形成真实
  权限隔离，声明已降级为进程内 provenance（提示文案 +
  evidence kernel-only 边界 + TOOL_LAYER_ONLY 执行标记）。
  bwrap 可用的宿主上 workspace-pack 自动走强隔离（既有）。
- 同范围不重复批准/取消批准不影响聊天：Approval Broker
  （0902-R1）与 consent 链既有覆盖。
- 真实模型验收：**NOT_RUN**（无 key，不合成冒充）。
