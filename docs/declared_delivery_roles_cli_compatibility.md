# Declared delivery roles CLI compatibility

## 已知失败（保留，不是本轮运行结果）

Root 给定：LOFT Native28819 启动 3.129s 后 RUNNER_FAILED；PTY94B 输出
`rosclaw-agent failed: INVALID_TOOL_CALL_POLICY: unknown top-level key: declaredDeliveryRoles`。
模型请求 0、managed operations 0，物理未启动。PR721 main.ts 的 CLI allowlist
漏了已受 budget hook 支持的字段。本补丁不删除声明，不改 GO、预算或 dispatch，原失败不撤销。

原 Native1457 候选父状态 CLOSED1：102.514s、正常 STOP/quit、无 tool errors，
但 review AssertionError；原 whole FAIL 永久保留，不因本次源码修复重标通过。
其字段接纳和严格结构校验可组件追溯；本次仅修复 canonical-deferral 与对应测试/说明。

## 补丁边界

CLI 接纳并原样返回该字段，缺省仍走旧行为；显式 null 不视作缺省。
早期校验要求 1..128 项、每项恰有 path/role、字符串 path 与 role、合法路径词法、
最长 128 字符的精确 role，以及不重复的 path。允许 report/plot/image/video/data、
progress/diagnostic 与其 ASCII 字母数字下划线后缀；严格全串匹配，不接受尾部换行。
必须允许 rosclaw_deliver，且 exactPaths.rosclaw_deliver 数组必须存在并通过原有词法校验。
Stage A 只拒绝 role 声明中的词法重复，不检查 path 的字面成员归属。
`./x.json` 与 `x.json`、relative/absolute 等价形式的成员归属延后由原 constructor
在 workspace 下 canonical binding 时检查；foreign.json 的词法透传不代表获得权限。
不做 path/role coercion、规范化或策略重建。

这只是 loader 的必要关联和 schema 校验，不是 workspace 权限授权。
别名、规范路径重复、workspace 外路径、symlink escape、实际文件类型等仍由既有
`createToolCallBudgetExtension` canonical binding 检查；不能凭 CLI 通过就声称其已通过。
hook、Python、原 tests、计数、UNKNOWN、模型 dispatch 和入口控制流均未修改。

## 新回归测试（已编写，未执行）

新测试 spawn 当前 Node 的真正编译产物 `dist/src/main.js`，不截取函数、不 eval
loader、不修改编译产物。必须先构建；缺少 main.js 明确失败，不 skip。
要求支持 `node:module.register` 与 `--import` 的 Node（至少 20.6，建议使用项目指定版本）。

临时 ESM import firewall 只允许真实 main、version、policy/admission validator 及指定
Node builtins；UI cancellation 使用惰性 stub。首个 post-loader 的 pi-sessions import
输出离线边界标记，其余 session/runtime/workspace/context/SDK imports 全部是无 SDK
副作用的离线 stub。runtime stub 记录实际收到的 toolCallBudget 然后受控失败：
与输入完整深比较，证明声明未丢弃或改写，而不是重新读取 policy 文件自证。
独立 preload 同时禁止网络和子进程 API，不只依赖 PI_OFFLINE；子进程不继承
凭证、代理、NODE_OPTIONS 或 provider 配置。临时文件仅在测试自身目录写入并清理。
任何未知 import、超时或信号都导致测试失败。

正例包括真实五项 diagnostic_report、所有合法 role 形态、128 项边界、字段省略、
无 policy flag；负例覆盖 null/非数组/空/129 项、坏 entry/额外或缺少 key、非字符串、
非法 role/path、词法重复、未允许 delivery、缺少 exact delivery paths。
原 invalidRoles 仅移除 foreign exact path 的 Stage A 负例，其余条目及原 helpers、
直接注册测试与断言保留；有限 inert 数组的 for-of 测试注册在本次公开 scope 内。
新增正例验证 `./x.json`/`x.json`、relative/absolute 双向形式和 foreign.json 原样
抵达 inert runtime；每例均完整比较实际捕获的 policy，不代表 constructor 授权。
foreign 非成员仍须由原 constructor 拒绝，canonical alias/symlink 权限检查未前移。
负例必须 exit 2 且含 INVALID_TOOL_CALL_POLICY，并不得出现首个后续边界标记。
原 main.ts baseline 的正例会因 unknown key 失败，因此不是只验证负例的空洞回归。

stub 标记不代表 SDK、认证、provider、模型或网络曾执行。该 CLI 测试故意不运行
canonical constructor；canonical 语义仍需由原 hook 测试独立验证，未声称本测试覆盖。
只读原 cli-tool-call-policy 测试使用不存在 resume session 作真实后续哨兵，全部保留。

## Root 后续验证前提及命令

本轮只有 read/edit；没有构建、Node、shell、测试、SDK 或任何进程执行，故无 PASS 结论。
以下是交付 Root 的待执行命令，不是已执行记录：

1. 使用仓库既定构建流程生成最新 `packages/rosclaw-agent/dist/src/main.js` 及
   `dist/test/*.test.js`，确认不是旧产物；构建命令由 Root 按项目配置确认。
2. 从仓库根目录运行：

   ```sh
   node --test packages/rosclaw-agent/dist/test/cli-declared-delivery-roles.test.js
   node --test packages/rosclaw-agent/dist/test/cli-tool-call-policy.test.js
   ```

3. Root 另运行原 budget hook（含 canonical path/alias/symlink）及项目相关既有套件，
   审核 AST scope 与编译/typecheck；原 suite 文件名未在本轮获准读取，不杜撰命令。

Root 实际验证结果、退出码目前均未提供。本文不宣称全系统完成、动力学仿真完成或真机完成。
