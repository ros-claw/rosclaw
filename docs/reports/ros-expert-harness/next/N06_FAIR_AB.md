# N06：同模型 Harness 消融与预算验收

## 当前结论（2026-10-09）

正式 A0–A3 × B1–B5 对照仍为 **NOT_RUN**。当前完成的是预算组件、真实 Native 合成文件任务和隔离边界的验收；没有机器人任务收益、Memory 因果收益或产品间胜负结论。

## 已封存的真实模型组件试验

`n06-real-native-budgeted-worker-smoke-attempt01` 使用实际 Native、真实 `openai-codex/gpt-6.1-sol`、相同 `low` 推理参数以及 V4 镜像 `952dfdc78730389c833bdbc397b4c38e4b767f6a3f26e8d4573e4341cf70a8cd`。镜像内 Core/Native 为 `5aa66c084ccb238ef129a5472980fb1521c00201`，宿主预算门禁为 `4291e2504a4044093aad9ac7fbb37ba70ab32fd4`。三个模式只改变合成 Memory 投影，任务是读取隔离信息、编写并交付文件。

| 模式 | 实际请求 | 输入 token | 输出 token | 总 token | Task/Artifact/容器收尾 |
|---|---:|---:|---:|---:|---|
| M0 | 9 | 42,998 | 1,036 | 44,034 | PASS |
| M1 | 9 | 46,589 | 1,070 | 47,659 | PASS |
| M2 | 10 | 57,780 | 1,125 | 58,905 | PASS |

三组预先登记相同的 16 次请求、每次 2,048 输出 token、120,000 总 token 和 600 秒预算。28 次实际请求均带输出上限；完整响应在交给 Native 前由独立门禁核验。原始 Native 请求、仅添加输出上限的上游请求、SSE、模型身份、usage、账本、交付文件和停止状态逐条关联，171 个公共文件已封存并校验。真实认证信息留在宿主，未进入容器或公共归档。

这些运行使用的 `4291` **没有工具次数门禁**。不能把后续新增工具门禁或离线回放当成当时已执行的控制。没有物理 World、真实清扫或新的机器人 Body。

## 预算语义与当前实现

`RegisteredModelBudgetV1` 冻结模型、推理参数、请求次数、输出 token、总 token、墙钟及工具次数上限。预算引擎位于 `src/rosclaw/connectors/ros/verification/model_budget.py`；operator broker 缓冲完整 SSE，通过门禁后才向实际 Native 释放。

- 请求一次只允许一个在途；失败同样占用请求次数，失败后不能重置预算继续。
- 输出上限写入真实上游请求，并核对实际响应中的上限回显和 usage。独立长输出探针观测到 `response.incomplete/max_output_tokens`，实际输出为登记的 32 token；该单次探针仅证明该请求的限制生效。
- 输入 token 在推理结束后才可知。超额请求实际消耗照实记录，整条响应被拒绝；这属于 **RESPONSE_RELEASE_GATE_NOT_PREBILLING_COST_CAP**，不能承诺计费前硬限制。
- token 仅采用上游权威 usage 一次，缓存输入仍计入输入；不叠加 Core 与 SDK 的重复记录。缺少 usage 或异常取消记为不完整。
- 新增工具门禁统计实际完成的 function calls，验证名字来自该次请求的工具声明、唯一调用 ID、完整 JSON 对象参数。流式工具名、ID、参数增量、完成项必须与最终响应一致；超预算拒绝整批工具调用。已释放调用达到上限后仍允许不含工具的最终回答。
- 工具次数代表向 Native **释放的模型工具请求数**，不是工具成功数、底层 ROS RPC 数或动作许可。嵌套 shell 子命令不单独计数；工具权限和 daemon 安全网关仍需独立约束。
- 账本保留开始、截止、每次请求准入及完成的单调时钟。墙钟超时由宿主 transport 取消实际请求；仅调用纯预算引擎本身不能取消网络操作。

新增门禁的离线回放只验证历史字节在新规则下的接受情况，不能证明历史运行使用新门禁，也不能用虚拟回放时间证明真实墙钟约束。新实验必须冻结新源码、明确工具上限并重新预注册。

## 正式实验要求

A0 最小 ROS bridge 与基本工具；A1 加 SystemModel/Doctor/Readiness；A2 加 Resolver/ContextCompiler/Coverage Mission；A3 加 evidence-gated Memory。四组保持相同实际模型、Native 协议、推理参数、自然语言输入、初始仓库、World、地图、种子、预算、权限和 daemon 网关，分别隔离 Memory 与缓存。

B1 已知机体静态清扫；B2 故障排查后清扫；B3 移动障碍与补扫；B4 未见 Body 导航和全房清扫；B5 迁移。先预注册 5 次 pilot，再冻结 10 个新配对任务；发布每条 run、失败、人工干预、配对变化、中位数和不确定区间。不能用三个文件任务代替这些场景。

Run Manifest 必须包含实施方案列出的 benchmark/scenario/robot、fixture/world 哈希、种子与初始位姿、源码与镜像、ROS/Nav2/opennav 版本、实际模型与推理参数、token/墙钟/工具预算、工具权限、人工干预、Memory 模式、许可动作模式、任务结果、coverage/collision、证据引用和失败码。未知项写 UNKNOWN，不能从镜像标签或模型目录推断。

正式运行前还须完成 N02 合并门、N03 实际动态验收、N04 未见 Body 的 L0–L4 与 N05 机器人任务的三组因果证据；模型目录可见不代表调用成功。产品间 Codex/Native 对照单列为实验 B，未验证两端实际同模型时标为 product-level comparison。

## Run Manifest 与整组声明核对

`src/rosclaw/connectors/ros/verification/fair_ab.py` 提供 `FairRunManifestV1`、具体的 `NativeInferenceParamsV1` 和 `audit_fair_ab_cohort()`。原计划的必需字段均保留，并补充地图、任务输入、提示词、初始仓库、Native 协议、安全网关策略哈希、工具次数预算、独立 worker/Memory/execution 标识、实际模型、usage 完整性及停止观察。

A0–A3 feature_flags 与 Memory 模式固定对应；相同 seed 的初始条件必须一致，不同 seed 可以有不同起始位姿或场景字节。同一整组的源码、镜像、模型、推理参数、预算、权限及网关策略必须一致。5 个 pilot seed 要有全部 20 条记录，10 个 evaluation seed 要有全部 40 条记录；检查已提供的历史 seed 列表，拒绝重复、缺失、替换和共享资源标识。失败 run 计入整组，保留原始失败码和 evidence refs；人工干预逐条列出。

未知观察用 null/UNKNOWN，不能默认成功、零碰撞或零人工协助。PASS 声明必须有原始证据引用、实际相同模型、完整 usage、至少 98% 覆盖、零碰撞和 verified stop。即便声明满足这些条件，整组核对也只输出 **DECLARATIONS_ONLY_NOT_EXECUTION_ACCEPTANCE**，始终 `evidence_bytes_verified=false`、`dispatch_authorized=false`、`robot_authorization=false`。它不读取证据原始字节、不检查实际进程隔离、不切换运行时 Harness 功能、不授予动作权限，也不证明因果收益。运行前后还必须由独立 operator 核对事实及 daemon Receipt，不能拿字段一致代替实测。

## 新门禁的真实 Native 与空参数兼容性复验（2026-10-09）

宿主门禁源码 `069a68411616ef4fe6fc70ec16fba9b6f41461f4` 独立登记了每组 12 次释放工具请求、16 次模型请求、2048 输出 token/请求、120000 总 token、600 秒。沿用精确 V4/Core+Native5aa 镜像与相同合成文件任务，M0/M1/M2 各自隔离运行。实际模型均为 gpt-6.1-sol/low；三组 TaskKernel SUCCEEDED、Artifact SHA/长度对应、容器全部停止/Pid0/exit0。

| 模式 | 实际模型请求 | 实际释放工具调用 | 总 token | 原单调时钟耗时秒 |
|---|---:|---:|---:|---:|
| M0 | 10 | 9 | 57843 | 63.0173 |
| M1 | 10 | 9 | 61653 | 61.8907 |
| M2 | 9 | 8 | 48455 | 59.0350 |

独立复核从原 Native 请求、仅增加输出上限的上游请求与 SSE 重建每笔预算决定，使用原始单调时间逐笔重算，与原账本完全一致。上游函数名、组合调用 ID、JSON 参数顺序对应真实 Native toolCall 和 toolResult；所有 26 个工具调用都有对应结果，未超过登记预算。174 个公开文件封存于 harness `evidence/2026-10-09/n06-live-tool-gate-three-real-native-synthetic-tasks`。这不是正式机器人 A/B，也不能因三次不同 token 用量推断 Memory 因果贡献。

随后单独预登记的一请求超预算探针实际返回两个空参数函数。服务保留 added、arguments.done、output_item.done 与完整终态，但没有 arguments.delta；旧门禁安全地以 MODEL_TOOL_CALLS_UNVERIFIABLE 整批拒绝，没有释放 SSE 或执行工具。因此这次原始记录不能宣称已证明 TOOL_CALL_BUDGET_EXCEEDED。官方 [Responses 流式事件定义](https://developers.openai.com/api/reference/resources/responses/streaming-events) 区分部分 delta 与完整 arguments.done；省略空参数 delta 的具体形状来自本次真实原始响应，且固定 PI0.85.1 消费者也使用 arguments.done 更新最终参数。

兼容修复仅允许没有 delta、但三处身份/完整参数都严格匹配的空 JSON 对象调用。非空参数仍需原始 delta 全量对应；已有 delta 不一致、任何 added/arguments.done/item.done 缺失均拒绝。七项新回归在旧源码为 3 FAIL/4 PASS，保留原始红测试；修复后预算与清单共 113 PASS。最初本地 venv 路径错误和测试 helper 重复关键字导致的失败也保留，不能当作有效旧源码反例。该修复不会追认历史任务执行新源码；真实新源码复验必须另行登记。


## 修复后源码704的完整真实 Native 集成复验

宿主预算门禁 `7043bbb4f5220db715219984d38cbb8dcd7df87e` 单独预登记并实际
完成 M0/M1/M2 三组隔离合成文件任务。模型、推理参数和预算仍为
`openai-codex/gpt-6.1-sol`、low、16请求、12释放工具调用、2048输出token/请求、
120000总token、600秒。镜像内 Core/Native 仍为旧5aa的V4，不能把宿主预算
源码704误称镜像源码。该镜像存在后来PR651修复的并发取消竞态；本文件任务
拒绝process-start，不验收镜像进程取消，也没有机器人或World。

| 模式 | 真实模型请求 | 释放工具调用 | 总 token | 原单调时钟耗时秒 |
|---|---:|---:|---:|---:|
| M0 | 9 | 8 | 44674 | 71.6957 |
| M1 | 10 | 9 | 58522 | 77.1634 |
| M2 | 10 | 9 | 62620 | 74.5255 |

三组 TaskKernel SUCCEEDED，交付Artifact的SHA和长度与原始字节一致，原容器均
停止/Pid0/exit0。独立复核用每笔真实请求与SSE、原始单调时间重建全部29次预算
决定，账本逐字段一致；26个释放工具调用与Native的toolCall/toolResult完全对应。
171个公开文件封存于 harness
`evidence/2026-10-09/n06-source704-live-tool-gate-three-real-native-synthetic-tasks`。
没有自动重试、替换失败、机器人验收、A0–A3正式对照或Memory因果收益结论。

## 修复取消竞态后的 V5 镜像真实 Native 文件任务

另外单独预注册并完成三组实际 Native 文件任务：镜像为
`sha256:77956d3b249f6290308cffdf0e497453985fd4c03dcc5dc43d213ec891fea320`，
Core/Native 为 `7dcc8861c9f5596cfa9d3201da4896ad04da2460`；宿主预算源码为
`c2112ae51d2cf49b179beaa22058cfc233f6c52d`（实际预算实现与704字节一致）。
这是新镜像运行，不能追认V4或旧源码任务。仍使用同一实际模型/low及上述
16请求、12释放工具调用、2048输出token/请求、120000总token、600秒登记预算。

| 模式 | 真实模型请求 | 释放工具调用 | 总 token | 原单调时钟耗时秒 |
|---|---:|---:|---:|---:|
| M0 | 9 | 8 | 44879 | 67.8139 |
| M1 | 9 | 8 | 48261 | 62.1040 |
| M2 | 10 | 9 | 59243 | 72.3116 |

三组TaskKernel SUCCEEDED，交付Artifact SHA/长度与原始字节一致，原容器全部
停止/Pid0/exit0。28次真实模型请求、25组toolCall/toolResult和原始单调预算账本
逐笔独立核验，168公开文件封存于harness
`evidence/2026-10-09/n05-v5-and-N06-current-real-native-file-tasks`。

该V5镜像另外通过两种真实并发进程取消组件测试（wrapper存活/已退出且孙进程
抗TERM），以及19字节中文/emoji Native写入；这属于独立隔离容器组件证据，
不是本文件任务执行机器人停止，也不是ROS Action grace的独立物理停止证明。
没有实际World、机器人任务、Memory因果收益或正式A0–A3结果。
本节为文档更新，预算代码未变；不把此前真实任务重标为新的文档提交。

## V6 当前镜像实际试验：两组通过、一组上游失败

新 V6 镜像 `a22b88b52a6c6712cca1c8f0a88c85878e47040a2c5e9b7f7e7a8b809f2828d9`
的 Core/Native 为 `4019018126b0844a3fe0013a20a643020dc344c6`，宿主预算
源码实际为 `bb58cfda78173c2b33093c6bbc9fd6c84c31926d`，实现仍与704字节一致。
V6另外修复并验证请求caller中断后的进程取消清理所有权；旧V5已通过的并发
取消组件不包含该情形，历史成绩不能当作不存在这一后续缺陷的证明。

第一份注册在模型调用前因helper保留旧OperationManager预期SHA失败，零请求、
零根任务，原失败保留。另行登记的attempt02修正预期SHA，保持原模型/low及
16请求、12释放工具、2048输出token/请求、120000总token、600秒预算：

| 模式 | 请求 / 完成请求 | 释放工具 | 权威usage已核验token | 任务结果 |
|---|---:|---:|---:|---|
| M0 | 9 / 9 | 8 | 47784 | SUCCEEDED，Artifact SHA对应 |
| M1 | 9 / 9 | 8 | 52050 | SUCCEEDED，Artifact SHA对应 |
| M2 | 2 / 1 | 1 | 4003，usage不完整 | FAIL，上游HTTP503，无根任务交付 |

M2第二次原始响应为HTTP503、`no healthy upstream`，没有权威usage。
门禁以`MODEL_RESPONSE_UNVERIFIABLE`拒绝释放该响应和工具，停止任务，没有
重试或替换。M2全部计费用量UNKNOWN；4003仅是第一次成功响应的已验证用量，
不能称为M2最终总量。整组三任务试验为FAIL，安全拒绝行为PASS不改写任务结果。

原请求/响应SHA、原始单调时钟预算决定、已释放工具与Native toolResult、两个
交付Artifact及三个原容器停止/Pid0逐项独立复核。公开原件封存于harness
`evidence/2026-10-09/n05-v6-attempt02-original-upstream-failure-and-two-deliveries`。
精确上游故障根因UNKNOWN；未发生World、机器人任务、Memory因果或正式A0–A3。
本节仅更新文档，不将实际bb58调用重标为本次文档提交。
