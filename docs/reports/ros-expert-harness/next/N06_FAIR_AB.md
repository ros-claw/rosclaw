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
