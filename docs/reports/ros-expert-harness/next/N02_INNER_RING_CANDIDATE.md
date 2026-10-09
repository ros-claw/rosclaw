# N02 已知机体附加内缩边界环候选：组件实现，实际验收未运行

## 动机与事实边界

本候选基于已闭合 Burger pilot `101204` 的只读诊断，不使用正式 evaluation 数据。现有 `3f4cd0c0` 候选的主覆盖加边界实测为 3,055 / 3,588 = 85.1449%；随后 59 个补扫目标贡献 464 个新单元，补扫阶段 655.48 SIM 秒、原地转向 221.92 秒。六个补扫目标没有新 credit。单次数据只能定位瓶颈，控制器根因仍 UNKNOWN，不代表整组统计。

对该次真实 primary mask 做 12 个离线几何评分（9 个单环、3 个双环组合）。在原 boundary center mask 内缩一格（该地图约 5 cm）的附加环，预测可补 215 个漏扫单元，primary 加预测环为 91.1371%，参考长度约 9.84 m。该数值是理想路径/刷具几何预测，没有实际 Nav2、额外清扫 credit 或耗时收益。它仍需后续实测补扫；新增沿边耗时可能抵消收益。

原始只读诊断与评分位于本地 harness 的 `recovery_2026-10-09/n02-burger101204-controller-turns-diagnostic.json` 和 `n02-burger101204-inner-ring-offline-screen.json`，保留输入哈希。实际物理来源是 `r9/b101204` 与已封存的同名完整配对 evidence。

## 显式候选行为

新增 preset `perimeter_stateless_clearance_inner_ring`，仅接受已知 Burger。其主覆盖参数与控制器参数完全继承 `perimeter_stateless_clearance`。在已有九个边界目标后，增加九个内缩一格的闭合环目标；两环共 18 个目标，均通过独立 daemon 内现有 Nav2 NavigateToPose 调用，不直接发布 ROS 指令。

`inset_rectangular_boundary_targets()` 只从原 boundary mask 内选点，验证网格间距及整条内环参考边均为已有合法中心；它不扩大 center domain。内部环、target 或静态预测不更新 coverage。真实独立 brush-on 轨迹与 canonical verifier 继续决定所有 credit。

每个目标仍至多 45 秒；附加环候选的 boundary stage 上限显式登记为 360 秒，已有单环为 180 秒。不可变全任务 deadline、原目标超时、速度、清扫分母、物理几何、租约与停止检查沿用。新增预算不延长全任务授权时限；全任务耗尽时仍在原执行入口终止。

原协议不能直接启动新候选。paired runner 在启动 World 前要求严格整数：

```json
{
  "candidate_boundary_stage_budget_sec": 360,
  "candidate_inner_boundary_inset_cells": 1
}
```

SDK experiment 记录相应阶段预算、内缩格数与 `sequential_inner_ring` 策略。独立 daemon 在建立 Runtime 前再次核对 preset、已知 Body、策略、显式开关和预算；不匹配立即拒绝。每个目标的实际结果、阶段和唯一 ID 继续进入原审计链。

## 验收与下一步

组件测试检查完整合法内环、缺边拒绝、参数类型、两环目标来源、故障/阶段超时停止、原全任务 deadline 传递、预注册拒绝及实际 daemon CLI 的启动前拒绝。SDK 准备与源码检查单独记录，不算 World 或机器人验收。

**实际新候选 physics / Native / paired diagnostic 均 NOT_RUN。** 当前 `3f` 五组 pilot 与其条件正式评估队列使用此前冻结源码和协议。本候选不混入该系列，也不替换任何失败。后续若试跑，须在现有完整系列收尾后使用新提交、新注册种子、完整原始证据和独立停止验收。单对改善不证明 30% 双目标；若晋级完整系列，仍需按实施方案重新冻结并完成适用的两 Body 5 pilot + 10 evaluation 配对验收。

## Waffle 候选与五组训练数据的适用性

新增 `perimeter_stateless_inner_ring`，只接受已知 Waffle，主覆盖与控制器参数和
`perimeter_stateless` 相同。它复用上述两环 18 目标、1 格内缩、360 秒阶段上限、
45 秒单目标上限与原全任务 deadline。Burger 原有候选不变；两个 preset 与机体
严格绑定，互换机体或更改预注册阶段字段会在 World/Runtime 建立前被拒绝。
这属于 N02 已知机体效率试验，不是 N04 未见 Body 泛化。

对五组已闭合 Waffle 训练 pilot 分别重放真实 primary mask，每组只离线比较
1–4 格内缩及三个双环组合，没有使用 evaluation 数据。下表的新增单元和合并
比例都是几何预测，不能作为实际清扫 credit、Nav2 可达性或时间收益。

| 训练 seed | 实测 primary | 一格环名义新增单元 | primary 加名义环 | 原沿边结果 |
| --- | ---: | ---: | ---: | --- |
| 101201 | 91.19% | 183 | 96.31% | SUCCEEDED |
| 101202 | 91.78% | 174 | 96.64% | SUCCEEDED |
| 101203 | 91.16% | 173 | 96.00% | SUCCEEDED |
| 101204 | 91.64% | 165 | 96.25% | SUCCEEDED |
| 101205 | 63.48% | 1104 | 94.35% | 第一个沿边目标 45 秒超时 |

最后一行尤其不能误读为候选解决了既有失败：顺序执行遇到外环目标失败就返回，
后面的内环不会执行。因此，该行只是从历史 mask 出发的反事实几何计算，对原
失败执行分支 **不适用**。新增内环没有修复该次超时，根因仍需实际轨迹、反馈
和控制器诊断。新试验仍保留原有失败回退和停止边界。

其余四次名义参考环约 9.04–9.05m，仍需补扫才能达到98%；实际内环跟踪、
额外耗时、原始几何最小净距和新的完整配对资格 **NOT_RUN**。当前正式系列
仍使用冻结 `3f`，本候选不能替换其中任何试验或放宽双30%工程目标。
