# Phase 8：MotionDecode 同步触球先验与主动接触学习闭环

日期：2026-08-11
范围：G1 / MuJoCo / SIM_ONLY；没有连接真实机器人，没有发送硬件命令。

## 1. 结论

本轮打通了一个可审计的闭环，但尚未打中 8.5 m 球门的左上角
`(y,z)=(3.40,2.18) m`，所以没有制作或宣称新的成功宣传视频。

已完成的工程突破：

1. 修复 MotionDecode 旧先验把“多条动作的中位姿态”和“单条动作的代表速度”
   拼在一起的问题。新 v6 `vertical_drive` 让 29 关节姿态与速度来自同一条射门、
   同一参考帧，保留触球因果。
2. 新增不确定性驱动的主动采样器。critic 交叉验证失败时，它只在最佳硬安全锚点
   周围、同一局部切片至少三个安全点支持的关节轴上填一个未测间隙；一次只改一个
   关节，最大步长 0.03 rad。
3. 新增严格候选评估器。actor 的预测不能替代物理结果；锚点、候选动作、轨迹、
   Body、实现、任务上下文和证据文件均用哈希绑定。精度、恢复稳定性或硬安全任一项
   不满足就输出 `REJECTED`。
4. 以四路并行 worker 运行候选，MuJoCo 物理仍在 CPU；4×A6000 只作为可用计算资源
   隔离，不宣称 GPU 物理训练。

本轮最重要的负结论也已固化：MotionDecode v1 CSV 没有同步足球状态、接触力、动作、
奖励或转移。它可以改善人体射门姿态先验，但不能单独提供准确的球—脚冲量策略。

## 2. MotionDecode 同步先验

完整本地快照包含 1,204 个注册文件。修复报告中 1,111 条达到 Q1，足球射门训练分区
有 107 条；`vertical_drive` 筛选出 13 条合格事件并选取 8 条训练事件，没有读取留出
指标。

代表动作：

```text
samples/3.3.Ball_Game_Interaction/3.3.3.Football/
3.3.3.3.Shooting/BGI_Shooting_00085.csv
```

- 参考帧：369 @ 120 Hz；
- 右脚速度：前向 4.9005 m/s、横向 0.9977 m/s、向上 3.3661 m/s；
- 工件中的 29 关节位置与原 CSV 对齐最大误差：0；
- 工件中的 29 关节速度与原 CSV 对齐最大误差：0；
- `position_distillation_strategy` 与 `velocity_distillation_strategy` 均为
  `synchronized_representative_event`；
- 最终 v6 工件哈希：
  `sha256:bd54d6deba4af700fa2ddab04a67614cb0a25a2842b8ad7bb98a680dcbf0482c`；
- 8 个事件的有界得分范围：0.6408–0.8527；
- activation ceiling：`SIM_ONLY`；promotion/hardware：false。

旧 v3/v4/v5 工件继续以原哈希加载，兼容哈希分别为：

```text
v3 sha256:44b17fd6a88d19487e61ed32104436ed7b6d5e889521bc9a4000314034bc1041
v4 sha256:71a0440ca6fec271a5bc5542085a093c2e37eb17f9af75f5736cc7d36298b0fe
v5 sha256:4894f7870ce44c705a7e2bcfa56c5f034ff016f476f7ee0f6f265ded65f1c1a9
```

## 3. 主动学习闭环

产品命令：

```text
rosclaw growth ballistic-contact-active-sample \
  --actor-critic actor-critic.json \
  --maximum-step-rad 0.03 \
  --output active-sample.json

rosclaw growth evaluate-ballistic-contact-candidate \
  --actor-critic actor-critic.json \
  --anchor-evidence anchor/g1-free-kick.json \
  --candidate-evidence candidate/g1-free-kick.json \
  --minimum-error-improvement-m 0.005 \
  --output evaluation.json
```

闭环顺序：

```text
MotionDecode 同步姿态/速度
  -> MuJoCo 成功/失败接触回放
  -> 支持秩约束 actor-critic
  -> critic 不可靠则主动补一个局部点
  -> 候选重新进入真实物理
  -> 哈希绑定的精度 + 稳定性 + 硬安全评估
  -> ACCEPTED 或 REJECTED；预测永不直接热更新
```

第一代混合四个关节轴的 critic 留一 RMSE 为 4.1527，超过 0.15 门限，正确返回
`sim_replay_recommended=false`。主动采样器随后把搜索收缩到有密集局部支持的右踝
pitch 轴。8 个局部样本把留一 RMSE 降到 0.0350，actor 才允许提出 `.18 rad` 候选。

实际候选没有被接受：

| 指标 | 锚点 `.21 rad` | actor 候选 `.18 rad` |
|---|---:|---:|
| 左上角误差 | 2.459767 m | 2.461383 m |
| 门线 `(y,z)` | (1.2424, 0.9989) m | (1.2734, 0.9406) m |
| 触球后关节速度峰值 | 4.6983 rad/s | 4.7261 rad/s |
| 稳定时间 | 3.3180 s | 3.3380 s |
| 硬安全 | 通过 | 通过 |

实测精度改善为 `-0.001615 m`，低于要求的 `+0.005 m`，因此自动凭据为：

```text
decision=REJECTED
accepted=false
precision_improved=false
stability_preserved=true
hard_safe=true
```

## 4. 物理发现

保持成熟的 RoboNaldo 射门形态非常重要。把动作参考点直接推到物理左上角会破坏
基础挥腿，前向出球约从 9–10 m/s 降到 7 m/s。恢复合格动作参考后，同步先验
`position=0.05 / velocity=0.02` 将门线高度从 0.134 m 提升到 0.949 m，apex 从
0.901 m 提升到 1.384 m，出球垂向速度从 3.851 提升到 4.934 m/s；但横向位置从
2.083 m 回落到 1.179 m，说明高度与横向控制仍然耦合。

层次化朝向 + 关节组合找到的最佳硬安全诊断点把误差降到 2.120 m，门线位置为
`(1.924,0.658) m`，但触球后关节速度峰值升到 6.051 rad/s。它不能作为稳定冠军，
只证明骨盆朝向、髋偏航和踝俯仰存在必须学习的耦合项。

将挥腿速度从 1.15 提到 1.20/1.30/1.40/1.50 没有形成更多能量：接触时序跨过
窄接触岛，1.30/1.40 只产生弱触球，1.50 直接漏球并跌倒。失败回放说明下一步不能
靠“更快/更大力”的开放环补丁。

## 5. 验证与证据

- Ruff：通过；
- mypy：通过；
- growth 全套：131 passed；
- 定向新模块：18 passed（包含哈希篡改、缺失饱和指标、非完整二维网格、单点不安全、
  浮点 JSON 往返及 CVaR 尾部退化 fail-closed）；
- 项目级非 slow 回归在人工中断前完成 2,685 passed / 58 skipped / 55 deselected，
  出现 7 个与本轮文件无关的环境/存量失败：会话时间排序 1 项、本机已有 Codex CLI
  导致“二进制缺失”假设失效 2 项、LeRobot runtime 配置不一致 4 项；随后停在既有
  SSL/外部访问等待。更早的 release 打包测试也因在线 `pip download` 长时间无进展
  被中断。因此不把项目级尝试计为全量通过，也没有发现 growth 定向回归失败。

关键外部证据：

```text
/code/rosclaw/rosclaw_football/evidence/
  motiondecode-synchronized-vertical-drive-prior-v1.json
  motiondecode-synchronized-vertical-drive-prior-v2.json
  motiondecode-upper-corner-posture-actor-critic-v1.json
  motiondecode-upper-corner-active-sample-v2.json
  motiondecode-upper-corner-anklep-actor-critic-v2.json
  motiondecode-upper-corner-anklep-actor-evaluation-v2.json
  motiondecode-upper-corner-coupled-grid-v1/
  motiondecode-upper-corner-coupled-actor-critic-v2.json
  motiondecode-upper-corner-coupled-candidate-v1/
  motiondecode-upper-corner-coupled-evaluation-v1.json
  motiondecode-upper-corner-coupled-holdout-v1/
  motiondecode-upper-corner-coupled-context-evaluation-v1.json
```

所有工件与回放保持在源码 checkout 外；所有新增学习与评估工件均不允许 promotion、
在线热替换或硬件执行。

## 6. 二维耦合与跨状态 Stability–Plasticity 门控

单轴 critic 不允许假设关节间没有交互，因此新增
`ballistic-contact-coupled-actor-critic`。它只有在两个轴形成完整笛卡尔网格、每轴
至少三个取值、其余四轴完全冻结，而且九个点全部硬安全、动作连续时，才拟合
`[x, y, x², xy, y²]`。这使“交互项”来自实测而非数学臆造。

本轮对右髋 yaw `(-0.08,-0.06,-0.04)` 与右踝 pitch `(0.18,0.20,0.22)` 运行了
9 个完整 MuJoCo 回放。全部通过连续性、无跌倒、无限位、无饱和；critic 留一 RMSE
为 `0.000898`。网格最佳点位于 `(-0.08,0.22)` 边界：

| 点 | 门线 `(y,z)` m | 左上角误差 m | 触球后关节速度峰值 rad/s |
|---|---:|---:|---:|
| 最佳实测锚点 | (1.2242, 1.0464) | 2.453410 | 4.7098 |
| 半步外环 `(-0.09,0.23)` | (1.2069, 1.0799) | 2.453507 | 4.7116 |

为避免在可靠单调边界过早停止，critic 允许提出至多半个网格步长的
`NEXT_RING_HALF_STEP`，但它只拥有“再花一次仿真”的权限。候选将高度提高了
`0.0335 m`，同时损失横向位置，综合误差退化 `0.000097 m`，严格物理评估输出
`REJECTED`。这不是冠军更新。

这次回放还发现并修复了动作绑定的浮点缺陷：工件中的
`0.22999999999999998` 经 CLI/JSON 表示为 `0.23` 后，旧代码用 tuple 精确相等会在
指标门控前错误拒绝。现改为 `1e-9 rad` 绝对容差；哈希、轨迹和动作维数约束保持。

然后以 seed 4/7/21 对锚点与候选做三组成对留出回放：

| planner seed | 锚点误差 m | 候选误差 m | 改善 m | 连续性 |
|---:|---:|---:|---:|---|
| 4 | 4.611899 | 4.664014 | -0.052115 | 通过 |
| 7 | 2.368798 | 2.253358 | +0.115440 | 通过 |
| 21 | 3.373518 | 3.375820 | -0.002302 | 失败 |

平均改善虽为 `+0.020341 m`，但 25% 下尾 CVaR 与最坏状态均为
`-0.052115 m`，改善状态数只有 1/3（要求至少 2/3），且 seed 21 连续性失败。
新增 `evaluate-ballistic-contact-context-holdout` 因此输出 `REJECTED`。这正是
Stability–Plasticity 的工程实现：允许探索获得新知识，但不能让局部幸运收益覆盖
跨状态的肌肉记忆冠军。

## 7. 下一阶段

1. 训练以 handoff yaw/roll/pitch、骨盆速度、支撑腿状态和球相对足端位置为输入的
   状态条件化接触 actor，而不是继续使用六个常数。
2. 将目标 `(y,z)` 显式作为 actor 条件，分别学习横向与垂直球速，再用真实弹道误差
   训练耦合 critic。
3. 将本轮已跑通的 CVaR/最坏状态门控接到 episodic contact memory，让运行时按真实
   本体状态选局部专家，而不是把任何常数残差冒充可泛化小脑。
4. 达到 0.1 m 左上角精度并跨多个 planner seed 通过后，再生成新的成功宣传视频；
   在此之前只输出带 `REJECTED / DIAGNOSTIC / SIM_ONLY` 水印的诊断视频。
