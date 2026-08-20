# Phase 8：球状态与目标条件化触球小脑

日期：2026-08-11

边界：`SIM_ONLY`；物理真值来自 CPU MuJoCo 连续世界。没有连接真实机器人，没有发送 ROS、DDS、串口、CAN 或厂商 SDK 命令。4 张 A6000 不替代本轮接触物理验证。

## 1. 结论

本轮把旧的“固定目标脚速补丁”升级成了数据驱动、球状态与目标点条件化的局部触球小脑，并用严格基线消融证明了一个有限但真实的开发突破：

- held-out 目标：`(x,y,z)=(8.5, 2.139, 0.765) m`；
- 无 actor 基线误差：`0.0192517 m`；
- v2 actor 误差：`0.0056551 m`；
- 绝对改善：`0.0135966 m`；相对改善：`70.63%`；
- actor 实际激活 1 个控制帧，输出横向 `30 N`、竖直 `0 N`，峰值解码关节力矩 `14.224 Nm`；
- 接触任务权威最小值 `1.0`，无倒地、无后退、无执行器饱和；
- 原目标所需弹道速度超出已学习包络时，actor 记录 2 个 OOD 帧并保持零输出，物理轨迹与零力基线一致。

Growth 评估器给出 `DEVELOPMENT`，而不是 `PASS` 或 promotion。完整 free-kick showcase 仍为 `passed=false`：当前固定助跑上下文本身在原目标上误差很大，且跑动最低骨盆高度 `0.659 m` 低于完整门的 `0.70 m`、跑动峰值倾角 `0.496 rad` 高于 `0.30 rad`。单目标、单 seed 的局部精度改善不构成封闭泛化证据。

## 2. 架构变化

### 2.1 旧 v1 的问题

旧 `G1BallisticContactImpulseActor` 只输入：

```text
[bias, right_foot_vy, right_foot_vz]
```

并把一个固定教师目标脚速蒸馏成二维线性 PD。其实验上下文绑定单一球门目标，因此改变目标高度、横向位置或来球速度后，策略没有条件变量可用于适应。

### 2.2 v2 目标条件化 actor

v2 在接触门内读取：

- 球的三维位置与线速度；
- 球门平面和目标 `y/z`；
- 实测右脚端 `vy/vz`；
- 训练得到的安全出球速度包络与任务力包络。

运行时先按弹道关系计算所需横向/竖直出球速度，再通过局部接触动力学的正则化逆映射生成任务空间力，最后使用当前 MuJoCo Jacobian 转成 29 维关节力矩。输出继续经过现有接触任务权威投影、总力矩投影、关节边界保护与 `SIM_ONLY` 边界。

v1 schema、字段集合与语义哈希保持兼容；v2 使用独立 schema：

```text
rosclaw.growth.g1_ballistic_contact_impulse_actor.v2
```

### 2.3 前向系统辨识，而不是直接逆回归

第一版尝试直接拟合：

```text
期望出球速度 -> 教师任务力
```

它在 held-out 目标上虽然达到 `0.0405 m`，但无 actor 基线为 `0.0299 m`，实际退化，因此被消融否决。

最终版本把零力严格回放加入 rehearsal，先学习：

```text
[lateral_force, vertical_force] -> [ball_launch_vy, ball_launch_vz]
```

再对局部前向 Jacobian 做正则化伪逆。这样自然动作偏置由零力截距解释，附加任务力只学习局部可控变化，避免把“本来就会发生的出球”错误归功于 actor。

### 2.4 Stability-Plasticity 边界

- 稳定性：绑定零力基线和所有拒绝探针；超出已见出球速度包络时拒绝激活；任务力只能落在严格安全样本实际覆盖的 `[0,30] N`，不允许对称限幅偷偷产生未见过的负向冲量。
- 可塑性：目标与球状态改变时重新计算弹道条件，只在已辨识的局部接触岛内输出不同任务力。
- provenance：运行前强制比较 actor 的 Body、实验上下文、artifact hash 与 `implementation_hash`；代码变化后的旧 actor 不能静默执行。
- 权限：`promotion_authorized=false`、`hardware_authorized=false`、`online_hot_swap_allowed=false`。

## 3. 数据闭环

最终 v4 训练包包含 9 条同实现、同 Body、同助跑和安全上下文的严格轨迹：

- 1 条零任务力基线；
- 5 条接触任务权威保持 `1.0` 的低力安全探针；
- 3 条被权威缩放的高力拒绝探针，最小权威比例约 `0.407–0.736`；
- 所有探针都无倒地、无执行器饱和；
- 安全拟合集中的二维任务力覆盖 `0–30 N`；
- 9 条样本均未通过原目标的 `0.10 m` 精度门，因此继续作为 reject rehearsal 绑定，不能被删掉来美化数据。

前向动力学训练内 RMSE 为 `0.000210 m/s`。该数值很低，但样本来自单 seed 的窄局部接触岛，只能说明局部拟合一致，不能当作跨 seed 泛化成绩。

## 4. 严格对比

| 回放 | 目标误差 m | 过门 `(y,z)` m | actor 力 `(Fy,Fz)` N | 倒地/后退/饱和 | 判定 |
|---|---:|---:|---:|---|---|
| 同目标零力基线 | 0.019252 | (2.129017, 0.748539) | (0, 0) | 否 / 0 / 否 | 基线 |
| v2 条件化 actor | **0.005655** | **(2.138378, 0.770621)** | (30, 0) | 否 / 0 / 否 | DEVELOPMENT |
| 原目标稳定锚点 | 0.859917 | (2.129017, 0.748539) | (0, 0) | 否 / 0 / 否 | OOD fail-closed |

actor 与基线的触球后稳定时间同为 `3.234 s`。峰值骨盆速度从 `1.1016` 到 `1.1038 m/s`，变化很小；峰值关节速度 RMS 从 `6.3740` 降到 `6.3692 rad/s`。本轮证明没有明显恢复退化，但没有证明小脑稳定性得到大幅提升。

## 5. 新评估器

新增产品命令：

```text
rosclaw growth evaluate-target-conditioned-contact \
  --actor ... \
  --baseline-evidence ... \
  --candidate-evidence ... \
  --stability-anchor-evidence ... \
  --output ...
```

评估器强制验证：

- 三条 evidence/trajectory 内容哈希；
- actor artifact、Body、实现与目标条件化上下文；
- 基线没有 actor 污染；
- 候选实际执行绑定 actor 且至少激活 1 帧；
- 稳定锚点因 OOD 拒绝而非错误执行；
- 候选相对基线至少改善 `1 mm`、进入精度半径并满足硬安全。

即使全部满足，单目标评估也只能返回 `DEVELOPMENT`；它在数据结构上硬编码 `sealed_generalization_evidence=false`、`promotion_authorized=false` 和 `hardware_authorized=false`。

## 6. 关键 artifact

- actor hash：`sha256:0feace693407dd31edd34848eb42717cd781440ff44eb47d464ab237bc3d058b`
- actor 文件 hash：`sha256:03258f449e874609e348e49f17e77b5d90c2388e299625657cfd5b953ca6b620`
- 物理实现 hash：`sha256:9b0b59ded4c77131f51b8f7d56de4018311239eb1678f7ab944c178804c61391`
- Growth evaluation hash：`sha256:a3ef48575f4b0cd189eac95b8e44cad8e6047c091f26b9e173be6c476473bf27`
- 原始外部证据：`/code/rosclaw/rosclaw_football/evidence/age10-target-conditioned-probes-v4/`

## 7. 当前局限与下一步

1. 当前局部岛只覆盖一个 planner seed 和很窄的出球速度范围，左上死角 `(3.4,2.18) m` 仍远在包络外，actor 会拒绝执行。
2. 当前只控制横向/竖直任务力，没有学习前向冲量、支撑状态、质心、骨盆姿态和接触时间；不能替代完整端到端小脑。
3. 当前稳定锚点证明“不会乱动”，没有证明旧目标技能成功；下一轮必须换成真实通过的冻结低角冠军上下文做 rehearsal。
4. 需要按 planner seed、球初速、摩擦和球质量分组采样，并用 sealed seed holdout；训练批次要混入冻结冠军，优化最坏分位数与 CVaR。
5. 下一阶段应扩展前向模型输入到足—球相对位置、足端三维速度、支撑脚接触、骨盆姿态/角速度、可用力矩余量，并先输出受保护的短时任务空间残差。
6. 只有跨 seed、跨目标、移动来球和恢复门都通过，才允许写入稳定肌肉记忆；真正左上角仍未突破，本轮不生成“死角成功”宣传视频。

## 8. 代码验证

- 定向 Growth/SimForge 测试：`43 passed, 2 deselected`；
- Ruff：通过；
- mypy：1220 个源文件通过；
- 完整非 slow 回归原始结果：`6838 passed, 114 skipped, 19 deselected, 6 failed`；6 个失败均为环境假设：2 个“Codex 缺失”用例在隐藏本机 Codex PATH 后复跑 `2 passed`，4 个 LeRobot runtime 绑定用例在显式绑定 LeRobot 0.6.1 后复跑 `4 passed`；
- 当前本机 Ruff formatter 对主干 302 个既有文件报告 would-reformat，因此没有在本轮批量改写无关文件；`ruff check src tests` 通过；
- 所有正式结果来自严格 CPU MuJoCo evidence，像素不参与评分。
