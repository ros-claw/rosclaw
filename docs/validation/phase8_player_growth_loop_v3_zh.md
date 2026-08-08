# Phase 8：G1 球员化 Growth 闭环 v3 实施报告

## 结论

本轮不再把“跑得更快”或“一次进球”当成突破，而是首次把连续自由球实验接成了可审计的 ROSClaw Growth 闭环：

1. 29 关节 SONIC 全身神经策略负责助跑和刹停；
2. 单一 MuJoCo 世界连续进入 RoboNaldo 射门和踢后恢复，不重置机器人；
3. 轨迹按 `APPROACH → ALIGN_BRAKE → PLANT_BRIDGE → LOAD → SWING → CONTACT → FOLLOW_THROUGH → RECOVERY → READY` 切片；
4. 每帧记录 commanded、safety-projected、executed 三态力矩；
5. 9 个独立、严格复跑的接触课程生成 1262 条离线 RL 转移；
6. IQL 候选完成训练并进入有界残差物理回放；
7. 候选因实际物理退化被自动拒绝，没有用训练 loss 冒充球技提升；
8. 从 action triplet 进一步学出逐关节 authority calibration，显著减少力矩触顶。
9. 将原先只匹配位置、会把关节速度硬刹为零的 bridge 改成可审计的五次 Hermite 速度匹配桥；
10. 对入桥速度、射门先验速度、接入相位和 aim 运行 15 个严格双回放 C1 课程，验证了速度连续性的收益和接触模式的不连续边界。

当前仍没有达到宣传级“真正球员”。C1 最均衡候选在 calibration v4 下把门线目标误差降到 0.039 m、下角距离降到 0.161 m、踢球峰值倾斜降到 0.296 rad，并把触顶物理步压到 2；但仍未达到零触顶门，动态接入相位 170 的零触顶候选又出现 0.370 m 横向过冲，因此全部仍是 `REJECTED / SIM_ONLY`。

## 一、ROSClaw 新增与打通的模块

### 1. SONIC 全身神经身体基础

`g1_sonic_runup.py` 现在执行完整的 planner + low-latency encoder/decoder：

- planner 输出 30 Hz 全身参考；
- encoder 输入 1247 维未来参考与姿态条件；
- decoder 输入 994 维 token 与 10 帧本体感历史；
- decoder 在 50 Hz 输出 29 关节动作；
- 500 Hz PD 在 MuJoCo 内闭环，才产生实际关节力矩；
- ONNX 文件、输入输出形状、关节映射和模型哈希均 fail-closed。

这不是 qpos 播放，也不是视频动画补丁。

### 2. 连续自由球 CLI 与产品入口

`rosclaw goalforge free-kick-showcase` 之前会被兼容 CLI 的封闭命令表提前拒绝。本轮修复了入口顺序，并把以下课程变量接入正式 CLI 和请求哈希：

- 起点 x/y；
- SONIC 跑速、刹停速度、执行时长、planner seed；
- bridge 时长、射门起始相位；
- aim bias、骨盆 yaw、脚 yaw、COM 横移；
- 球门目标 y/z；
- authority calibration；
- approach-to-strike IQL 候选与残差上限。

### 3. PARC 事件和动作真值

自由球轨迹新增稳定事件编号和三态动作：

| 动作字段 | 含义 |
|---|---|
| `commanded_torque` | 未经过 authority guard 的上游 PD/IQL 请求 |
| `safety_projected_torque` | 按硬力矩边界投影后的请求 |
| `executed_torque` | MuJoCo actuator 实际执行值 |

适配器会验证三者形状、有限性、硬边界、投影标志以及 executed 与 projected 的一致性。旧 v2 轨迹因缺少动作真值只能被路由为 safety-blocked；v3 轨迹可进入 ILC / motion-tracking 路由。

### 4. Approach-to-strike 数据集

新增 `growth/approach_strike_dataset.py` 和正式命令：

```text
rosclaw growth approach-strike-dataset ...
```

每条转移包含：

- 110 维 state / next-state；
- commanded / projected / executed 29 维动作；
- 6 维 reward vector；
- 5 维 safety cost vector；
- 事件阶段、episode、frame、projection flag；
- 原始关节目标 kinematic reference。

数据集要求 trajectory/evidence 一一配对、严格复跑、身体哈希和实现哈希一致、轨迹哈希独立。训练门要求至少 8 个独立触球回合。失触回合不会被悄悄塞进以 CONTACT 为终点的训练集。

### 5. Manifest 驱动的通用 IQL

旧 `train-iql` 只接受踢后恢复的固定状态维度。本轮改为读取 manifest 中的：

- task id；
- state/reward/cost 名称和维度；
- reward/cost scalarization；
- environment hash；
- episode split。

旧恢复数据仍兼容；新 approach-to-strike 数据也能走同一个安全 NPZ、无 pickle 的候选训练链。

### 6. 支持域约束的力矩残差

IQL 不能直接接管 29 个电机。新增的 residual controller 只在 `ALIGN_BRAKE → early SWING` 生效：

- 标准化状态 RMS 或最大值超出训练支持域时立即回退；
- 每关节残差先限幅，再乘残差比例和置信度；
- 当前峰值实际残差为 0.921 Nm；
- 所有输出仍经过原有硬力矩投影；
- 候选只能是 `SIM_ONLY / CANDIDATE_UNEVALUATED`。

### 7. 数据驱动 authority calibration

新增：

```text
rosclaw growth sonic-authority-calibration ...
```

它从 9 个回合的 APPROACH/ALIGN 和 FOLLOW_THROUGH action triplet 取 99.5% 分位力矩需求，推导 29 关节 gain scale。结果只修改真正过载的关节：

- 左/右 ankle pitch 是助跑主要过载源；
- 一个 hip 关节是触球后主要过载源；
- 其余大部分关节保持 1.0，不做无差别“全身变软”。

artifact 绑定全部源轨迹哈希、body hash、implementation hash 和标定参数。

### 8. C1 速度匹配 transition bridge

旧 bridge 使用位置 minimum-jerk 曲线，但 PD 阻尼目标始终是零速度；这会在 SONIC 仍有约 0.59 rad/s 关节 RMS 速度时主动刹停，再从近静止进入射门先验。现在新增 `g1_transition_bridge.py`：

- 五次 Hermite 同时约束 29 关节的入口/出口位置、速度和零边界加速度；
- 入口/出口速度比例与最大边界速度都进入请求哈希；
- 所有边界非有限、维度错误或超配置范围都会 fail-closed；
- 零速度配置逐公式复现旧 bridge，支持控制器等价 A/B 基线；
- 轨迹新增 `controller_target_velocity`；
- 结果新增入口速度 RMS、目标出口速度 RMS、出口速度误差和峰值目标加速度。

这不是新的动作补丁：它把 transition actor 将要学习的相位速度和支撑时序变成了显式、可回放的低维控制边界。

## 二、接触课程实验

共运行 episode-00 至 episode-08。episode-06 为失触并摔倒，保留给未来 contact-mode classifier；IQL 数据集使用另外 8 个实际触球回合。

| 回合 | seed | 触球 | 门线误差 | 摔倒 | 触顶物理步 | 结论 |
|---|---:|---|---:|---|---:|---|
| 00 | 0 | 是 | 0.164 m | 否 | 73 | 基线，精度门略失败 |
| 01 | 1 | 是 | 1.285 m | 否 | 216 | 大幅偏门，含 joint-limit 失败 |
| 02 | 2 | 是 | 0.273 m | 否 | 180 | 接触模式变化 |
| 03 | 3 | 是 | **0.030 m** | 否 | 106 | 精准，但 authority / 跑姿失败 |
| 04 | 4 | 是 | 未过门线 | 否 | 90 | 弱接触负样本 |
| 05 | 5 | 是 | **0.013 m** | **是** | 94 | 精准但跌倒，不能晋升 |
| 06 | 6 | **否** | 无 | **是** | 53 | 失触分类负样本，不进 CONTACT 数据集 |
| 07 | 7 | 是 | 未过门线 | 否 | 39 | 慢速/弱接触负样本 |
| 08 | 0 | 是 | 可过门 | 否 | 77 | 相同 seed、不同边界状态 |

这一组结果证明接触是混合动力学分支：几厘米起点、相位或 seed 改动会造成“准确进角、偏门、弱接触、失触、摔倒”之间跳变。线性 aim 插值不足以解决，需要 contact-mode classifier 和 mixture-of-experts transition actor。

## 三、IQL 训练与闭环判决

### 数据

- 8 个独立触球 episode；
- 1262 条 transition；
- 6 个训练 episode，1 个验证 episode，1 个完全保留 episode；
- `promotion_truth_allowed=false`。

### 离线训练

800 步 CPU IQL：

| 指标 | 结果 |
|---|---:|
| 验证 normalized MSE before | 0.7788 |
| 验证 normalized MSE after | 0.1395 |
| 相对下降 | 82.1% |
| 状态 | `CANDIDATE_UNEVALUATED` |

### 真正物理回放

| 指标 | 无残差基线 | IQL 残差 | 变化 |
|---|---:|---:|---:|
| 残差作用占基线 RMS | 0 | 3.09% | 未达到 5% 最小可测作用 |
| 门线误差 | 0.1640 m | 0.1675 m | 退化 0.0035 m |
| 触顶物理步 | 73 | 78 | 退化 5 |
| 踢球峰值倾斜 | 0.4138 rad | 0.4372 rad | 退化 0.0234 rad |
| 最终机身速度 | 0.000519 m/s | 0.000354 m/s | 略改善 |

自动门结论为 `REJECTED`：

- `minimum_learned_effect=false`；
- `authority_non_regression=false`；
- `tilt_non_regression=false`；
- `absolute_task_gate=false`。

这说明 loss 下降代表“更会拟合数据动作”，不代表“闭环更会踢球”。该结果是闭环本身的重要突破：ROSClaw 已能拒绝一个看起来训练成功、实际物理退化的模型。

## 四、authority calibration 结果

| 候选 | 触顶物理步 | 峰值需求比 | 门线误差 | 下角距离 | 结论 |
|---|---:|---:|---:|---:|---|
| 未校准 SONIC | 73 | 1.311 | 0.164 m | 0.364 m | REJECT |
| calibration v1 | 26 | 1.228 | 0.137 m | 0.337 m | REJECT |
| calibration v3 | **6** | **1.046** | **0.112 m** | 0.312 m | REJECT |
| calibration v4 / aim 0.6 | 2 | 1.008 | 0.211 m | 0.411 m | REJECT |

v3 是当前较均衡的开发候选：力矩触顶下降 91.8%，门线精度提高 31.6%，没有摔倒，踢后最终速度为 0.00027 m/s。但跑动最低骨盆 0.666 m、跑动峰值倾斜 0.489 rad、下角距离 0.312 m，且仍有 6 个触顶物理步，所以不能改写为 PASS。

v4 继续压 authority 能把触顶降到 2～3 步，但过软 ankle 改变了接触状态，落点进入另一个分支。安全和精度之间的非单调耦合进一步说明下一步不能只做单标量调参。

## 五、为什么仍不像真正球员

视觉上的主要边界仍是：

1. SONIC 学会的是通用跑动，不知道“下一步要射门”；
2. RoboNaldo 学会的是定相位射门，不知道“上一刻正在跑”；
3. 0.35 s bridge 虽然连续且没有停顿，但仍是两个动作语汇之间的插值；
4. 当前 IQL 学的是全力矩模仿，数据太少，且没有显式 contact-mode latent；
5. 静态站立式的 runup 高度/倾斜门对真正跑步过严，但在新的动态支撑指标验证完成前不能直接放宽。

因此“跑—刹—踢一气呵成”的根治点不是把 bridge 再缩短，而是训练一个覆盖最后两步的统一 transition actor。

## 五-A、C1 速度匹配课程结果

第一组保持 SONIC、射门先验、目标、seed 和 calibration v3 完全一致，只改变 bridge 的入口/出口速度比例：

| 入口/出口速度比例 | 门线误差 | 下角距离 | 踢球峰值倾斜 | 触顶步 | 峰值需求比 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 0 / 0（旧 bridge） | 0.170 m | 0.370 m | 0.400 rad | 8 | 1.081 | REJECT |
| 0.25 / 0.50 | **0.027 m** | 0.173 m | 0.300 rad | 6 | 1.036 | REJECT |
| 0.45 / 1.00 | 0.037 m | **0.163 m** | 0.287 rad | 6 | 1.042 | REJECT |
| 0.70 / 1.00 | 0.029 m | 0.229 m | **0.273 rad** | 5 | 1.029 | REJECT |

速度匹配不是主观“看起来顺”：相同控制器下，精度、躯干倾斜和 authority 三项同时改善，而且四组均严格双回放、无摔倒、无触球前静止段。

第二组使用更保守的 calibration v4：

| 候选 | 门线误差 | 下角距离 | 踢球峰值倾斜 | 触顶步 | 结论 |
|---|---:|---:|---:|---:|---|
| phase 150 / entry 0.45 | **0.039 m** | **0.161 m** | 0.296 rad | 2 | 当前均衡 C1，REJECT |
| phase 150 / entry 0.70 | 未过门 | 无 | 0.283 rad | 1 | 接触分支退化，REJECT |
| phase 170 / entry 0.45 | 0.370 m | 0.170 m | 0.300 rad | **0** | 横向过冲，REJECT |

phase 170 把 handoff-to-contact 从约 2.77 s 缩短到 2.13 s，并首次清零触顶，但 aim 0.15～0.50 课程出现偏内、不过门、触顶激增和横向过冲等非单调分支。它适合作为 contact-mode classifier / MoE 的训练集，不适合直接晋升。

逐帧复查 C1 视频后，助跑末步和桥接的动力学明显改善，但 phase 150 射门先验本身仍含短暂直立蓄力。因此 C1 解决了“速度被硬清零”，尚未解决“两个专家的动作语义不一致”；下一层必须学习接入相位和 support-foot timing，而不是继续手工缩短 bridge。

## 六、下一轮实施顺序

### P8-A：C1 transition actor，而不是继续放大 torque IQL

先输出低维、可解释动作：

- 相位速率；
- 最后两步步幅；
- COM 横移；
- 骨盆/脚 yaw；
- 摆臂幅度；
- support-foot timing。

冻结 SONIC 与射门专家，只学习它们之间的条件混合。C1 通过后再扩大到关节目标残差，最后才考虑更大的力矩残差。

### P8-B：contact-mode classifier + MoE

将 episode-06 的失触、episode-04/07 的弱接触、episode-03/05 的精准接触组成四类：

- instep strike；
- inside-foot strike；
- glancing contact；
- no contact。

critic 不只回归落点，还要预测模式概率、margin 和失败成本；transition actor 按模式专家选择动作，减少参数附近的非连续跳变。

### P8-C：动态跑动稳定门

在保留跌倒、关节、力矩、最终稳定硬门的前提下，新增：

- 支撑多边形 / capture point margin；
- 触地冲量与滑移；
- flight phase 合法性；
- pelvis vertical oscillation；
- 刹停距离与动量卸载；
- 触球前 support-foot timing。

只有这些指标验证后，才能讨论用“跑步门”替代目前偏静态的 0.70 m / 0.30 rad 门，而不是为了让结果通过而降低标准。

### P8-D：持续学习与 stability-plasticity

- SONIC body foundation 与现有通过基线冻结；
- 只更新 transition actor / contact critic；
- 每次更新回放基线保留集、失触集、跌倒集和不同 seed；
- candidate 必须有至少 5% 可测 learned effect；
- 任何 authority、跌倒、精度或恢复退化立即回滚；
- 保留回合指标在训练期间不可访问。

## 七、证据与产物

- 课程：`/code/rosclaw/phase8_evidence/g1-sonic-curriculum-v1/`
- 训练数据：`/code/rosclaw/phase8_evidence/g1-approach-strike-dataset-v2/`
- IQL 候选：`/code/rosclaw/phase8_evidence/g1-approach-strike-iql-v1/`
- IQL 物理回放：`/code/rosclaw/phase8_evidence/g1-sonic-iql-residual-v1/`
- IQL 自动拒绝：`/code/rosclaw/phase8_evidence/g1-sonic-iql-residual-v1/evaluation.json`
- authority calibration v3：`/code/rosclaw/phase8_evidence/g1-sonic-authority-calibration-v3.json`
- 当前均衡候选：`/code/rosclaw/phase8_evidence/g1-sonic-authority-aim050-v3/`
- 22.77 s 拒绝候选诊断视频：`/code/rosclaw/phase8_evidence/g1-sonic-authority-aim050-v3/g1-sonic-authority-aim050-v3-diagnostic.mp4`
- C1 速度课程：`/code/rosclaw/phase8_evidence/g1-sonic-transition-c1-v1-*` 至 `g1-sonic-transition-c1-v4-*`
- C1 当前均衡候选：`/code/rosclaw/phase8_evidence/g1-sonic-transition-c1-v2-entry045/`
- C1 Growth triage：`/code/rosclaw/phase8_evidence/g1-sonic-transition-c1-v2-entry045/growth-triage.json`
- C1 22.77 s 诊断视频：`/code/rosclaw/phase8_evidence/g1-sonic-transition-c1-v2-entry045/g1-sonic-transition-c1-v2-entry045-diagnostic.mp4`

逐帧复查与数值结论一致：助跑阶段已有摆臂、腾空和前倾；速度匹配后触球倾斜和落点显著改善，踢后能快速稳定，不再连续后退或抖动；但 phase 150 接入仍会短暂直立，随后才重新进入支撑脚和摆腿姿态。这一可见停顿正是 SONIC 与射门专家动作语义之间的剩余架构边界。视频因此保留 `REJECTED SONIC CANDIDATE / DIAGNOSTIC ONLY / NOT PROMOTED` 水印，不作为宣传通过样例。

全部产物为 `SIM_ONLY`，没有连接真实机器人，没有发送硬件命令，也没有停止用户现有 GPU 任务。
