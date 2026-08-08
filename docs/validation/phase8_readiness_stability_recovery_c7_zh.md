# Phase 8 C7：本体 readiness、拒绝射门与神经小脑恢复闭环

日期：2026-08-07
边界：`SIM_ONLY / CPU_MUJOCO / PRETRAINED_SONIC_NEURAL_POLICY`。没有真实机器人、ROS/DDS 或电机命令；本轮没有声称训练出新的端到端力矩 actor，也没有 Promotion。

## 1. 结论先行

本轮把 C6 暴露的关键断点补成了可执行闭环：

```text
连续 SONIC 助跑
  → 读取真实 handoff 本体状态
  → readiness gate 查询三个击球专家的邻域安全支持
      ├─ 有安全支持：允许受限专家选择（射门能力仍需继续加固）
      └─ 无安全支持：ABSTAIN，不碰球
           → 延续 SONIC 冻结减速尾段
           → 终止站姿参考补齐
           → 神经 decoder 持续读取本体感闭环纠偏
           → 稳定门 + 严格双重回放 + 独立聚合评估
```

冻结 recovery 条件验证 seeds 34/37/38 的结果：

| 指标 | 结果 |
|---|---:|
| 通过 episode | **3/3** |
| strict replay | **3/3** |
| recovery 零 actuator saturation | **3/3** |
| 不跌倒 | **3/3** |
| 平均平移速度 | 0.6055 → **0.0802 m/s**（-86.8%） |
| 平均关节速度 RMS | 1.5372 → **0.0911 rad/s**（-94.1%） |
| 最坏最终平移速度 | **0.1185 m/s**（门限 0.20） |
| 最坏最终关节速度 RMS | **0.1275 rad/s**（门限 0.50） |
| 最坏恢复峰值倾角 | **0.3949 rad**（门限 0.65） |
| 最低恢复骨盆高度 | **0.7135 m**（门限 0.65） |

正式聚合报告：

`/code/rosclaw/phase8_evidence/g1-growth-c7-readiness-recovery-frozen-v2/g1-readiness-recovery-evaluation.json`

- 语义 `report_hash`：`sha256:6a803df8b5ac08597040573dae04a262f6bf9dad08a7e1b2c20e72cf1e29694c`
- 文件 SHA-256：`e5f07bb8c4af5200320ad07cb4a31b3ff9c4d80748ed768ecad85d33519f826f`
- `accepted=true` 只表示 **readiness 已拒绝后的冻结条件恢复**通过；不外推到非 abstain 状态、真实机器人或射门 Promotion。

## 2. 为什么还需要 readiness，而不是再选一个 kick phase

C6 的三专家路由已经证明本体状态可以改善 phase 选择，但 sealed holdout 仍有“所有 phase 都不安全”或“路由选中不安全 phase”的状态。若系统无论如何都必须射门，router 的 fallback 实际上会把“不知道”伪装成“继续执行”。

C7 将“不射门”提升为一等决策：

1. 输入仍来自助跑结束时的真实本体状态，而不是 seed 或结果标签；
2. 对 phase 190/205/214 查询两个最近 development 状态；
3. 只有邻居一致支持该 phase 安全且距离不超过 2.0，phase 才进入 safe-supported set；
4. set 为空则 `abstained=true`，不能把 OOD/无支持状态强行送入击球控制器；
5. readiness 不输出 torque，不具有硬件权限，只决定“可尝试 / 应拒绝”。

六维输入：

- `abs pelvis yaw/roll/pitch`；
- pelvis x/y；
- 29 关节速度 RMS。

artifact：

`/code/rosclaw/phase8_evidence/g1-growth-c7-readiness-gate-v1.json`

- 语义 `gate_hash`：`sha256:efbebbd6fb2b78df70f9264e7a98253bd83f216c527c0cea2eee7a363d9dacb0`
- 文件 SHA-256：`a5b76c8af35a9af4e2b81441b832a31514e6de92f509fe900e1cc7062a76a29e`

## 3. readiness 数据闭环

### 3.1 Development 反事实

seeds 0–31，每个状态分别运行 phase 190/205/214，共 96 条严格回放证据。leave-one-seed-out 结果：

| 指标 | 数值 |
|---|---:|
| 总状态 | 32 |
| 尝试射门 | 25 |
| abstain | 7 |
| 尝试中的不安全 episode | **0** |
| 精度命中 | 11 |
| 平均惩罚误差 | 0.5501 m |
| 三专家全不安全状态 | 2 |
| 全不安全且成功 abstain | **2/2** |

### 3.2 Sealed seeds 32–47

冻结 gate 后执行 16 个新 seed × 3 个 phase，共 48 条严格物理反事实：

| 指标 | 固定 router | readiness gate |
|---|---:|---:|
| 尝试 | 16 | 12 |
| abstain | 0 | 4 |
| 尝试中的不安全 episode | 2 | **0** |
| 精度命中 | 6 | 5 |
| 尝试覆盖率 | 100% | 75% |

正式 evaluator 接受的是“决策安全性”，不是射门技能 Promotion：

`/code/rosclaw/phase8_evidence/g1-growth-c7-readiness-holdout-v1/g1-readiness-gate-evaluation.json`

- `report_hash`：`sha256:3bd960a994857a54706cebddb08f212d9f33687999e3b3fb1a9e861334b51a4c`
- 文件 SHA-256：`1700c1e96901b03267ad33f946084491b0868f86a38db733e2bd5a01a48fef72`

必须保留的限制：这个 sealed set 没有“三专家全不安全”状态，4 次 abstain 事后看都是保守拒绝。因此它证明 unsafe attempt 从 2 降到 0，但也揭示 gate 还存在 coverage/recall 损失；后续要扩大独立 OOD 状态，而不是把 4 次拒绝重新调成尝试。

## 4. 不能把“降低控制增益”当成稳定性突破

从 seeds 32–47 的 48 条轨迹重新按严格稳定合同审计：

- phase 190/205/214 的 hard-safe 数量分别是 15/14/13；
- 同时满足零饱和、run-up tilt ≤ 0.60、kick tilt ≤ 0.40、最终高度/速度的 episode 分别只有 0/2/0；
- 更严格的 run-up tilt ≤ 0.30，三个 phase 都是 0；
- saturation physics steps 分别为 192/228/400。

基于这些真实 torque traces 拟合了新的每关节 authority candidate：

`/code/rosclaw/phase8_evidence/g1-growth-c7-stability-authority-v1.json`

- `calibration_hash`：`sha256:0eac0db9247721d171549811e3e793d55a972e8758b492a8ebcc76979d5dc742`
- 文件 SHA-256：`b29b37b9d8b13306d3094f4537f2fcfe96120461612b90f0f1b06e686b157eed`

在 sealed seeds 48–55、固定 phase 205 上与原 authority 成对对比：

| 指标 | baseline | candidate |
|---|---:|---:|
| hard-safe | 6/8 | **5/8** |
| stability-qualified | 1/8 | **0/8** |
| saturation episodes | 7 | 7 |
| saturation steps | 862 | **1195** |
| precision hits | 1 | **0** |
| 平均 run-up tilt | 0.4161 | **0.4214 rad** |
| 平均 kick tilt | 0.6995 | **0.7109 rad** |

虽然平均惩罚落点误差从 1.3221 降到 1.0547 m，但安全和稳定全面回归，所以候选被拒绝：

```text
HARD_SAFETY_REGRESSION
STABILITY_QUALIFICATION_REGRESSION
SATURATION_NOT_REDUCED
PRECISION_REGRESSION
KICK_TILT_REGRESSION
```

评估报告：

`/code/rosclaw/phase8_evidence/g1-growth-c7-stability-holdout-v1/g1-sonic-authority-evaluation.json`

- `report_hash`：`sha256:95e3c6cbedd6f961562b9815a03a8ec0a1f9dc1d7c79bc033fe03951664be2ad`
- 文件 SHA-256：`e911255d756ba7c245872a7acdf95c814539832ca36fbcc9493b131ed2961c3d`

这是 Stability–Plasticity Dilemma 的一次负样本：数据驱动不等于应该接受；只优化平均误差而牺牲硬安全的 candidate 必须进入失败记忆，不能覆盖稳定策略。

## 5. recovery 的三次迭代：失败不是被删除，而是成为成长数据

### 5.1 v1：关节姿态回默认站姿——跌倒

最初尝试在同一 MuJoCo 世界中用 1.2 秒速度匹配 quintic bridge，把当前 29 关节状态拉回 SONIC default pose，再保持 1.8 秒。seed 33 严格双重回放结果：

- 最低骨盆 0.0775 m；
- 峰值倾角 3.0015 rad；
- 最终骨盆 0.0872 m；
- joint limit violation；
- `passed=false`。

原因：动态行走不是“29 个关节回零”问题。固定姿态过渡破坏了步态相位、落脚时机和质心—支撑面关系。

失败证据保留在：

`/code/rosclaw/phase8_evidence/g1-growth-c7-readiness-recovery-dev-seed33-v1/`

### 5.2 v2：延续神经减速，但冻结最后 target——不倒、仍滑移

第二版延续 SONIC planner 的减速/停止尾段 1.8 秒，再冻结最后一个关节 target 1.2 秒：

- 不跌倒；
- recovery 零饱和；
- 最低骨盆 0.6990 m；
- 峰值倾角 0.6277 rad；
- 但最终速度 **0.9048 m/s**，不合格。

这说明“最后一个神经动作”不是站立控制器。切断本体反馈后，固定 target 仍会让身体漂移。

证据：

`/code/rosclaw/phase8_evidence/g1-growth-c7-readiness-recovery-neural-dev-seed33-v2/`

### 5.3 v3：终止参考 + 神经本体反馈——通过

最终方案：

1. 保持同一个 SONIC encoder/decoder 和 10 帧历史；
2. 使用 planner 已生成的冻结减速尾段；
3. 尾段结束后只重复终止站姿 reference，不冻结 decoder action；
4. decoder 每 20 ms 继续读取真实 pelvis orientation、base velocity、joint position/velocity 和历史 action；
5. 500 Hz PD/torque loop 继续闭环；
6. hold 段 authority scale 0.75，hard torque limit 仍逐步检查；
7. 不 reset qpos/qvel，不瞬移，不触球。

seed 33 从 v1 跌倒、v2 滑移，提升到：

- 最终速度 0.0749 m/s；
- 最终关节速度 RMS 0.0615 rad/s；
- 最低骨盆 0.7503 m；
- 峰值倾角 0.3345 rad；
- recovery saturation 0；
- strict replay；
- `passed=true`。

这不是新训练的 ROSClaw recovery actor，而是把已有的预训练 SONIC 小脑从“助跑片段播放器”正确改造成“减速—站立期间仍闭环感知身体”的连续控制器。证据字段明确记录：

```text
pretrained_neural_recovery_tail = true
rosclaw_trained_recovery_policy = false
promotion_evidence = false
```

## 6. seed 55 压力案例：不是助跑饱和，而是错误地继续踢

原 stability holdout 的 seed 55：

| 路径 | saturation steps | peak demand | fall | goal |
|---|---:|---:|---|---|
| baseline 继续踢 | 759 | 5.777× | 是 | 否 |
| authority candidate 继续踢 | 1101 | 2.852× | 是 | 是，误差 0.989 m |
| readiness abstain + neural recovery | **0（恢复段）** | **0.454×** | **否** | 不尝试 |

按阶段拆账后，readiness recovery 的 pre-abstention saturation 也是 0，说明 759/1101 个饱和步主要由错误击球阶段产生，并非助跑本身。恢复路径最终：

- speed 0.0664 → 0.0147 m/s；
- joint velocity RMS 0.5952 → 0.0431 rad/s；
- peak tilt 0.1399 rad；
- final pelvis 0.7854 m；
- strict replay，`passed=true`。

证据：

`/code/rosclaw/phase8_evidence/g1-growth-c7-readiness-recovery-stress-seed55-v1/`

这个案例是“自我认知”的具体工程含义：系统知道当前身体状态缺少安全击球支持，所以选择不证明勇敢，而是保护身体、恢复可控状态。

## 7. 新增与加固模块

### Growth

- `proprioceptive_readiness_gate.py`：邻域一致安全支持、显式 abstention、hash-bound loader；
- `proprioceptive_readiness_evaluation.py`：sealed 三专家反事实评估；
- `sonic_authority_evaluation.py`：安全优先的成对 authority candidate 验收；
- `readiness_recovery_evaluation.py`：请求/轨迹/实现/Body/router/gate/config 全绑定的冻结条件恢复聚合门；
- CLI：
  - `growth proprioceptive-readiness-gate`；
  - `growth evaluate-proprioceptive-readiness`；
  - `growth evaluate-sonic-authority`；
  - `growth evaluate-readiness-recovery`。

### SimForge / GoalForge

- `G1SonicRunupController.update_recovery_extension()`：射门代码不能隐式越过原 execution boundary，recovery 必须走显式 API；
- `extend_stationary_recovery()`：只补 terminal reference，保持世界、policy history 和本体反馈连续，并更新 reference digest；
- `g1_readiness_recovery.py`：只允许 readiness 已 abstain 的状态进入，严格双重回放；
- `g1_readiness_recovery_video.py`：三 episode 下游可视化，pixels 不参与评分；
- CLI：
  - `goalforge readiness-recovery run`；
  - `goalforge readiness-recovery export`。

所有运行路径保持：

```text
activation_ceiling = SIM_ONLY
hardware_command_sent = false
promotion_truth_allowed = false
activation_authorized = false
hardware_authorized = false
```

## 8. 视频

24.60 秒、3 个完整严格回放 episode：

`/code/rosclaw/phase8_evidence/g1-growth-c7-readiness-recovery-frozen-v2/g1-readiness-recovery-three-seed.mp4`

- 1280×720 / 30 fps / 738 帧；
- seed 34、37、38 各 8.2 秒；
- 蓝点：approach；黄点：neural brake；绿点：neural stable hold；
- 保留助跑、abstain、减速、站稳全过程；
- 像素不参与任务评分；
- 视频 SHA-256：`817ab7ff7a2c1cdfdcc7627b6726864a1af4806da5626dd4ca74dc081e5fd542`；
- manifest SHA-256：`a17b9fe69a222001b5df7e9ddb43e7cd5cebec98437744a46d06d4617da21ef1`。

## 9. 当前“自进化”到了哪一层

本轮已经实现的是：

1. **经验化**：保留所有成功/失败 strict physics traces；
2. **自我表征**：决策输入是本体状态，不是脚本 seed；
3. **能力边界认知**：模型能输出 abstain，而不是永远行动；
4. **候选形成**：authority、router、readiness、recovery 作为独立 candidate；
5. **稳定—可塑性门控**：新候选必须在冻结验证中不回归硬安全；
6. **失败记忆**：v1/v2 和 authority regression 均保留，不覆盖 accepted path；
7. **可审计成长**：每层绑定输入证据、实现、Body、配置和输出摘要。

尚未实现、不能冒充已完成的是：

- ROSClaw 自己训练出的端到端神经 recovery 小脑；
- 在线持续更新 actor-critic；
- 直接输出关节 torque 的已晋升策略；
- 对真实机器人安全有效；
- readiness + attempt + recovery 在一个生产 runtime 命令中的自动分支；
- “更会收住”自动等于“射门更准、更自然”。

## 10. 下一阶段建议

按证据优先级继续：

1. **单入口 readiness-aware strike runtime**：同一个 MuJoCo episode 内自动走 `ATTEMPT` 或 `ABSTAIN→RECOVER`，不依赖操作员选两个 CLI；
2. **扩大独立 abstention 分布**：新 seeds、初始横向速度、yaw/pitch 扰动、球初速和地面摩擦变化；
3. **Recovery replay buffer**：把 v1/v2/v3、成功/失败和 counterfactual 组织为 transition dataset；
4. **蒸馏小脑**：先蒸馏 SONIC neural recovery tail，再训练 bounded residual actor；
5. **在线 actor-critic 只在 shadow SIM 更新**：稳定策略冻结，plastic residual 有 KL/torque/support/OOD 门；
6. **双策略验收**：新 actor 必须同时改善 stop time、backward reversal、tilt、jerk 和 saturation，任何硬安全回归都拒绝；
7. **重新攻射门能力**：readiness 解决“不该踢时别踢”，下一轮仍需降低支持状态中的 run-up tilt、kick tilt 和饱和，并恢复高角精度。

这一轮的突破不是“踢得更猛”，而是 ROSClaw 第一次在连续物理世界中把“我现在不适合踢”转化为可执行、可验证、可恢复的身体行为。
