# ROSClaw Phase 8：MotionDecode 小脑先验与恢复在线学习闭环

## 结论先行

本轮不是只把 MotionDecode “接进来”，而是完成了一个可审计、可拒绝、可回滚的闭环：

```text
外部动作数据
  → 来源/许可/语义审计
  → G1 运动学清洗与 Body 绑定
  → 4×A6000 自监督本体先验
  → 恢复段行为克隆
  → MuJoCo 密集恢复 replay
  → 受约束在线 actor-critic
  → 物理信赖域搜索
  → 独立 Validation 门
  → REJECTED（不晋升）
```

确实训练出了模型，也确实更新了直接输出 29 维关节力矩的 actor。MotionDecode 稳定性先验在 source-episode-disjoint 留出集上将 next-proprioception Smooth-L1 从 `0.01797995` 降到 `0.01666687`，改善 `7.30298%`；4 张 A6000 的独立 seed 全部超过 `2%` 表征门。恢复段行为克隆的验证损失进一步降到 `0.00326415`。

但是，最终在线候选在 4 个独立 Validation 场景中与父策略的成功率、临界失败率、恢复分和支撑脚滑移完全相同，没有形成可测的物理可塑性。因此最终决定是 `REJECTED`，没有晋升、没有硬件授权，也没有把“loss 下降”包装成“G1 已经更稳”。

这次突破主要在 ROSClaw 的自进化基础设施：系统已经会摄取外部经验、训练表征、迁移到小脑、在线更新、验证并在收益不足时自我否决。尚未突破的是“数据表征收益稳定转化为可见踢后恢复收益”。

## 1. 下载数据的真实状态

本地 MotionDecode 快照包含：

| 项目 | 本地事实 |
|---|---:|
| CSV | 30,387 个 |
| CSV 字节数 | 76,903,739,232 |
| 本地主类 | 6 个（`1.1`、`1.10`、`1.11`、`1.12`、`1.13`、`1.14`） |
| 元数据声明主类 | 23 个 |
| 本地足球/球类 CSV | 0 |
| 同步球体位姿 | 0 |
| 视频/逐 episode 语义标签 | 0 |

本地内容量很大，但不是上游元数据声明的完整 23 类镜像。尤其缺少 `3.3 Ball_Game_Interaction`，所以它不能被称为“足球数据集”。本轮只把它用作人体/G1 运动学先验，不用它学习踢球接触、奖励或力矩。

每个 CSV 的 36 列是根位置 3、根四元数 4 和 G1 关节角 29；时间仅由 `120 Hz` 隐含。没有关节力矩、控制 action、接触、球位姿或 reward。代码把这些缺失语义写入 `ExperienceCapsule`，并将能力上限硬限制为 `MOTION_PRIOR_ONLY`。

许可检查采用非空的 `LICENSE.md`，而不是空的 `LICENSE`。当前用途允许研究/个人学习/非商业原型，但商业使用需要书面许可，且不允许再分发原始数据。本 PR 不提交原始 CSV、训练 pack 或派生权重。

## 2. 数据质量闭环

### 2.1 来源与版本

`rosclaw collective source inspect` 会生成：

- 上游项目、40 位 revision、许可哈希和本地 inventory 哈希；
- 本地/索引分类差异；
- 足球、对象位姿、视频和语义标签是否存在；
- `VERIFIED` 或 `UNVERIFIED_LOCAL_SNAPSHOT` revision binding；
- 不可用于晋升证据、不可授权硬件的 Experience Capsule。

本地快照没有能把每个 payload 密码学绑定到上游 revision 的元数据，所以状态保持为 `UNVERIFIED_LOCAL_SNAPSHOT`。代码不会仅因为调用者传入一个 commit 字符串就声称已经验证。

### 2.2 400 条确定性 pilot

pilot 原计划在足球、平衡、步态、过渡/恢复四个 strata 各抽 100 条。因为本地足球为 0，系统显式记录 `football_shortage=100`，再加入 100 条 coordination supplement；替代数据仍标作 supplement，不冒充足球。

首轮整段严格审计发现少量根瞬移和速度尖峰。第二轮引入连续 clean-span 提取：根瞬移、非有限值、关节限位、异常关节速度附近被切断，只从连续合格区间产生训练窗口。最终 pilot：

- 400/400 条达到 `Q1_KINEMATIC_ONLY`；
- 2,086,231 帧，17,381.925 秒（约 4.83 小时）；
- 加权 clean-frame 比例 `99.6577%`；
- 128,911 个可训练连续窗口；
- 0 条达到 Q2 或更高，因为数据没有 action/contact/reward 语义。

每条 episode 保存内容哈希、有限性、四元数误差、关节边界、速度/加速度、重复帧、clean spans 和抽样 MuJoCo 几何检查；证据目录不复制原始 CSV。

## 3. 训练出的“小脑先验”是什么

### 3.1 输入与目标

稳定性 prior 使用 61 维本体特征：

```text
29 关节角 + 29 关节速度 + 3 维躯干投影重力
```

模型是 96 隐状态的 GRU residual predictor。它查看连续 32 帧，预测下一帧本体状态相对“保持上一帧”的残差。目标不是动作或力矩，而是学习人体/G1 姿态变化、协调和恢复的时序表征。

数据按 source episode 做 80/20 分割，杜绝同一条动作的相邻窗口同时进入训练和验证。最终稳定性 pack 只使用 `balance_proxy` 与 `transition_recovery`：200 个源 episode、6,400 个训练窗口、1,600 个验证窗口。

### 3.2 四卡结果

4 张物理 A6000 分别运行 seed `8600..8603`。四个 worker 的验证改善为 `7.1875%`、`7.3030%`、`7.2909%`、`7.2504%`，均通过至少 `2%` 的表征门。选中的 GPU 1 artifact 哈希为：

```text
sha256:9037a55a2d519c90e1623a1f4a3a147d6055599b59fc633964af09cf4a401abb
```

MotionDecode prediction head 永远不能变成 torque head。迁移时只初始化 direct-torque actor 中语义对应的 GRU 输入列，并把源/目标 normalization 做代数变换；artifact 必须匹配完整 GoalForge G1 Body 哈希。之后仍必须经过教师 BC 和独立物理验证。

## 4. Stability–Plasticity 的工程实现

本轮增加了稳定 actor 与可塑 actor 的上下文组合：

- stable actor 保留已验证父能力；
- plastic actor 接受 MotionDecode prior、恢复段 BC 和在线 actor-critic 更新；
- 只有进入踢后恢复相位、骨盆高度合格、躯干接近直立、至少一脚接触且连续满足 warmup 时，plastic actor 才能接管；
- 任一条件失效立即回退 stable actor；
- 两个 actor 都经过同一独立力矩安全投影器；
- receipt 记录 plastic 激活比例、相位/姿态/接触拒绝和全部 fallback 原因。

这不是在动作脚本上再叠一个参数补丁。最终控制路径仍是 102 维 observation 经 GRU 输出 29 维关节力矩，并在 500 Hz MuJoCo 物理子步中生效；上下文门只决定此刻使用稳定神经 actor 还是在线可塑神经 actor。

## 5. 密集在线恢复学习

在线 replay 与 MuJoCo trace 做精确时间对齐：500 Hz 策略记录与 50 Hz trace 是严格 `10:1`，只截取踢球后的恢复段。每个 transition 计算：

- 躯干 roll/pitch、角速度和骨盆线速度惩罚；
- 支撑脚滑移、COM 支撑裕量、单/双脚接触奖励；
- 是否摔倒、关节/力矩约束与投影 fallback cost；
- 相对父动作的 imitation/anchor 项；
- episode 级恢复分。

持续 learner 包含 twin reward critics、twin fall critics、twin constraint critics、target networks、熵温度、拉格朗日乘子、RECENT/ANCHOR/BOUNDARY replay、parent churn、历史 anchor 和 EWC。失败/临界轨迹可以训练 critic，但不能让 actor 模仿危险动作。

最终 generation 的 replay 有 3,820 条 critic transitions、612 条新鲜 actor transitions、2,596 条 anchors；actor 和三组 critics 都发生了有限更新。随后沿父权重到 proposal 的线段做 MuJoCo 信赖域搜索，Development 选择了 `1%` 更新幅度。

## 6. 为什么最终仍被拒绝

独立 Validation 有 4 个从未参与在线更新和步长选择的场景：

| 指标 | 父策略 | 在线候选 |
|---|---:|---:|
| 成功率 | 25% | 25% |
| 临界失败率 | 75% | 75% |
| 平均恢复分 | -11.7240 | -11.7240 |
| 支撑脚滑移 | 0.04905 m | 0.04905 m |
| 躯干 roll 峰值 | 2.41809 rad | 2.41809 rad |

虽然候选 tensor 与父策略不同、在线 actor 确实更新、Development 信赖域也找到安全步长，但独立物理指标没有任何可测提升。候选因此同时未通过：

```text
validation_gate_passed = false
validation_plasticity_measurable = false
decision = REJECTED
```

这暴露出目前最关键的断点：prior 学会了“身体下一步通常怎么动”，BC 学会了“模仿父控制器后半段”，critic 也收到密集恢复信号，但 1% 的安全更新被父策略回退和投影壳大量吸收；更大的更新又容易导致混合动力学分支改变。现有 4 条在线轨迹也不足以估计高难度恢复的价值面。

## 7. 新增的 ROSClaw 产品能力

### 7.1 Collective Experience Plane

新增 `rosclaw.collective`：

- 外部数据 source descriptor、license result 和 Experience Capsule；
- MotionDecode 精确 36 列 parser 与 120 Hz 导数；
- 确定性分层 pilot、短缺/替代显式化；
- G1 Body/关节顺序绑定；
- clean-span 数据质量审计；
- 无 pickle 的有界 `.npz` pack/artifact；
- source-episode-disjoint split；
- 4 个独立物理 GPU worker 和至少 2% 的留出门。

用户接口：

```bash
rosclaw collective source inspect motiondecode ...
rosclaw collective ingest motiondecode ...
rosclaw collective prior build ...
rosclaw collective prior train ...
```

### 7.2 持续恢复控制面

- MotionDecode GRU 到 direct-torque GRU 的语义受限迁移；
- 只选择 episode 末段的 recovery-focused BC；
- stable/plastic 双 actor 上下文门；
- 500/50 Hz 严格对齐的 dense online recovery replay；
- simulator-side actor interpolation 与完整物理信赖域；
- 独立 Validation 的 measurable-plasticity 硬门；
- 全路径 `SIM_ONLY`、Body/父策略/数据哈希绑定和 fail-closed receipt。

## 8. 数据质量、局限和禁止外推

1. 本地 MotionDecode 仍缺上游 17 个主类，不能说“完整数据集已验证”。
2. 本地没有 football、球位姿、接触、reward 或 torque，不能用此 prior 证明足球技巧提升。
3. 坐标约定未得到上游机器可验证契约，现有模型只作表征初始化。
4. 4 个 Validation 场景对安全回归很敏感，但统计功效仍不足；不能据此估计真实成功率。
5. 候选仍有 75% 临界失败率，绝不能制作成“已成功的小脑”或进入真实机器人。
6. 所有结果来自 MuJoCo；没有 ROS/DDS/Unitree SDK/真实机器人动作。

## 9. 下一轮最有价值的开发

下一轮不应简单增加 BC epoch，而应解决“可塑性穿不过安全壳”的问题：

1. 建立并行 recovery curriculum：按冲量、支撑脚、目标高度、摩擦、质量和传感延迟覆盖更多踢后状态；
2. 把 actor 分成 locomotion/balance foundation trunk 与 football residual head，先保证卸力迈步，再学击球风格；
3. 在安全动作空间训练 residual torque 或 latent skill，逐步放宽接管，而不是一次改变 29 维全身力矩；
4. 用 distributional/ensemble safety critics 给信赖域提供不确定性，而不只看点估计；
5. 对 stable/plastic 的实际施加力矩差异设置最小 effect-size 门，避免“权重变了、运动没变”；
6. 增加 20+ 独立恢复 Validation、历史 retention suite 和跨代 lineage，再允许任何宣传视频使用新候选；
7. 等本地 `3.3` football 数据真正出现后，重新审计其列语义；只有同步球位姿/接触存在时，才进入球技表征训练。

“见天地”对应外部经验和环境分布；“见自己”对应 Body、proprioception、自我模型和可测 effect；“见众生”对应跨机器人/跨任务的经验胶囊与共同验证。Dream/Collective 可以提出候选，但只有清醒世界的独立物理门能够决定它是否成长。

## 10. 证据索引

- `motiondecode-pilot-v3/motiondecode-pilot-report.json`：400 条 pilot 与数据质量；
- `motiondecode-stability-prior-v1/stability-prior-pack.json`：61 维稳定性 pack；
- `motiondecode-stability-prior-v1/four-gpu-residual/four-gpu-report.json`：4×A6000 训练；
- `g1-recovery-online-rl-v2/g1-recovery-online-rl-report.json`：最终在线闭环和拒绝决定；
- 上述目录都位于外部 `phase8_evidence` 根目录，不进入源码提交。

证据中的程序退出成功只代表流水线完成；是否可用必须读取 `decision`、`checks`、`blockers`、`promotion_evidence_eligible` 和 `hardware_authorized`。
