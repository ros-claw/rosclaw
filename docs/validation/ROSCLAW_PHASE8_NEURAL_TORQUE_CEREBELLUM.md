# ROSClaw Phase 8：端到端神经力矩小脑与持续 Actor-Critic

## 结论先行

本阶段已经把 G1 踢球从“参数补丁/动作片段选择”推进到一个真实的数据驱动控制闭环：

- GRU actor 从 102 维本体感知与任务上下文直接输出 29 维关节力矩；
- 力矩在 MuJoCo 的 500 Hz 物理子步中生效，不再只是修改目标落点或动作参数；
- 四张物理 A6000 分别训练独立随机种子的行为克隆候选；
- DAgger 用闭环偏移状态重新采集教师纠正；
- 双 reward critic、双 fall critic、双 constraint critic 持续更新；
- actor 只学习新鲜安全轨迹，失败轨迹只训练 critic；
- anchor replay、父策略蒸馏和 EWC 抑制灾难性遗忘；
- 每次 actor 更新必须经过多步长、匹配场景的 MuJoCo 物理回放，失败即恢复完整 checkpoint；
- artifact 绑定 Body、父策略和数据集哈希，并被硬限制为 `SIM_ONLY`。

这证明了“端到端小脑 + 在线强化学习 + 持续 actor-critic + 直接力矩输出”的工程链路能够运行、学习、筛选与回滚。但最新密封验证的最终决定仍为 `REJECTED`：安全与留存门已经通过，泛化增益和直接力矩放行比例还没有达到晋升标准。因此它是可继续训练的研究基础，不是已可部署到真实 G1 的控制器。

## 1. 什么叫端到端小脑

这里的“端到端”是运动控制意义上的端到端，不是从摄像头像素开始：

```text
29 关节位置 + 29 关节速度
          + 躯干投影重力
          + 球相对位置/速度
          + 目标高度/横向位置
          + 动作相位、双脚接触、上一拍力矩
                         │
                         ▼
                   96 维 GRU
                         │
                         ▼
                 29 维关节力矩
                         │
                         ▼
       独立安全投影器 + 父控制器故障回退
                         │
                         ▼
                 MuJoCo 500 Hz 物理环
```

冻结后的 observation 维度是 102，action 维度是 29。GRU 提供短期运动记忆，使策略能够根据“刚才身体怎么动、上一拍用了多少力”决定下一拍力矩，而不是逐帧孤立决策。

独立安全投影器限制力矩幅值、变化率、机械功率、关节边界和相对父控制器的偏差。姿态异常、分布外输入、过度投影、非有限数或 artifact 不匹配都会 fail closed 到已验证父控制器。当前代码没有 ROS、DDS、Unitree SDK 或真实硬件授权路径。

## 2. 数据与学习闭环

### 2.1 教师数据

合格的 RoboNaldo + PD 控制器在每个 2 ms 物理子步提供父力矩。采集器原样透传父力矩，同时记录 observation、实际施加力矩和父力矩。因果透明测试要求加入采集器前后：

- episode summary 完全相同；
- 全轨迹哈希完全相同；
- 共记录 6520 个物理步/episode。

### 2.2 四卡行为克隆

四个隔离 worker 分别绑定 `CUDA_VISIBLE_DEVICES=0..3`，核验 GPU UUID 与 PCI bus id，并训练独立 seed。worker 只输出安全 tensor artifact，不使用 pickle。主流程选择验证损失最低的 seed，再在同一配置上进行 DAgger。

### 2.3 DAgger

只看教师轨迹会产生 covariate shift：开放环 BC loss 很低，机器人一旦产生微小姿态偏差，网络就进入没见过的状态并持续犯错。DAgger 让当前神经 actor 在闭环 MuJoCo 中运行，再对它真正访问到的状态记录父控制器纠正。最新完整实验中，验证 MSE 随代际持续下降；但密封物理指标而不是 MSE 决定是否晋升。

### 2.4 持续 actor-critic

训练器包含：

- twin reward critic：学习任务收益；
- twin fall critic：学习摔倒风险；
- twin constraint critic：学习约束/投影风险；
- target critics、SAC 熵温度和两个拉格朗日乘子；
- RECENT、ANCHOR、BOUNDARY 三种 replay 分区；
- policy lag：actor 只看 lag 不超过 1 的 RECENT 数据，critic 可以看全部数据；
- 父策略蒸馏、历史 anchor rehearsal 和 EWC Fisher 正则。

失败或临界轨迹进入 BOUNDARY：它们用于教 critic 识别危险，但不会让 actor 模仿失败动作。这是 stability-plasticity dilemma 的第一层工程解法。

### 2.5 物理信赖域

一次很小的神经参数变化也可能在混合控制器中触发不同的投影/回退分支。实验观察到约 `1e-12` 的 CUDA 末位权重差异在数秒后可形成不同轨迹。因此增加了两层处理：

1. 导出前对推理 tensor 做 `1e-6` 确定性量化，消除无意义的末位噪声；
2. actor 更新依次尝试 `1.0, 0.5, 0.2, 0.1, 0.05, 0.035, 0.02, 0.01, 0.005` 信赖域比例，每个比例都跑完整匹配物理回放。

任一新临界失败、非有限值、成功率下降、均分下降超过 1% 或小脑放行比例下降超过 5 个百分点都会拒绝该步长。全部步长失败时，actor、critics、optimizers、熵温度、拉格朗日乘子、EWC 状态和 CPU/CUDA RNG 一起恢复。

## 3. 最新完整实验结果

证据目录：`/code/rosclaw/phase8_evidence/g1-neural-torque-pilot-v6`

实验使用 4 个训练场景、2 个 DAgger 场景、4 个密封 Validation 场景和 3 个密封 Holdout 场景。在线安全门扩展到全部 6 个 Development 场景；Validation/Holdout 不参与更新或步长选择。

### 3.1 在线更新局部门

| actor 步长 | 6 场开发结果 | 决定 |
|---|---|---|
| 100% | 产生 1 个新临界回归，均分退化 | 拒绝 |
| 50% | 均分 `0.146 → 0.783`，临界失败率 `33.3% → 16.7%`，小脑放行 `37.4% → 36.4%` | 接受 |

这证明在线 actor 不是只更新了文件或 loss：经过缩步后的新权重真实改变了 MuJoCo 物理结果，并且完整步长被安全门拦截。

### 3.2 密封聚合结果

| 指标 | 合格父控制器 | BC 小脑 | 在线 RL 小脑 |
|---|---:|---:|---:|
| 成功率 | 28.57% | 28.57% | 28.57% |
| 临界失败率 | 57.14% | 42.86% | 42.86% |
| 平均任务分 | -6.459 | -3.105 | -4.897 |
| 平均躯干 roll 峰值 | 1.460 rad | 1.286 rad | 1.521 rad |
| 平均支撑脚滑移 | 0.0591 m | 0.0506 m | 0.0292 m |
| 神经力矩实际放行 | 0% | 28.11% | 29.53% |

在线小脑相对父控制器没有新增临界失败，平均分也守住了 3% 留存门；但相对 BC 小脑平均分下降，未获得要求的至少 5% 在线可塑性增益。直接力矩实际放行比例也低于 75% 门。因此最终为：

```text
decision = REJECTED
blockers = online_plasticity_gain_at_least_5pct,
           learned_output_fraction_at_least_75pct
```

所有密封 episode 均通过严格重放，所有候选均为 `SIM_ONLY`，没有真实机器人证据，也没有硬件授权。

## 4. 已验证的 ROSClaw 能力

- MuJoCo G1 Body/父策略哈希绑定；
- 500 Hz direct torque callback 确实进入物理环；
- 教师采集不扰动原轨迹；
- 四张不同 UUID 的 A6000 均被实际使用；
- 安全 tensor artifact 的确定性序列化、大小限制、shape 校验和篡改拒绝；
- checkpoint 使用 `torch.load(..., weights_only=True)`；
- 训练服务完整恢复 optimizer、critic、target、EWC 与 RNG；
- 分区 replay 与 policy lag 防止 stale/boundary 数据污染 actor；
- DAgger、在线 SAC、约束 critic、信赖域物理回放和逐步回滚；
- 密封 Validation/Holdout 严格重放和 fail-closed 晋升。

## 5. 尚未解决的问题

1. 当前神经 actor 主要是在蒸馏父控制器，其能力上限仍受父数据覆盖限制。
2. 安全壳频繁回退，说明网络的闭环状态覆盖不足；29.53% 放行率不应被描述为“已接管全身控制”。
3. 在线 replay 规模仍小，critic 对高难度 g4/g7/g9 的价值估计不够可靠。
4. 在线局部门提升没有泛化成密封收益；需要更丰富的 development curriculum 和 domain randomization，而不是使用密封集调参。
5. 当前是 MuJoCo 中的力矩控制研究候选，不是 Isaac Lab 大规模训练完成的 locomotion foundation policy，也没有 sim-to-real 结论。
6. 本阶段没有制作宣传视频。被拒绝的候选不应作为最终成果视频；后续应在候选通过密封门后再录制长时、多难度对比。

## 6. 下一阶段建议

优先顺序如下：

1. 在 Isaac Lab/MJWarp 扩展到数千并行环境，训练站立、迈步、击球、随动卸力和恢复的分层课程；
2. 给 critic 增加更密集的 COM、角动量、足底接触、关节裕量和动作平滑奖励，同时保留任务落点收益；
3. 使用受安全壳限制的 stochastic rollout 收集真正有动作多样性的在线数据，而不只在已有 BC 轨迹上做 critic 拟合；
4. 引入独立 locomotion/balance foundation policy，踢球 actor 只学习残差或技能条件，再逐步提高直接力矩接管比例；
5. 建立固定历史技能库和跨任务 retention suite，防止“足球进步、站立退化”；
6. 只有连续多个密封代际通过安全、留存、可塑性和直接控制比例门后，才讨论更接近真实机器人的 HIL 或受控硬件验证。

## 7. 复现实验

```bash
PYTHONPATH=src .venv/bin/python -m rosclaw.entrypoint \
  simforge validate g1-goalforge \
  --profile neural-torque-pilot \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --gpu-epochs 12 \
  --dagger-generations 3 \
  --online-updates 1 \
  --output /code/rosclaw/phase8_evidence/g1-neural-torque-pilot-v6
```

退出码表示流水线是否完成，不表示候选是否晋升。候选是否可用必须读取报告中的 `decision`、`blockers` 和 `promotion_checks`。
