# ROSClaw Phase 8：完整足球数据成长闭环实施报告

## 结论

本轮把最新 `origin/main` 合并到 Phase 8 开发分支，并第一次跑通如下可审计链路：

```text
MotionDecode 足球轨迹
→ 许可/内容寻址注册
→ G1 运动学审计与 repair-manifest 重放
→ source-episode 隔离的本体运动先验
→ 多 GPU 独立种子训练
→ 语义受限地初始化 29 维直接力矩 actor
→ 先验比例与 actor 信赖域搜索
→ Development 选择
→ 7 个 Validation/Holdout 严格物理回放
→ 绝对安全门拒绝
→ 失败证据进入 agentd 负记忆
```

MotionDecode 已经带来可测的运动表征和部分物理改善，但尚未让困难场景达到可接受成功率。因此最终状态仍是 `REJECTED`，没有晋升、激活或硬件授权。真正的进展是：ROSClaw 不再把 loss 下降、相对改善或视频观感自动等同于能力突破。

## 代码同步和边界

- 开发分支：`agent/phase8-context-dream`。
- 同步基线：`origin/main@fbf9a69`，合并提交 `135c44d`。
- 所有新增控制/训练能力均为 `SIM_ONLY`。
- 没有启动 ROS DDS、Unitree SDK、电机命令或真实机器人动作。
- GPU 1 上约 28 GB 的外部任务全程未被中断；开发训练只使用 GPU 0/2/3，物理迁移训练使用 GPU 2。

## 1. 数据事实与 Dataset Doctor

数据根位于 `/code/rosclaw/rosclaw_football/datasets`，检查时仍有后台同步任务，因此所有 inventory 都明确保持 `transfer_active=true` 和 `training_eligible=false`。

Dataset Doctor 修复了一个误报：旧实现把路径中任意 `pass` 都算足球，且把 Hugging Face `.cache/*.metadata` 镜像也算独立资产。现在只有明确的 `football/soccer/futsal/goalkeeper`，或相邻的 `kick/shoot/pass + ball` 上下文才命中，并排除缓存元数据。

修复后的快照事实：

- MotionDecode：1,204 个真实足球资产，其中 1,203 个 CSV；
- OmniContact：当前只有 1 个实际 soccer object，206 个 soccer NPZ 仍只是缓存元数据；
- GEAR-SONIC、MOSAIC、MotionDecode 和 OmniContact 都因传输仍在进行而禁止训练资格声明。

## 2. MotionDecode 足球轨迹审计

足球 CSV 是 120 Hz、36 列：root pose 7 维加 G1 29 关节角。它不含球位姿、接触、动作、奖励或力矩，因此能力上限只能是运动学/本体先验，不能直接证明足球策略或在线 RL 能力。

本轮确定性注册了 400 条足球轨迹：

- 原始 Q1：47；
- repair-manifest 后 Q1：328；
- 拒绝：25；
- 最终可用于表征的源 episode：375。

严格动力学 qualification 实际执行 238 条候选和 1,235,176 个 MuJoCo CPU physics steps，结果 Q3=0。全部执行轨迹都出现 pelvis fall、root orientation 或非足部触地等问题。这说明这些 retargeted 轨迹不能用简单 PD/位置跟踪直接当成可执行策略。

## 3. 数据血缘缺口及修复

审计发现旧 prior build 虽然采用了修复轨迹的 clean spans，却在生成窗口时再次读取原始 CSV；报告显示“328 条 repaired Q1”，张量却没有使用修复后的时序。

现在 `build_motion_prior_pack` 会：

1. 校验 ingest/repair 报告和源文件内容哈希；
2. 从不可变输入确定性重放 `repair_motiondecode_snapshot`；
3. 对每个 `REPAIRED_Q1` 调用 `replay_segmentation_repair`；
4. 用 derivation-manifest hash 决定 episode split 和窗口抽样；
5. 在最终张量生成时再次重放同一 manifest；
6. 把 repair report、repaired episode 数和 selection commitment 写入 pack。

回归测试规定：repaired Q1 若退回原始 CSV 路径必须失败。

另一个修复是分离两个 Body 身份：

- `body_hash`：经过 `qualify_g1_assets` 的 GoalForge actor；
- `kinematic_body_hash`：用于数据审计的 e-urdf/MJCF 模型；
- transfer contract：只允许精确 29 关节特征语义迁移，不声称动力学可迁移。

## 4. 修正后的足球本体先验

v3 pack：

- pack hash：`sha256:293b88d7eabf5a0cd9a32532e556bb7d88e5421e43e7284a470566e60493b12d`；
- 61 维特征：29 关节角、29 关节速度、3 维 projected gravity；
- 9,600 个训练窗口、2,400 个验证窗口；
- 训练/验证按源 episode 隔离；
- 375 个源 episode，其中 328 个来自 repair-manifest 重放，25 个拒绝。

GPU 0/2/3 的独立 seed 结果：

| GPU / seed | validation loss | persistence baseline | 改善 |
|---|---:|---:|---:|
| 0 / 8800 | 0.01822500 | 0.02131246 | 14.4867% |
| 2 / 8802 | 0.01821364 | 0.02131246 | 14.5400% |
| 3 / 8803 | 0.01824005 | 0.02131246 | 14.4160% |

这是三卡 development 复现，不冒充正式四卡门。最佳 artifact 为 GPU 2 / seed 8802。

## 5. 从表征到直接关节力矩

运动先验本身不输出 torque。迁移只初始化直接力矩 actor 中语义对齐的 GRU 部分，之后重新做 torque teacher BC。正式 CLI 会搜索 5 个先验初始化比例和 7 个后 BC actor 信赖域，并在独立物理场景验证：

```bash
rosclaw simforge validate g1-goalforge \
  --profile motion-prior-transfer \
  --motion-prior <artifact.json> \
  --device cuda:2 \
  --output <external-evidence-dir>
```

迁移门新增两个绝对条件：

- 独立物理验证严重失败必须为 0；
- 独立物理成功率必须至少 50%。

因此“候选与一个很差的基线同样会摔倒”不再能仅凭无新增回归得到 `TRANSFER_CANDIDATE`。

## 6. v3 物理结果

修正数据血缘后的最佳初始化比例为 100%，Development 选择的最终 actor 信赖域为 10%。在 7 个独立 Validation/Holdout 场景中：

| 指标 | 无先验 torque BC | MotionDecode v3 transfer |
|---|---:|---:|
| 平均得分 | -5.1592 | -3.5036 |
| 严重失败率 | 42.86% | 42.86% |
| 成功率 | 28.57% | 28.57% |
| COM 最差裕量均值 | -0.2045 m | -0.0560 m |
| 支撑脚滑移均值 | 0.03191 m | 0.02050 m |
| learned output fraction | 14.49% | 18.20% |

动作先验明显改善了平均得分、COM 和滑移，但没有消除 3 个严重失败，也没有提高成功率，所以结果为 `REJECTED`。失败集中在 generation 4/7/9：高 restitution、低支撑摩擦、控制延迟、较大外扰、关节零偏或移动球等组合。

## 7. 教师污染修复

进一步复盘发现 4 条 torque BC 教师轨迹中，`torque-train-03-g7` 本身就是 `JOINT_LIMIT_EXCEEDED`、fall。旧实现会把这条失败轨迹当作正确 torque 监督。

新实现给每条 teacher rollout 生成审计记录。非有限、跌倒、关节越界或力矩越界的轨迹：

- 不进入 behavior cloning；
- 保留为恢复 curriculum / critic 的负经验；
- 在 transfer report 中列出 scenario、status 和 rejection reasons。

这将 Stability–Plasticity 落到了数据入口：稳定 actor 不模仿危险行为，可塑 critic 仍可从失败中学习。

安全教师 v4 A/B 同时证明“数据卫生正确”不等于“能力自动提升”：去掉 g7 危险教师后，Development 选择了 50% 先验初始化和 100% actor trust；独立集滑移由 2.74 cm 降到 1.29 cm、COM 最差裕量均值由 -13.63 cm 改善到 -4.82 cm，但严重失败从 3/7 增加为 4/7，平均得分从 -5.0491 退化到 -5.6978。`no_new_critical_failure`、零严重失败和最低成功率三道门均拒绝该候选。

这说明被过滤的 g7 教师虽然危险，却提供了困难状态覆盖。正确方案不是恢复模仿它，而是把它放进带成本的 critic/recovery curriculum；只用剩余安全 BC 会扩大分布盲区。

## 8. IQL residual 与 agentd 成长记忆

离线 IQL actor 不再直接接管 29 维力矩，而是作为结构化恢复控制器周围的小残差：

- 默认只作用于 12 个腿部关节；
- 有界为 `maximum_residual_nm × residual_fraction`；
- 使用冻结标准化包络做 support heuristic，超包络立即输出零残差；
- 记录参与率、fallback、置信度、峰值残差和 support RMS；
- 每个候选都要做 8 场景父/候选严格重复回放。

首轮 IQL residual 在 Development 的平均关节 jerk 改善 4.72%，但 8 个 case 仅 7 个通过；center-high-grip 的 jerk 回归 12.18%，且支撑脚滑移 5.56 cm 超过绝对门，因此 `REJECTED_BY_SIM_GATE`。

agentd bridge 会把通过的 measured SIM evidence 作为不可部署 HOW candidate，把失败证据作为负 MEMORY。两者都禁止 promotion/activation/hardware，并对内容哈希保持幂等。桥接器还会拒绝超大或含 NaN 的 JSON、非 SIM/SIM_ONLY 证据、硬件授权字段，以及 `passed` 与 committed gates 不一致的报告。

## 9. 当前瓶颈与下一步

MotionDecode 解决的是“身体通常怎样连续运动”，不是“在高 restitution、低摩擦和 63 N 外扰下应该输出什么 torque”。当前三个失败场景中 learned actor 大量回退，父策略自身也会跌倒；继续增加先验权重只会扩大风险。

下一阶段应按以下顺序推进：

1. 用 generation 4/7/9 失败簇构建 development-only recovery curriculum，holdout 只拒绝、不反向调参；
2. 安全 teacher 只用于 BC，失败 rollout 用于 distributional safety critic 和 recovery actor-critic；
3. 学习 unload step、换支撑脚和躯干/手臂反向摆动的 latent recovery skill，再由有界 residual torque 落到关节；
4. 使用 ensemble critic/UCB 选择在线更新信赖域，保留 anchor replay、parent distillation 与 EWC；
5. 达到 20+ 独立场景零严重失败、成功率和自然度 effect-size 门后，才生成宣传视频候选；
6. 数据传输完成后重新冻结完整 inventory 和 revision，不沿用 transfer snapshot 的资格结论。

## 10. 证据索引

- `/code/rosclaw/phase8_evidence/motiondecode-football-pilot-v4-transfer-snapshot`
- `/code/rosclaw/phase8_evidence/motiondecode-football-prior-v3`
- `/code/rosclaw/phase8_evidence/g1-football-motion-prior-transfer-v3`
- `/code/rosclaw/phase8_evidence/g1-football-motion-prior-transfer-v4-safe-teacher`
- `/code/rosclaw/phase8_growth_evidence/g1-recovery-residual-iql-v1`
- `/code/rosclaw/phase8_growth_evidence/growth-agentd-bridge-v1`

证据目录不进入源码提交。任何结果必须同时查看 `decision`、`blockers`、绝对安全门、`promotion_evidence_eligible` 和 `hardware_authorized`，不能只看训练 loss 或单段视频。
