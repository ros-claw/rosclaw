# ROSClaw Phase 8：安全在线力矩小脑迭代报告

日期：2026-08-01
证据域：`SIM_ONLY`
最终结论：`REJECTED`（候选未晋升，原因是未见场景支撑脚滑移回归）

## 1. 本轮到底做了什么

本轮不是继续给 G1 叠加踢后动作补丁，而是把恢复控制推进到真正的数据驱动闭环：

1. 29 关节神经 actor 直接输出力矩；
2. MotionDecode 的 61 维运动表征只初始化 GRU 躯干，不把无动作语义的数据伪装成控制策略；
3. 本体感从 102 维扩展到 108 维，加入基座三轴线速度与三轴角速度；
4. 在 MuJoCo 中施加有时间相关性的 OU 探索噪声；
5. 每代执行“新策略采样 → critic 更新 → actor 更新 → 7 档信赖区间 → 开发集晋升或回滚 → 新策略重新采样”；
6. 旧代 replay 自动老化，只能训练 critic，不能再次驱动 actor；
7. 只有塑性 actor 的力矩真正被安全投影并应用的时刻，才有资格进入 actor replay；
8. 倾斜、基座线/角速度、滑移与质心偏移形成稠密奖励和安全代价；
9. 跌倒轨迹按首次本体感失稳时刻隔离危险前缀，不再错误地按 episode 末尾隔离；
10. 候选必须同时通过分数、成功率、新临界失败、输出覆盖、支撑脚滑移、滚转和俯仰护栏。

所有直接力矩输出仍受独立的幅值、变化率、机械功率、关节限位、父策略偏差与状态保护投影约束。配置硬限制为 `SIM_ONLY`，没有打开 ROS、DDS、厂商 SDK 或真实机器人接口。

## 2. 为什么原来的学习没有救到困难场景

第一版 replay 把一次 episode 的最终 `joint_limit_violation` 当成整条轨迹都不允许 actor 学习。实际剖析发现，部分关节限位来自固定踢球先验的早期动作；在后续恢复阶段，机器人仍直立且塑性 actor 已经安全工作。结果是四个训练场景中只有最简单场景的 303 条 transition 能更新 actor，其余困难数据全部只给 critic。

修正后，资格判断改为逐 transition：

- 当时确实由塑性 actor 贡献最终力矩；
- 姿态、滑移、质心和速度仍在局部安全区；
- 不处于检测到的跌倒临近隔离窗；
- replay 来自当前或上一代策略。

困难但局部安全的状态由此可以教 actor，危险状态仍保留给 reward/fall/constraint critics。

## 3. Stability–Plasticity 如何落实

稳定性一侧：

- teacher anchor replay；
- parent distillation；
- EWC 参数保持；
- 陈旧策略数据禁止进入 actor；
- 每次 actor 更新都从父快照做 0.01/0.02/0.05/0.10/0.20/0.50/1.0 七档插值；
- 开发集不回归才可安装精确量化 artifact；
- 未见验证集另设滑移与姿态护栏。

可塑性一侧：

- 当前策略重新采样；
- 受限 OU 探索；
- twin reward critic、fall critic 与 constraint critic；
- 逐代 fresh actor 更新；
- 基座速度进入本体感与稠密恢复奖励。

这不是“训练一次就叫持续学习”，而是可回滚、可审计、多代重新采样的持续适应协议。

## 4. 五次闭环实验复盘

| 实验 | 关键变化 | 接受代数 | 未见集滚转 parent→candidate (rad) | 未见集滑移 parent→candidate (m) | 结论 |
|---|---|---:|---:|---:|---|
| v1 | 安全 OU 探索、多代框架初版 | 1/2 | 1.5223→1.5181 | 0.02877→0.02877 | 效应量不足，拒绝 |
| v2 | 局部安全困难状态进入 actor、稠密风险 | 3/3 | 1.5258→1.5176 | 0.02877→0.02877 | 效应量不足，拒绝 |
| v3 | phase=0.20 预防性恢复、困难开发集 | 1/2 | 1.5223→1.4194 | 0.02584→0.03833 | 旧门曾通过；复盘发现滑移 +48%，结论作废 |
| v4 | phase=0.05 全面提前接管 | 0/1 | 1.5311→1.5311 | 0.02968→0.02968 | 父临界率恶化至 75%，回滚 |
| v5 | 恢复 phase=0.20、真实贡献 mask、新滑移/姿态护栏 | 1/2 | 1.5223→1.5519 | 0.02584→0.03351 | 滑移 +29.7%，拒绝 |

v3 是一次重要的“假阳性”发现：平均滚转改善 6.76%，但代价是脚底滑动明显增加。若只看上半身姿态，它会被错误宣传成成功；新护栏使这种指标投机不能再晋升。

## 5. 最终 v5 数据

### 5.1 流水线检查

通过：

- dense replay 对齐；
- critic 与 actor 都发生有限更新；
- 安全探索实际执行；
- 困难但局部安全的状态进入学习；
- 每次 actor 更新前 fresh resampling；
- 开发集找到信赖区域；
- 未见集严格确定性回放；
- `SIM_ONLY` 边界保持。

未通过：

- `validation_gate_passed`；
- 具体 blocker：`matched_support_slip_regression`。

### 5.2 未见验证集聚合

| 指标 | 父 MotionDecode+BC | 在线 actor–critic 候选 | 变化 |
|---|---:|---:|---:|
| 成功率 | 25% | 25% | 持平 |
| 临界失败率 | 50% | 50% | 持平 |
| 平均任务分 | -6.8161 | -6.7560 | 小幅改善 |
| 平均滚转峰值 | 1.5223 rad | 1.5519 rad | +1.95%，恶化 |
| 平均支撑脚滑移 | 0.02584 m | 0.03351 m | +29.7%，恶化 |
| 神经输出覆盖 | 29.04% | 29.23% | +0.19 pp |

### 5.3 在线 replay

第一代 fresh actor transition 共 592 条；第二代 656 条。最困难训练 g7 在塑性门打开前已经失稳，因此两代均为 0 条 fresh actor transition、520 条 critic-only transition。这个现象说明下一阶段需要独立的早期平衡 head 或更强的 foundation locomotion policy，不能继续把一个恢复 head 生硬提前。

第一代 actor 原始动作相对硬力矩上限的 RMS 变化为 0.473%，峰值 1.297%；第二代为 0.726% / 1.992%。所有候选仍经过七档信赖区间和独立安全投影。

## 6. 视频怎么看

视频是 39.13 秒、1280×720、30 fps 的四挑战 matched split-screen：左侧是父 MotionDecode+BC，右侧是 v5 在线 actor–critic。四个场景都各自做了父/候选严格确定性双重回放。

- g0：两侧都成功；候选俯仰稍好，但滚转和滑移变差；
- g2：候选滚转 0.355→0.335 rad、滑移 0.0135→0.0077 m，但仍然射偏；
- g4：两侧完全相同并跌倒，说明塑性策略没有及时介入；
- g7：候选滚转仅轻微下降，滑移 0.0515→0.0867 m，仍跌倒，因此拒绝。

视频不是宣传剪辑，而是把“哪里变好、哪里没救到、为什么不能晋升”同时可视化。

## 7. 本轮代码产出

- 安全、可复现、上下文门控的 OU 直接力矩探索；
- 108 维本体感（新增基座线/角速度）；
- 多代 actor–critic、fresh resampling、旧 replay 老化；
- 精确量化 artifact 再安装；
- 局部安全困难状态 replay 与真实失稳时刻隔离；
- 稠密恢复 reward/fall/constraint costs；
- 滑移与姿态回归门；
- 可选的分层预防 residual composer（默认关闭，已做 2%/5%/10% smoke）；
- 证据绑定的神经力矩对照视频导出器和 manifest。

## 8. 下一阶段建议

1. 将 foundation balance/locomotion actor 与 kick/recovery residual head 分开训练；
2. 对 phase<0.20 的平衡 head 使用独立 replay 和动作幅度预算，不复用恢复 actor；
3. 在奖励中加入逐时刻足底切向速度、接触冲量与 tail jerk，而不只使用 episode 峰值；
4. 扩大训练域中的校准偏差、延迟和摩擦随机化，并保留独立困难开发集；
5. 引入 risk-conditioned critic ensemble，遇到 epistemic uncertainty 时自动缩小 residual；
6. 继续保持未见集一次性使用、失败即回滚和 `SIM_ONLY`。

## 9. 证据位置

- 最终报告：`/code/rosclaw/phase8_evidence/g1-anticipatory-safe-exploration-v5-final/g1-recovery-online-rl-report.json`
- 最终视频：`/code/rosclaw/phase8_evidence/g1-neural-torque-v5-video/g1-neural-torque-online-comparison.mp4`
- 视频 manifest：`/code/rosclaw/phase8_evidence/g1-neural-torque-v5-video/g1-neural-torque-online-comparison.json`
- 视频轨迹：`/code/rosclaw/phase8_evidence/g1-neural-torque-v5-video/g1-neural-torque-online-comparison-evidence/`

证据中的程序退出成功只代表流水线执行完成；能否晋升必须读取 `decision`、`checks` 和 `blockers`。本轮最终 `decision=REJECTED`，没有真实机器人授权。
