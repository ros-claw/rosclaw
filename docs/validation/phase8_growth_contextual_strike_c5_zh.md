# Phase 8 C5：本体感上下文射门与 Growth 反事实闭环报告

日期：2026-08-07
边界：`CPU_MUJOCO / DEVELOPMENT_SHOWCASE / SIM_ONLY`，不包含真实机器人结论。

## 1. 本轮结论

本轮实现的突破不是再增加一组固定踢球补丁，而是把“同一个动作不适合所有交接状态”变成了 ROSClaw 可学习、可复验、可拒绝的 Growth 合同：

1. 在助跑结束时读取真实本体偏航，而不是读取 planner seed 或预设身份；
2. 对同一 seed、同一世界、同一控制参数分别执行 phase 214 与 phase 190，形成成对反事实；
3. 仅用 development seeds 拟合一维专家路由器；
4. 在隔离的 holdout seeds 上检查精度、安全和稳定性—可塑性回归；
5. 产出绑定轨迹、Body、实现版本和实验上下文的内容寻址标定文件；
6. 运行时只能加载通过门控的标定，但标定仍永久为 `SIM_ONLY`，不能授权晋升或硬件执行。

最困难的 development seed 1 在固定 phase 214 下门线落点误差为 **1.0142 m**；路由到 phase 190 后误差为 **0.1487 m**，下降 **85.3%**，无跌倒、无关节越界。最初四种子集合由固定专家的 3/4 精度命中提升为 4/4。

扩大到八种子后，结果不是完美泛化：当前两专家路由为 5/8 命中 0.16 m 精度圈。Growth 因而只接受“局部、无 holdout 回归的路由标定”，整个足球候选仍由严格门控拒绝，没有把局部进步伪装成整体通过。

## 2. 新增工程能力

### 2.1 本体感上下文专家

`G1FreeKickFlowConfig` 新增上下文相位配置和标定哈希。物理执行在 SONIC 助跑完成后，从 MuJoCo 实际 pelvis quaternion 计算 handoff yaw，再选择常规 phase 214 或高偏航 phase 190。证据新增：

- `handoff_yaw_rad`；
- `selected_kick_phase_start_frame`；
- `contextual_phase_expert_executed`；
- `contextual_phase_calibration_hash`。

选择发生在速度匹配桥和射门 prior 初始化之前，因此选中的专家完整决定桥接目标、参考锚点和射门时间轴。planner seed 不进入选择函数。

### 2.2 Growth 成对反事实标定器

新增 `rosclaw growth contextual-phase-calibration`。它会：

- 重新计算每个 `.npz` 轨迹哈希，拒绝不绑定或非 strict replay 的 JSON；
- 要求每个 seed 同时存在两个 phase 的反事实 episode；
- 要求 Body hash、implementation hash 和除 seed/phase 外的实验上下文完全一致；
- 仅用 development seeds 拟合 `abs(handoff_yaw) >= threshold` 的单分裂路由；
- 对未过门线 episode 使用明确记录的 2.0 m 惩罚，不丢弃失败样本；
- 在 holdout 上检查平均惩罚误差、精度命中数和安全事件；
- 任何回归都保留 artifact 但返回拒绝码；运行时 loader 只接受门控通过且安全边界字段未被篡改的 artifact。

正式标定文件：

`/code/rosclaw/phase8_evidence/g1-growth-c5-contextual-phase-calibration-v1.json`

标定哈希：

`sha256:aee03355c6864acd786d8f978d38862fff32d94a415c3a905dfc94aa1c794908`

学习到的阈值为 **0.15606 rad**。development 使用 seeds 0–5，holdout 使用 seeds 6–7。

### 2.3 可组合的关节 authority 标定

SONIC authority 标定从仅处理 approach/follow-through 扩展为 approach、strike、follow-through 三段独立标定，并支持以旧标定为 base 继续学习。新标定不允许把已有增益调高，避免“为了可塑性覆盖稳定性记忆”。

本轮保留的 Pareto 标定是 `target_demand_ratio=0.85` 版本；从第二轮 episode 再派生的标定使 seed 3 饱和步数由 39 增至 44，因此被拒绝，未替换保留版本。

### 2.4 SIM-only 2D 足端 teacher

loft teacher 增加：

- 足—球距离门控；
- 前向与竖直速度目标；
- 由足端 Jacobian 转成关节力矩的二维操作空间作用；
- 非有限输入和远距离状态 fail closed。

大范围扫描显示：当前低球 motion prior 的形状限制了上角射门。增大竖直力没有单调增加球高，100 N 以上反而出现关节边界问题；距离门控消除了长期施力，却没有达到 1.35 m 上角目标。本轮据此拒绝“继续堆外力补丁”，后续需要独立高球 motion expert 或由动作数据蒸馏出的高球 prior。

## 3. 八种子成对反事实结果

共同配置：4.4 m 初始球距、SONIC 1.5 m/s 指令、0.16 s 速度匹配桥、shot amplitude 1.0、speed scale 1.05、aim bias 0.8 m、T0.85 authority 标定。

| seed | handoff yaw (rad) | phase 214 误差 (m) | phase 190 误差 (m) | 两专家路由 |
|---:|---:|---:|---:|---|
| 0 | -0.0067 | 0.0768 | 0.0350 | 214 |
| 1 | +0.2367 | 1.0142 | 0.1487 | 190 |
| 2 | -0.0108 | 0.0016 | 0.1928 | 214 |
| 3 | -0.0493 | 0.0996 | 0.1464 | 214 |
| 4 | -0.0754 | 0.0212 | 0.0560 | 214 |
| 5 | -0.0061 | 0.3559 | 0.0349 | 214（尚未识别高速交接） |
| 6 | -0.0721 | 0.1885 | 未过门线 | 214（holdout） |
| 7 | -0.0833 | 0.4176 | 未过门线 | 214（holdout） |

Growth 门控统计：

- development 平均惩罚误差：**0.2615 → 0.1173 m**，下降 **55.2%**；
- development 精度命中：**4/6 → 5/6**；
- holdout 平均惩罚误差：**0.3031 → 0.3031 m**；
- holdout 精度命中：**0/2 → 0/2**；
- 路由选择的 episode：0 跌倒、0 关节越界、0 力矩硬越界。

因此标定的含义是“在已识别的高偏航区间安全改善且不伤害 holdout”，不是“holdout 已经踢好”。

## 4. 第三专家的发现

对两个 holdout 失败状态继续做 phase 200/205/210/218 扫描后，phase 205 显示出新的专门能力：

- seed 6：phase 214 为 0.1885 m，phase 205 为 **0.1523 m**；
- seed 7：phase 214 为 0.4176 m，phase 205 为 **0.0146 m**；
- 两者均无跌倒、无关节越界；
- phase 218 在 seed 7 虽达到 0.1529 m，却发生关节越界，因此明确拒绝。

这说明问题不是“RL 没用”，而是一个 phase 无法覆盖不同的交接极限环。偏航也不是完整自我状态：seed 5 的偏航很小，但交接关节速度 RMS 为约 1.61 rad/s，phase 190/205 明显优于 phase 214。下一轮应把 handoff roll/pitch、pelvis 位置、关节速度 RMS 和支撑相位一起加入三专家路由，并用新的保留 seeds 8+ 做真正的泛化检查；本轮没有把 phase 205 偷塞进已接受 artifact。

## 5. 连贯性与稳定性诚实边界

正向进展：

- 常规专家 handoff-to-contact 为约 0.89–0.94 s，不存在静止等待；
- 高偏航专家约 1.32 s，虽然更慢，但把严重偏航状态救回精度圈；
- 选中的四种子演示均无跌倒和关节越界；
- 相位选择会同时改变速度匹配桥，不是接触前临时切补丁。

仍未解决：

- run-up peak tilt 在八种子中约 0.39–0.60 rad，超过宣传级稳定性目标；
- seed 3 常规专家仍有 39 个 actuator saturation physics steps；
- 两专家只达到 5/8 精度命中；
- 高球上角尚没有合格的动作先验；
- 所有视频仍必须标记 development/rejected candidate，不能称为已晋升技能。

## 6. 可视化证据

运行时通过正式标定 artifact 重放的两个代表 episode：

- 常规专家：`/code/rosclaw/phase8_evidence/g1-growth-c5-calibrated-base/g1-contextual-base.mp4`
- 高偏航恢复专家：`/code/rosclaw/phase8_evidence/g1-growth-c5-calibrated-high-yaw/g1-contextual-high-yaw.mp4`
- 两案例连续版（42.13 s）：`/code/rosclaw/phase8_evidence/g1-growth-c5-contextual-router-two-case.mp4`

连续版 SHA-256：`f0408015ec5368ed2b3f9e836aaf2f31ad6068cf1736ca8dcafac7420a10a4ca`。

视频 overlay 显示实际 handoff yaw、被选 phase、门线误差和标定候选边界；像素不参与评分。

## 7. 下一轮优先级

1. 将 phase 205 纳入三专家成对课程，增加 seeds 8–15 作为全新 holdout；
2. 把 handoff joint velocity RMS、roll/pitch、pelvis xy 和支撑腿相位写入结果和轨迹合同；
3. 学习浅层可审计 router 后，再蒸馏为小型 gating network，保留规则模型作 fail-closed shadow；
4. 为跑步阶段单独训练 balance residual actor-critic，先降低 0.39–0.60 rad peak tilt，再允许射门专家学习；
5. 从 MotionDecode/足球动作数据提取独立高球 motion prior，禁止用长时间外力伪装动作能力；
6. 只有新的 sealed holdout 同时满足精度、跌倒、关节、饱和和倾斜门控，才允许候选进入下一层 Growth，而仍不直接授权硬件。
