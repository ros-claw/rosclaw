# Phase 8：状态路由的情景式触球肌肉记忆

日期：2026-08-11

边界：`SIM_ONLY`。物理真值来自 CPU MuJoCo 连续世界；4 张 A6000 用于并行实验隔离，但没有连接真实机器人，也没有发送 ROS、DDS、串口、CAN 或厂商 SDK 命令。

## 1. 结论

本轮把单一 seed 的目标条件化 actor 扩展成按本体状态路由的局部接触动力学记忆，并完成了训练、冻结、回放、拒绝和精度消融闭环。

- 最终训练集：4 个 planner seed、24 条严格回放；每个情境包含 1 条零力基线和 5 条二维教师力探针。
- 记忆保留 seed `0/4/7` 的 3 个状态原型，seed `21` 因连续性/安全门失败进入拒绝记忆；seed 只作审计标签，不参与运行时路由。
- 18 条硬安全回放进入局部系统辨识，6 条失败回放继续绑定在 artifact 中，不能被选择性删除。
- 最近两个原型的归一化 RMS 距离为 `0.769622`，大于 `0.50` 支持半径，状态岛不重叠。
- held-out 目标 `(8.5, 2.139, 0.765) m` 上，seed 0 的零力误差为 `0.0192517 m`，记忆误差为 `0.0093144 m`；绝对改善 `0.0099373 m`，相对改善 `51.62%`。
- 记忆实际激活 1 个控制帧，选择 seed 0 状态岛，输出横向 `30 N`、竖直 `0 N`，经当前 Jacobian 解码的峰值关节力矩为 `13.8001 Nm`。
- 候选无倒地、无后退、无关节/力矩越界、无执行器饱和，接触任务权威最小值为 `1.0`。
- seed 4、7 的状态能正确识别，但当前目标所需出球速度超出各自局部包络，因此保持零输出；seed 21 为未支持身体状态，同样零输出。

Growth 评估器返回 `DEVELOPMENT`，同时记录 `active_context_coverage_complete=false`。这证明了局部精度改善和跨状态 fail-closed 路由，不是多 seed 泛化成功，更不是硬件 promotion。

## 2. 为什么第一轮被否决

最初版本从“教师首次激活的前一个 20 ms 控制帧”读取状态。逐子步复查发现，足端进入 `0.18 m` 接触门通常发生在下一控制帧内部；前一帧和真正决策子步的足速差异可超过数米每秒。第一版记忆因此把四个运行情境全部判为 OOD，虽安全归零，但没有学会执行。

最终版本在教师力矩尚未施加的那个 2 ms 物理子步记录 11 维因果观测：

```text
[足踝-球相对位置 xyz,
 足踝线速度 xyz,
 骨盆线速度 xyz,
 躯干 roll/pitch]
```

训练端和部署端调用同一个观测函数。这样既消除控制帧错位，也避免使用教师力矩已经改变后的状态造成事后信息泄漏。第一轮失败证据被保留在外部 evidence 目录，没有混入最终 artifact。

## 3. 架构

每个合格状态岛保存一个局部模型：

```text
[1, 教师横向力 Fy, 教师竖直力 Fz]
  -> [实测出球横向速度 vy, 实测出球竖直速度 vz]
```

模型使用带 ridge 的前向系统辨识，再对局部斜率矩阵做正则化伪逆。运行时流程是：

1. 读取当前 11 维本体/球状态；
2. 用固定尺度计算到各原型的归一化 RMS 距离；
3. 超过支持半径时零输出；
4. 根据目标点计算所需出球速度；
5. 超出该原型实测出球包络时零输出；
6. 在实测安全力范围内求局部逆解；
7. 用 MuJoCo 足端 Jacobian 转为 29 维关节力矩；
8. 继续通过接触任务权威投影、总力矩保护和关节边界保护。

planner seed 不进入第 1–7 步，只用于证明路由结果是否与采样情境一致。记忆 artifact 固定为 `SIM_ONLY`，且 `promotion_authorized=false`、`hardware_authorized=false`、`online_hot_swap_allowed=false`。

## 4. Stability–Plasticity 闭环

稳定性来自四层约束：

- 状态 OOD：未知身体状态不输出；
- 目标 OOD：已知身体状态遇到未学习出球需求也不输出；
- 动作 OOD：逆解只能落在该局部岛实际通过安全门的任务力范围内；
- 证据绑定：Body、实验上下文、源码实现、每条 JSON 与轨迹内容哈希必须一致。

可塑性体现在 seed 0 的同一冻结记忆可根据 held-out 目标重新计算横向任务力，并将误差降低 `51.62%`。稳定锚点使用 seed 7：状态路由正确，但目标超出包络，2 个 OOD 帧保持零力，原有稳定性不变。

## 5. 数值结果

| 回放 | 路由 | 激活帧 | 目标误差 m | 后退 m | 倒地/饱和 | 结论 |
|---|---:|---:|---:|---:|---|---|
| seed 0 held-out 零力基线 | 无 | 0 | 0.019252 | 0 | 否/否 | 基线 |
| seed 0 情景记忆 | 0 | 1 | **0.009314** | 0 | 否/否 | 局部改善 |
| seed 4 原目标 | 4 | 0 | 2.182416 | 0 | 否/否 | 目标 OOD，零输出 |
| seed 7 稳定锚点 | 7 | 0 | 3.624855 | 0 | 否/否 | 目标 OOD，零输出 |
| seed 21 拒绝情境 | 最近为 7 | 0 | 0.599579 | 0 | 否/否 | 状态 OOD，零输出 |

seed 0 的触球后峰值骨盆速度从 `1.10162` 轻微变为 `1.10283 m/s`，峰值关节速度 RMS 从 `6.37397` 降到 `6.37103 rad/s`，稳定时间同为 `3.234 s`。因此本轮没有宣称恢复能力大幅改善，只能确认精度增益没有造成可见稳定性退化。

三个局部前向模型的训练 RMSE 分别为：

- seed 0：`0.000210 m/s`；
- seed 4：`0.007247 m/s`；
- seed 7：`0.000752 m/s`。

## 6. 产品接口与可审计字段

新增命令：

```text
rosclaw growth episodic-contact-memory ...
rosclaw growth evaluate-episodic-contact-memory ...
rosclaw goalforge free-kick-showcase run --episodic-contact-memory ...
```

轨迹新增精确教师前动作观测、状态/目标支持位、选择的审计 seed、上下文距离、任务力和直接关节力矩。结果 schema 能区分“未执行”“状态 OOD”“目标 OOD”和“实际激活”，避免把安全归零误报为学习成功。

## 7. Artifact

- memory hash：`sha256:26da81c603ad4d90256855a045d86e79de09f91dfe1ddbb7777e18537cd417d5`
- memory 文件 hash：`sha256:118956e13ae4e346f06c99fe8e6013803bf88b81942a1810a1c69f3d25087ce5`
- 物理实现 hash：`sha256:83d7158250bbfc620b66aa8fcbe62d9c183acfa47b29f093eb1fae26d5e6b1da`
- Growth evaluation hash：`sha256:d0ea1b79eab245b5eb5448749d3ee24e8fff3494cde9aa98292e72e9b4bdef8f`
- 外部证据根：`/code/rosclaw/rosclaw_football/evidence/age10-episodic-final-*v3/`
- memory：`/code/rosclaw/rosclaw_football/evidence/age10-episodic-contact-memory-v3.json`
- evaluation：`/code/rosclaw/rosclaw_football/evidence/age10-episodic-contact-evaluation-v3.json`

## 8. 局限与下一步

1. 三个状态岛中只有 seed 0 在当前单飞弹道模型下找到合法球门目标内的可执行出球需求；主动覆盖不是完整的。
2. seed 4、7 的球会经历地面接触/反弹，单飞抛体公式不能正确描述落点。下一步应学习 `任务力 + 接触状态 -> 球门平面落点/到达时间`，把反弹动力学直接纳入世界模型。
3. 当前是局部线性 episodic memory，不是端到端神经小脑，也不是持续在线更新的 actor-critic。
4. 下一阶段应加入支撑脚接触、骨盆角速度、关节力矩余量和球自旋，并对 seed/摩擦/球质量做分组留出；冻结冠军复演继续作为稳定性约束。
5. 只有多个未见状态岛、多个合法目标、移动来球和恢复门都通过，才能把 `active_context_coverage_complete` 置为真。

## 9. 代码验证

- 新旧触球小脑与 free-kick 定向回归：`40 passed, 2 deselected`；
- `ruff check src tests`：通过；
- `mypy src`：1222 个源文件通过；
- 完整非 slow 回归：`6842 passed, 114 skipped, 19 deselected, 6 failed`；
- 6 个失败均已隔离复跑：隐藏本机 Codex CLI 后 2 项 `passed`，显式绑定本机 LeRobot 0.6.1 runtime 后 4 项 `passed`。它们是测试环境假设，不是本轮代码失败。
- 所有正式仿真 evidence 均为 strict replay，像素不参与评分。
