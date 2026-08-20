# Phase 8 C9：远距离高目标连续射门与冲击可学习性闭环

日期：2026-08-07
范围：G1 / CPU MuJoCo / SIM_ONLY；未连接真实机器人，未发送硬件命令。

## 1. 结论先行

本轮把用户提出的三个要求同时放进同一个连续物理实验：跑动到踢球不停止、球门更远、目标更高。

- 球从 `x=1.0 m` 射向 `x=5.5 m` 门线，射程 **4.5 m**，比上一代 4.0 m 基准增加 12.5%。
- 目标从贴地的 `z=0.115 m` 提高到 `z=0.65 m`，提高 0.535 m；目标横向为 `y=1.0 m`。
- G1 连续助跑 3.31 m，助跑峰值速度 **1.843 m/s**；控制器交接至触球为 **1.448 s**，触球前实测运动暂停为 **0.000 s**。
- 球在门线实际通过 `(y,z)=(1.0373, 0.6045) m`，声明目标误差 **0.0588 m**；无跌倒、无关节越界，最终身体平移速度约 `0.00018 m/s`。
- 该 episode 仍有 10 个 2 ms 执行器过载物理步，峰值需求比 1.210；SONIC 助跑最低骨盆为 `0.662 m`、峰值倾角为 `0.496 rad`，仍低于/高于严格姿态线；同时它命中的是自定义高目标而非既有上下角中心。因此证据明确标记为 **REJECTED DEVELOPMENT CANDIDATE**，没有冒充 promotion 或真实机器人能力。

结果说明“跑远一点、不中途停、高目标仍能命中”已经得到严格重放的开发级突破；“零饱和、上角死角、多种子泛化”仍未解决。

## 2. 本轮工程改动

### 2.1 距离和高度成为任务上下文

- `G1TrainingGoalSpec` 的门线位置可在 4–12 m 内配置；CLI 新增 `--goal-plane-x-m`。
- 结果新增 `shot_distance_m`，证据不再只能通过场景配置间接推断射程。
- 新增有界 `aim_bias_z_m`，把“策略为了抵消重力而瞄得更高”与“真正用于评分的目标高度”分离。物理策略使用 `target_z + aim_bias_z`，评分仍只使用未篡改的 `target_z`。
- 上下文哈希保留旧 artifact 对零垂直偏置的兼容，但非零偏置会进入严格绑定，避免拿不同球路混训或误复用模型。

### 2.2 修复 Growth 看不到 2 ms 冲击峰值的问题

控制周期是 20 ms，内部 MuJoCo 物理步是 2 ms。旧轨迹的 `commanded_torque` 只保存每个控制帧最后一个物理子步；一次真正危险、但在帧末已经消失的冲击会被结果门控发现，却不会进入校准训练数据。

本轮新增每关节 `commanded_torque_peak_abs`：

1. 每个 2 ms 子步更新该控制帧的绝对力矩峰值；
2. 轨迹同时保留帧末有符号力矩和完整子步峰值；
3. SONIC authority learner 优先使用峰值字段，旧轨迹才回退到帧末字段；
4. 形状、有限值和非负性不满足时 fail closed。

在最终开发轨迹中，旧帧末数据没有暴露峰值，但新字段准确定位到摆腿期右髋偏航关节的 `1.2097 ×` 需求，时间为 `4.84 s`。这让失败第一次真正变成可学习数据，而不只是报告上的红灯。

### 2.3 修复 CONTACT 与运行时增益语义错位

运行时在首次接触锁存后使用 follow-through gain；旧校准器却只用 `FOLLOW_THROUGH(6)` 学习该增益，把 `CONTACT(5)` 只分给 strike gain。结果是最需要学习的接触冲击永远不会更新真正负责执行的增益。

authority calibration v5 现在把 `CONTACT + FOLLOW_THROUGH` 都用于 follow-through authority 学习，同时仍把 `LOAD + SWING + CONTACT` 用于 strike 观测。新 artifact 继续绑定 Body、implementation、源轨迹哈希、base calibration 和 SIM_ONLY 权限边界。

## 3. 参数课程与结果

本轮不是只跑一个“好看参数”，而是扫描了门线 5.5/6.0/6.5/7.0 m、目标高度、触球相位 P180–P220、横向/垂向瞄准、摆幅、摆速、质心偏置、残差比例和增益启动帧。关键结论如下：

| 阶段 | 门线目标误差 | 关节越界 | 饱和步 | 峰值需求比 | 结论 |
|---|---:|---|---:|---:|---|
| 4.5 m / P180 / 幅值 1.14，旧 impact 学习前 | **0.0130 m** | 有（右踝约 0.001 rad） | 24 | 1.316 | 很准，但不安全 |
| contact-aware authority v5 | **0.0588 m** | **无** | 10 | 1.210 | 当前展示冠军；安全明显改善 |
| substep-aware authority 再学习 | 未过门线 | 无 | 5 | 1.112 | 继续降过载，但任务能力回退，拒绝替换冠军 |
| 更早启用强 authority | 未过门线 | 无 | 0 | <1.0 | 身体安全但不会完成足球任务，拒绝 |

这正是 Stability–Plasticity Dilemma 的实证：第二次学习让力矩更温和，却破坏了有效击球路径。Growth 必须保留“精度冠军”和“安全候选”的分别证据，只有同时守住任务和身体指标的新策略才允许替换；不能因为 loss 或单项安全指标变好就称为成长。

## 4. 当前 Growth 闭环

```text
远距离/高目标课程
  -> 单世界连续助跑、触球、进门、恢复
  -> 20 ms 状态 + 2 ms 冲击峰值严格证据
  -> 目标误差 / 角点 / 关节 / 力矩 / 身体稳定判分
  -> Growth triage
  -> contact-aware + substep-aware authority candidate
  -> 再次严格双回放
  -> 任务能力和 Stability 同时不回归才可替换冠军
  -> 失败候选回流为下一轮课程数据
```

对当前展示冠军运行 Growth triage 后，失败签名为：

- `declared_corner_miss`：它精确命中自定义高目标，但还不是现有 1.35 m 上角中心；
- `authority_projection_required`：仍有 10 个瞬时过载子步；
- learner route 为 `ILC + motion_tracking`，promotion 为 false；缺少多 episode reward/cost 和 online rollout 时，不会错误路由成已成熟在线 RL。

## 5. 通俗解释

以前可以把系统想成每 20 ms 才看一次“肌肉用了多大力”。但踢球冲击只可能持续 2–10 ms：机器人在两次观察之间猛地过载，然后在下一次观察前恢复正常。门卫看到了事故，教练的数据本却没有记下来，所以训练再多次也学不会避开同一个瞬时错误。

本轮相当于给每块关键肌肉增加了“这一帧最用力是多少”的峰值记忆，并纠正了“触球到底归挥腿教练还是随动教练负责”的分工。第一次学习把脚踝越界消掉并保住 5.88 cm 精度；第二次学习进一步减力，却把球路弄坏。ROSClaw 因此没有覆盖上一代，而是把这次失败留下来继续训练。这才是基于成功和失败成长，而不是把手调参数或更低训练 loss 当作自进化。

## 6. 证据与视频

- 当前 4.5 m 高目标冠军证据：`/code/rosclaw/phase8_evidence/g1-growth-c9-substep-telemetry-seed0/g1-free-kick.json`
- 轨迹：`/code/rosclaw/phase8_evidence/g1-growth-c9-substep-telemetry-seed0/g1-free-kick-trajectory.npz`
- Growth triage：`/code/rosclaw/phase8_evidence/g1-growth-c9-substep-telemetry-seed0/growth-triage-c9.json`
- 21.5 s / 1280×720 / 30 fps 视频：`/code/rosclaw/phase8_evidence/g1-growth-c9-substep-telemetry-seed0/g1-growth-c9-4p5m-high-development.mp4`
- 视频 SHA-256：`02c880b5cea661514ddaa38b6898ca3aaa40cca2f3ee4eda7899519db7ceee09`
- substep-aware calibration：`/code/rosclaw/phase8_evidence/g1-growth-c9-substep-authority-cli-v5-seed0.json`
- calibration hash：`sha256:647588729c46cdbaf7a5f4f2ab05e7e0879a7cb24cfc3ae9d38f8d8577eaf3eb`

视频连续段来自 strict physics replay；慢动作只用于可视化，像素不参与评分。视频带 `REJECTED CANDIDATE / SIM ONLY` 水印，这是对当前安全边界的真实表达。

## 7. 验证

- 当前冠军两次仿真的 result 和 trajectory digest 完全一致，`strict_replay=true`。
- 相关 Growth / football outcome / free-kick / joint guard：**25 passed，2 deselected**。
- Ruff：相关源码与测试通过。
- mypy：6 个相关源码模块通过，0 issue。
- ROSClaw 产品 CLI 已实际运行 `growth sonic-authority-calibration`、`growth free-kick-triage`、`goalforge free-kick-showcase run/export`。
- 视频 645 帧，时长 21.5 s；评分全来自 MuJoCo 物理状态。

## 8. 下一轮明确任务

1. 把当前人工扫描结果转成有界 action recipe learner：联合学习触球相位、横纵瞄准补偿、摆幅、摆速和 impact authority，而不是只学习一个离散相位。
2. 至少采集 32 个规划种子，并按 4.0/4.5/5.0 m、左右侧、0.5/0.65/1.0 m 高度分层；完整保留成功和失败反事实。
3. critic 同时惩罚子步峰值、关节边界、躯干角动量、支撑脚滑移、踢后后退和 jerk；足球未过门线必须是硬任务失败。
4. 采用冻结冠军 replay + 行为蒸馏 + 独立 sealed seeds 处理遗忘；新 actor-critic 只能成为 candidate，不能在线热替换冠军。
5. 在 4.5 m 自定义高目标达到零饱和后，再推进真正的 `z=1.35 m` 左/右上角以及来球初速度课程；当前 0.65 m 结果不能宣传为“上角死角已解决”。
