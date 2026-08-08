# Phase 8 C11：在线弹道小脑、教师蒸馏与多种子否决报告

日期：2026-08-08
范围：G1 / MuJoCo / SIM_ONLY；没有连接真实机器人，没有发送硬件命令。主机为 4×RTX A6000，本轮用四路任务并行做课程采样，MuJoCo 物理和 SONIC ONNX 仍运行在 CPU；A6000 用于 EGL 视频渲染，未宣称 GPU 物理训练。

## 1. 结论先行

本轮在固定 seed 0 上取得了真实但有限的高球突破，同时用新多种子门证明它还不能晋升为通用球技。

1. 新增六维右腿触球残差“小脑”。它只在触球附近短窗口修改关节目标，不直接输出力矩，后面仍有 PD、硬力矩投影和关节边界保护。
2. 新增 replay-stabilized quadratic contextual actor-critic。它从严格 MuJoCo 成败轨迹拟合 critic，以最佳安全轨迹为 replay anchor，在信赖域内提出新动作；任何提议都必须重新进入物理世界，模型预测不能替代验证。
3. 在 5 m 左上目标 `(y,z)=(1.0,1.35) m` 上，无触球残差基线门线位置为 `(0.940,0.598) m`、误差 `0.7548 m`；新的无教师稳健开发候选达到 `(0.902,0.945) m`、误差 `0.4165 m`。误差下降 **44.8%**，过门高度增加 **34.7 cm**，垂向出球速度从 `4.336` 提升到 `5.103 m/s`。
4. 该候选仍保持连续跑射：交接到触球 `0.928 s` 左右、最低前向速度 `0.265 m/s`、无低速停顿；无摔倒、无关节越界、无硬力矩越限。
5. 找到非常窄的接触模式边界：髋俯仰残差 `-0.0218 rad` 时球高 `0.956 m`，到 `-0.0220 rad` 就骤降到 `0.315 m`。因此没有选择单点更高的悬崖边候选，而选择离边界更远的 `-0.015 rad` 作为开发复验动作。
6. 4 个未见规划种子全部暴露泛化失败：平均门线高度仅 `0.115 m`，2/4 连续性不合格，1/4 跌倒。新命令 `growth evaluate-ballistic-contact` 以五项失败码拒绝候选，activation/promotion/hardware 均为 false。

所以本轮的准确表述是：**seed 0 高球技能和 ROSClaw 学习闭环取得突破；通用高球能力尚未突破，候选已被泛化门正确拦截。**

## 2. 新增和加固的 ROSClaw 模块

### 2.1 有界触球残差

文件：`src/rosclaw/growth/ballistic_contact_residual.py`

- 动作维度：右髋 pitch/roll/yaw、右膝、右踝 pitch/roll 共六维；
- 采用接触中心的平滑 `sin²/cos²` 包络，避免阶跃目标；
- 单关节开发上限 `0.25 rad`；
- 不直接输出 torque，不绕过现有安全投影；
- 每条证据记录是否执行、激活帧数、目标残差峰值和逐帧残差数组。

### 2.2 在线 episodic actor-critic

文件：`src/rosclaw/growth/ballistic_contact_actor_critic.py`

产品命令：

```text
rosclaw growth ballistic-contact-actor-critic \
  --evidence-json ... --output ...
```

训练输入必须同时满足：严格双回放、轨迹内容哈希有效、Body 一致、实现哈希一致、除动作外实验上下文一致、动作互不重复。reward 由真实目标误差、垂向出球速度、连续性、身体硬安全、执行器饱和和峰值需求共同构成。

第一代 critic 从 12 条成功/失败 replay 中提出六维联合动作。正确候选哈希下的严格实测把误差从 `0.6175` 降到 `0.5129 m`，证明 actor 产生了有效新信息。第二代在 9 条局部 replay 上把 critic 留一法 RMSE 降到 `0.0154`，但其大步联合提议又被物理验证否决。这说明低 critic 拟合误差不代表混合接触系统可平滑外推。

### 2.3 2 ms 教师遥测加固

文件：`src/rosclaw/simforge/g1_free_kick_showcase.py`

发现旧实现只记录每个 20 ms 控制帧的最后一个 2 ms 物理子步。教师力可能在该帧中间进入足球距离门，改变了轨迹，但帧末已经退出，于是证据错误声称 `active_frames=0 / peak_force=0`。

修复后：

- 物理仍只施加当前子步力矩；
- 遥测对 10 个物理子步聚合 `any-active`；
- 保存帧内最大垂向/前向力和最大关节力矩样本；
- 不再允许“教师改了轨迹但证据称没用教师”。

修复后的 80 N 示范准确记录 `1` 个控制帧激活、峰值教师 torque `25.46 Nm`。它只是 SIM_ONLY 训练标签，没有进入自主候选的最终执行。

### 2.4 多种子 fail-closed 评估

文件：`src/rosclaw/growth/ballistic_contact_evaluation.py`

产品命令：

```text
rosclaw growth evaluate-ballistic-contact \
  --evidence-json ... --output ...
```

它校验同一动作、唯一规划种子、同一 Body/实现/任务上下文、轨迹哈希和严格回放，并聚合：硬安全、跑射连续性、强制射门覆盖、最坏目标误差、最低过门高度、最大饱和步。单个 seed 的好成绩不能使它通过。

本轮实际 CLI 返回码为 `3`，失败码：

- `HARD_SAFETY_REGRESSION`
- `CONTINUITY_GENERALIZATION_FAILED`
- `WORST_CASE_TARGET_ERROR_FAILED`
- `HIGH_TARGET_GENERALIZATION_FAILED`
- `SATURATION_GENERALIZATION_FAILED`

评估证据：`/code/rosclaw/phase8_evidence/g1-growth-c11-ballistic-contact-gen21-holdout-evaluation.json`

## 3. 数据闭环经过

```text
OmniContact 触球运动先验 + 冻结 IQL approach residual
  -> 六维接触动作课程
  -> 单世界 MuJoCo 严格双回放
  -> 2 ms 触球几何/出球速度/教师峰值 + 连续性 + 身体安全评分
  -> actor-critic 提议
  -> 新提议重新进入物理世界
  -> 教师只生成探索方向，不算自主成绩
  -> 分量蒸馏为无教师肌肉记忆
  -> 接触悬崖检测与保守选点
  -> 未见 seed 多种子门
  -> 失败则保持 REJECTED，不替换冠军
```

本轮共产生 85 条 C11 严格回放 episode。中途有 16 条误用了另一份 IQL candidate（hash `0a886...`，正确上下文为 `66c8...`）；它们被保留为可审计物理数据，但从同上下文比较和 actor 训练中隔离，没有用来证明改善。

## 4. 关键实验结果

| 候选 | 门线 `(y,z)` m | 目标误差 m | 出球 `vz` m/s | apex m | 连续 | 跌倒/关节越界 | 饱和步 |
|---|---:|---:|---:|---:|---|---|---:|
| C10/C11 无触球残差基线 | (0.940, 0.598) | 0.7548 | 4.336 | 0.937 | 是 | 无/无 | 23 |
| 单踝 pitch 0.25 | (0.683, 0.827) | 0.6120 | 4.898 | 1.138 | 是 | 无/无 | 23 |
| actor 第一代联合动作 | (0.926, 0.843) | 0.5129 | 4.919 | 1.146 | 是 | 无/无 | 23 |
| actor + 踝 pitch 0.25 | (0.889, 0.877) | 0.4855 | 5.024 | 1.185 | 是 | 无/无 | 23 |
| 教师蒸馏、无教师 hp=-0.015 | **(0.902, 0.945)** | **0.4165** | **5.103** | **1.216** | **是** | **无/无** | 23 |
| 悬崖边 hp=-0.0218（未选） | (0.904, 0.956) | 0.4052 | 5.116 | 1.221 | 是 | 无/无 | 23 |
| hp=-0.0220 | (0.937, 0.315) | 1.0365 | 4.314 | 0.932 | 是 | 无/无 | 24 |

### 多种子结果

| seed | 门线 `(y,z)` m | 误差 m | 连续 | 跌倒 | 饱和步 |
|---:|---:|---:|---|---|---:|
| 1 | (0.044, 0.115) | 1.5617 | 否 | 否 | 8 |
| 2 | (0.528, 0.115) | 1.3220 | 否 | **是** | 20 |
| 3 | (0.640, 0.115) | 1.2865 | 是 | 否 | 34 |
| 4 | (1.762, 0.115) | 1.4512 | 是 | 否 | 29 |

失败根因不是固定触球窗口偏了几帧：各 seed 实际触球时间都约 `4.284–4.336 s`。真正差异是 SONIC 交接本体状态变化很大，例如 handoff yaw 从 `+0.159` 到 `-0.188 rad`、骨盆横向位置从 `0.105` 到 `0.251 m`。因此下一步必须让动作条件化于本体状态，而不能继续把 seed 0 的六个常数复制给所有来球姿态。

## 5. 视频和证据

- 新的 5 m 高目标开发视频：`/code/rosclaw/phase8_evidence/g1-growth-c11-high-target-development.mp4`
- 20.9 s / 627 帧 / 1280×720 / 30 fps / H.264
- SHA-256：`47714a709faaff345a4f6aca3fe2ef8427de020ec671b5b5931576410536ee21`
- 源证据：`/code/rosclaw/phase8_evidence/g1-growth-c11-ballistic-contact-gen17-hip-p-fine-m015-seed0/g1-free-kick.json`
- Growth triage：同目录 `growth-triage-c11.json`

视频来自严格物理轨迹重放，像素不参与评分。由于目标误差仍超出 0.16 m 精度半径且多种子失败，画面明确带 `REJECTED / DIAGNOSTIC ONLY / SIM_ONLY` 水印。

## 6. 验证

- C11 物理 episode 均执行严格双回放并保存 trajectory digest；
- 新 actor-critic、残差、多种子评估、教师和 free-kick 定向测试：`24 passed, 2 deselected`；
- Ruff：通过；
- mypy：通过；
- 视频经 ffprobe 核验为 20.9 s、627 帧、1280×720、30 fps；
- 所有新 artifact 均声明 SIM_ONLY，不允许在线热替换、promotion 或 hardware activation。

## 7. 下一轮明确任务

1. 把 handoff yaw/roll/pitch、骨盆位置与速度、支撑腿速度、球相对足端位置加入 actor 状态，训练 contextual actor，而不是一个固定六维常数。
2. 每个训练 batch 同时采样多 seed，并混入 seed 0 冻结冠军 replay；reward 的最坏分位数和 CVaR 必须进入 critic，解决 Stability–Plasticity。
3. 用多专家课程分别覆盖高 yaw、横向偏置和低前向速度，再由本体路由器选择；不存在安全专家时必须 fallback/拒绝。
4. 将教师示范的完整关节力矩方向蒸馏进受限 residual SAC，但最终评估教师力必须为零。
5. 先要求 4 seed 全部连续、无跌倒且过门高度不低于 0.65 m，再扩到 16/32 seed；没有达到前不制作“成功晋升”宣传结论。
6. 继续降低当前 23 个饱和子步；任何高球提升若以饱和、后退或踢后稳定退化为代价，都不得替换稳定冠军。
