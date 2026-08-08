# Phase 8 C4：连续助跑射门与 Growth 教师残差学习闭环报告

日期：2026-08-07
范围：仅 CPU MuJoCo / SIM_ONLY；未连接真实机器人；所有候选默认不可部署、不可晋升。

## 1. 结论先行

本轮同时取得了一个明确进步和一个有价值的拒绝结果：

1. **快而准的低角度射门**：把 SONIC 助跑到踢球的直接交接推进到第 210 策略帧，交接到触球时间从早期课程的 1.344 s 降为 **0.978 s**，触球前运动停顿为 **0 s**；球以 **10.211 m/s** 穿过球门平面，平面目标误差为 **0.0241 m**。这是当前最连贯、最精确的直接射门开发样本。
2. **高角度教师残差模型**：用 8 个不同规划种子的连续物理 episode 构造 875 条接近—制动—支撑—摆腿—触球 transition，训练了 3 个 IQL 教师残差候选。候选真实进入闭环后，把保留种子上的执行器饱和步数从 50 降到 **13**，左上角距离由 0.7660 m 降到 **0.7334 m**，但峰值踢球倾角增加 0.0172 rad，绝对射门精度也仍未达标。因此 Growth 闸门正确给出 **REJECTED**，没有把“局部改善”冒充“能力晋升”。

所以，本轮不是“高角任意球已经解决”。真正完成的是：连续性课程得到一条可见突破；教师动作第一次被转成可执行的数据驱动残差“小脑”；同版本保留集验证发现 Stability–Plasticity 冲突并阻止错误晋升。

## 2. 为什么以前跑到踢会停顿

助跑网络和射门先验原本是两个不同动力学分布：

- SONIC 输出 29 关节、带速度和相位的全身动作；
- 射门先验期望较稳定的支撑姿态，并从固定摆腿相位开始；
- 交接过早，机器人会为了满足射门初始姿态而“等一等”；
- 交接过晚，残余前向动量和关节速度超过射门控制器的捕获能力，出现饱和、倾斜、后退甚至跌倒。

本轮把“停顿”从肉眼印象改成事件量：`handoff_to_contact_sec` 测量技能交接至首次真实接触，`pre_contact_motion_pause_sec` 测量触球前是否停止运动。同时在每个 20 ms 策略帧内锁存 2 ms 物理子步的瞬时接触，解决了此前已经记录接触时间、却偶尔漏掉 `CONTACT` 事件的错误。这样数据分段和视频标注都由同一物理事件驱动。

## 3. 连续性课程实验

固定 SONIC 助跑、0.16 s 速度匹配 bridge、摆腿幅值 1.0、速度比例 1.05，逐步把射门起始帧后移：

| 射门起始帧 | 交接至触球 | 目标平面误差 | 结果解释 |
|---:|---:|---:|---|
| 190 | 1.344 s | 0.138 m | 稳定但仍有明显等待 |
| 200 | 1.160 s | 0.298 m | 更连贯，精度下降 |
| 210 | 0.978 s | 0.307 m | 首次进入 1 s 内，仍需修正横向瞄准 |
| 220 | 0.798 s | 0.489 m | 更快但饱和和倾角明显增加 |
| 230 | 未触球 | — | 跨过可捕获边界 |
| 240 | 未触球 | — | 发生跌倒/关节边界失败 |

这组负样本很重要：更短交接并非单调更好，210–220 帧附近是当前混合控制器的动力学边界。随后在第 210 帧扫描横向瞄准偏置，误差从 0.307 m 逐步下降到 **0.0241 m**。这说明当前低角度精度误差主要是可重复的接触几何偏差，而不是随机噪声，适合进入 ILC/残差学习闭环。

当前快而准样本仍未通过全部门控：执行器饱和 20 步、峰值需求比例 1.428，助跑峰值倾角 0.495 rad。它是开发突破，不是可部署策略。

## 4. 数据驱动“小脑”做了什么

### 4.1 训练数据

高角目标采用 SIM-only 操作空间 loft teacher 产生受控探索。8 个规划种子都在同一 MuJoCo 世界中完成连续助跑、支撑、触球和恢复，teacher 激活约 65–67 帧。Growth adapter 将轨迹转成：

- 110 维状态：29 关节位置、29 关节速度、本体高度/速度/姿态、球相对状态、9 个事件相位、29 维基础关节目标；
- 29 维已执行力矩与 29 维 teacher residual；
- 具名 reward vector：相位推进、接近球、直立、动作平滑、触球速度、终端精度；
- 具名 safety cost vector：力矩投影、过驱动、低骨盆、过度倾斜、episode 安全失败；
- 以 episode 而非随机帧划分 train、validation、reserved，防止同一轨迹泄漏。

最终 manifest 包含 **8 episodes / 875 transitions**，内容哈希为 `sha256:28993c89...`，manifest 哈希为 `sha256:68edbcc7...`。数据只允许离线学习，不允许直接产生部署真值。

### 4.2 IQL 语义修复

旧 IQL actor 只把网络输出解释为“目标绝对力矩”，再减去 baseline 得到 residual，无法正确学习老师给出的增量。本轮增加明确的 `action_source`：

- `executed_action`：保持原恢复任务兼容，网络输出绝对执行力矩；
- `teacher_residual_action`：网络直接输出 teacher residual，运行时不再错误减去 baseline；
- 候选元数据记录 `actor_output=sim_teacher_residual_torque_nm`、状态特征、reward/cost 名称和训练来源；
- 只有 approach-to-strike 数据且 train/validation/reserved 都含真实 teacher 激活样本时，才允许训练 teacher residual；
- 运行时仍经过标准化支持域、置信缩放、力矩上限、关节组和 residual fraction 限制，模型不能绕过基础控制器和硬安全边界。

在 GPU 0 上训练的三个 2000-step IQL 候选，validation normalized MSE 为：

| seed | 训练前 | 训练后 | 相对下降 |
|---:|---:|---:|---:|
| 20260807 | 0.19536 | 0.00968 | 95.0% |
| 20260817 | 0.19095 | 0.00740 | 96.1% |
| 20260827 | 0.19201 | 0.01116 | 94.2% |

低监督误差不等于闭环足球能力。最终物理效果最好的候选反而是 seed 20260827，因此 Growth 以真实闭环指标而不是 validation loss 选取候选。

## 5. Stability–Plasticity 严格复核

对同版本、同目标、同规划种子 7 的冻结 baseline 与学习候选进行比较：

| 指标 | baseline | learned residual | 变化 |
|---|---:|---:|---:|
| 学习残差实际作用比例 | 0 | 5.36% | +5.36 pp |
| 执行器饱和步数 | 50 | 13 | **-37** |
| 目标平面误差 | 0.53516 m | 0.53394 m | **-0.00122 m** |
| 左上角距离 | 0.76597 m | 0.73341 m | **-0.03256 m** |
| 峰值踢球倾角 | 0.35834 rad | 0.37551 rad | **+0.01718 rad（退步）** |
| 跌倒/关节越界 | 无/无 | 无/无 | 无新增硬失败 |

严格闸门中，candidate executed、minimum effect、precision non-regression、authority non-regression、settling、strict replay 和无新增安全失败均通过；`tilt_non_regression` 与 `absolute_task_gate` 未通过。最终处置为 **REJECTED**，promotion/activation/hardware authorization 全为 false。

这就是当前对 Stability–Plasticity Dilemma 的工程回答：允许模型学会新东西，但只有在保留旧稳定性且达到绝对任务线时才合并到“肌肉记忆”。新模型降低饱和并改善角点方向，说明有 plasticity；倾角变差，说明 stability 尚未守住，所以它只能进入失败记忆和下一轮训练，不能覆盖冻结 baseline。

## 6. ROSClaw Growth 闭环现在怎样运转

```text
本体/环境轨迹
    -> 物理事件锁存与严格重放
    -> Growth triage（失败签名与 learner routing）
    -> episode 级 transition dataset
    -> teacher residual IQL 候选
    -> 支持域/幅值/安全投影后的闭环执行
    -> 同版本 baseline + reserved seed 对照
    -> Stability / Plasticity / absolute task 闸门
    -> ACCEPT 才可进入后续晋升；REJECT 回流为负样本和课程边界
```

本轮两个最终样本的 triage 也给出了不同方向：低角度样本是稳定、可重复的误差，路由至 ILC/motion tracking；高角候选同时包含接触精度、角点未命中和力矩投影问题。多 episode 数据具备 transition/reward/cost 后，才路由到 IQL。也就是说 learner 由证据结构选择，而不是把所有问题都硬塞给 RL。

## 7. 可复核证据与视频

### 快而准的直接射门

- 严格证据：`/code/rosclaw/phase8_evidence/g1-growth-c4-fast-precision-seed0/g1-free-kick.json`
- Growth triage：`/code/rosclaw/phase8_evidence/g1-growth-c4-fast-precision-seed0/growth-triage-c4.json`
- 20.9 s 诊断视频：`/code/rosclaw/phase8_evidence/g1-growth-c4-fast-precision-seed0/g1-fast-precision-direct-diagnostic.mp4`
- 视频 SHA-256：`4700d41c2668751ef3a1ab2a404f560811d2273662bfe75fcda9ab6497d60d85`

### 高角学习残差候选

- 同版本 baseline：`/code/rosclaw/phase8_evidence/g1-growth-c4-learned-baseline-seed7/g1-free-kick.json`
- 学习候选证据：`/code/rosclaw/phase8_evidence/g1-growth-c4-learned-upper-seed7/g1-free-kick.json`
- 严格对照闸门：`/code/rosclaw/phase8_evidence/g1-growth-c4-learned-upper-evaluation.json`
- Growth triage：`/code/rosclaw/phase8_evidence/g1-growth-c4-learned-upper-seed7/growth-triage-c4.json`
- 21.5 s 诊断视频：`/code/rosclaw/phase8_evidence/g1-growth-c4-learned-upper-seed7/g1-learned-upper-residual-diagnostic.mp4`
- 视频 SHA-256：`b4be1b8e1377cdc829815ef4888f745222fbf51af3b6113510ce8dd18044ee84`

两个视频均从严格重放轨迹渲染；画面明确标注 `REJECTED CANDIDATE / SIM ONLY`。本轮还把此前容易误导的 `NO PRE-KICK PAUSE` 字样改为实测 `HANDOFF-CONTACT X.XX s`。

## 8. 下一轮开发计划

当前最重要的不是继续手扫一个固定踢球参数，而是让 Growth 学会“不同球路用不同小脑”，并让稳定性直接进入训练目标：

1. **双技能/混合专家**：冻结已经达到 0.978 s / 2.41 cm 的低角度技能；另建高角度 expert，使用目标高度、左右角、来球速度和支撑状态作为 context，禁止一个残差模型同时平均互相冲突的低平球与高球动作。
2. **接触时机 actor**：把 180–230 帧课程边界变成可学习的离散/连续 timing action；以成功触球、handoff latency、capture point、饱和和倾角为联合回报，不再固定选择第 210 帧。
3. **稳定性 critic**：加入 capture point、质心相对支撑多边形、角动量、支撑脚滑移、踢后恢复步数和躯干 jerk。倾角回退必须在训练阶段就被 critic 处罚，而不是只在最终门控发现。
4. **32+ 分层 episode**：按左右脚、左右角、高低目标、规划种子和轻量扰动分层采样；保留独立 sealed validation/holdout。当前 8 个 episode 只能证明管线和初步可学习性，不能证明泛化。
5. **SIM 在线 actor-critic**：IQL 先从 teacher/replay 得到安全初始策略，再在仿真中以支持域约束做在线 rollouts；优先学习支撑腿、躯干、手臂反摆和恢复步 residual，踢球腿保持课程化解冻。所有在线更新先成为 candidate，永不热替换冻结冠军。
6. **持续学习防遗忘**：保留冠军 replay、按技能专家分离参数、用旧技能 non-regression suite 和参数/行为蒸馏约束更新。出现 tilt、跌倒、关节越界或既有技能回退，自动 rollback 并把失败写回 MEMORY。

下一轮的明确验收目标是：在左/右上角的分层 development set 上，把 handoff-to-contact 控制在 1.0 s 左右、角点误差压入 0.16 m、零跌倒/零关节越界，同时饱和和峰值倾角均不劣于冻结 baseline。达到这些条件后，才进入新的 sealed holdout，而不是先制作“成功”宣传结论。

## 9. 本轮验证

- Ruff：相关 Growth、IQL、GoalForge/SimForge 源码与测试全部通过；
- mypy：13 个相关源文件通过，0 issue；
- Growth 与全部 G1 SimForge 非 integration 回归：**272 passed，6 deselected**（其中本轮聚焦集合 51 passed）；
- 真实 MuJoCo 资产 integration：**2 passed，9 deselected**，包含连续射门严格重放和原生碰撞球门；
- C4 baseline/candidate 均为同一 implementation hash，严格重放为 true；
- 代码与证据全程未发送硬件命令。
