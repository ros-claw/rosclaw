# Phase 8：SONIC 全身神经助跑候选 v2 实施与复盘

## 结论先行

本轮把 G1 的助跑从“15 关节步行网络 + 14 个手臂回零 PD”升级成了真正的 **29 关节全身神经策略**：GEAR-SONIC 规划器生成全身跑动参考，低延迟 encoder/decoder 读取 10 帧本体感历史并在 50 Hz 输出 29 关节动作，500 Hz PD 才把动作变成 MuJoCo 力矩。

视觉和速度都有实质进步：v2 在单一连续物理世界内跑了 3.34 m，物理峰值 1.94 m/s，跑动时有明显的摆臂、腾空与身体前倾；以 0.318 m/s 的剩余前进速度进入 0.35 s 技能桥，随后完成 9.32 m/s 射门，踢后未摔倒，最终机身速度为 0.00052 m/s。

但该候选 **没有晋升**。严格门控仍发现：门线误差 0.164 m（阈值 0.160 m）、离下角 0.364 m（阈值 0.250 m），而且上游 PD 需求有 73 个物理步触及硬力矩边界。视频因此明确标记为 `REJECTED SONIC CANDIDATE / DIAGNOSTIC ONLY`，不能作为宣传通过样例。

## 这次真正开发了什么

### 1. 可审计的 SONIC 身体基础

新增 `g1_sonic_runup.py`，实现：

- 对 planner、encoder、decoder、observation config 的存在性、ONNX 输入输出形状和 SHA-256 做 fail-closed qualification；
- 固定 MuJoCo / IsaacLab 29 关节语义映射；
- 将 30 Hz 生成式全身参考连续重采样到 50 Hz；
- 按官方部署语义构造 1247 维 encoder 输入：未来 10 帧关节位置、速度和相对根姿态；
- 按官方部署语义构造 994 维 decoder 输入：64 维 token 加 10 帧角速度、关节位置、关节速度、上一动作和重力方向；
- decoder 直接输出全部 29 关节动作，使用官方 action scale、Kp/Kd 和本项目硬力矩上限闭合物理环；
- 规划参考本身只作为动作条件，不参与成绩判定；所有分数来自 MuJoCo 物理状态。

这不是把 SONIC 生成的 qpos 直接播放，也不是渲染层补丁。

### 2. 双身体基础接入同一足球闭环

`free-kick-showcase` 现在支持：

- `groot_history`：原有 516→15 历史步态网络；
- `sonic_fullbody`：新增 1247→64 encoder 与 994→29 decoder 的全身策略。

两者共享同一个球、球门、碰撞、RoboNaldo 射门先验、踢后恢复和严格复跑逻辑。SONIC 模式不重置或瞬移机器人，不另开世界。

### 3. 足球阶段合同

轨迹新增稳定的事件阶段编号：

| id | 阶段 |
|---:|---|
| 0 | APPROACH |
| 1 | ALIGN_BRAKE |
| 2 | PLANT_BRIDGE |
| 3 | LOAD |
| 4 | SWING |
| 5 | CONTACT |
| 6 | FOLLOW_THROUGH |
| 7 | RECOVERY |
| 8 | READY |

后续离线数据、失败课程、IQL/actor-critic 和候选门控可以按同一事件语义切片，不再只按固定时间切窗口。

### 4. 低维候选参数进入合同

技能桥时长、骨盆 yaw、右脚 yaw 和 COM 横向偏置已成为有边界、可哈希、可严格复跑的 flow 参数。它们是 Phase 8 规划中的 C1 低维动作层，可由 growth engine 搜索；当前仍不会触及硬件。

### 5. 拒绝候选诊断视频

视频导出新增显式 `--allow-rejected-candidate`。默认仍拒绝渲染失败证据；只有调用方明确要求时，才允许对严格复跑候选生成带有 `REJECTED / DIAGNOSTIC ONLY / NOT PROMOTED` 水印的视频。manifest 同时记录：

- `source_evidence_passed=false`
- `candidate_only=true`
- `visualization_only=true`
- `pixels_used_for_scoring=false`

## 实验结果

### 与当前通过基线对比

| 指标 | 通过基线 v4 | SONIC v1 | SONIC 动态交接 v2 |
|---|---:|---:|---:|
| 严格复跑 | 是 | 是 | 是 |
| 最终晋升 | **PASS** | REJECT | REJECT |
| 助跑距离 | 3.409 m | 3.001 m | 3.343 m |
| 助跑峰值速度 | 1.318 m/s | 1.810 m/s | **1.941 m/s** |
| 交接速度 | 0.217 m/s | 0.206 m/s | **0.318 m/s** |
| 技能桥时长 | 0.60 s | 0.60 s | **0.35 s** |
| 技能桥最大关节差 | 1.157 rad | **0.318 rad** | 0.463 rad |
| 交接到触球 | 2.772 s | 2.750 s | **2.526 s** |
| 射门峰值速度 | 8.731 m/s | 8.773 m/s | **9.321 m/s** |
| 门线目标误差 | **0.0078 m** | 0.1980 m | 0.1640 m |
| 离下角距离 | **0.2078 m** | 0.3979 m | 0.3640 m |
| 踢球期最低骨盆高度 | 0.699 m | 0.686 m | 0.697 m |
| 踢后摔倒 | 否 | 否 | 否 |
| 最终机身速度 | **0.00013 m/s** | 0.00067 m/s | 0.00052 m/s |
| 力矩需求触顶步数 | **0** | 51 | 73 |

v2 相比 v1 的变化不是调视频：速度指令从 1.3 增至 1.5 m/s，执行窗口从 3.52 调整为 3.40 s，起点相应后移到 -3.40 m；机器人在已回正但仍保留 0.318 m/s 前进速度时进入 0.35 s 技能桥。这使助跑更像跑步，交接更早，射门更快，门线误差从 19.8 cm 降到 16.4 cm。

### 参数与接触敏感性复盘

本轮实际扫描了：

- SONIC gain：0.72、0.95、1.00；
- 速度/执行窗：1.3/3.52、1.4/3.40、1.5/3.30、1.5/3.40；
- bridge：0.35、0.40、0.45、0.50、0.60 s；
- kick phase start：145、150、155；
- aim bias：0.295、0.299、0.300、0.305、0.367、0.500 m；
- 起始 x/y、foot yaw、pelvis yaw、COM shift 多组局部探针。

最重要的发现是：射门落点对触球前几厘米的纵向位置和支撑相位呈非平滑变化。起点从 -3.150 m 改到 -3.075 m，虽然只改变约 7.5 cm，门线 y 可以从约 0.80 m 跳到约 1.38 m；简单线性插值反而可能落到约 0.07 m。目标偏置也存在同样的接触分支切换。因此继续用标量网格“磨参数”不能真正突破。

## 视觉复查

逐帧复查确认：

- 跑动阶段已有明显的双臂协调、屈膝、蹬地、腾空和躯干前倾，比原来的慢走更接近球员助跑；
- v2 比 v1 更晚刹停，进入球前仍有前进动量；
- 但在 `PLANT_BRIDGE → LOAD` 之间仍能看出动作语汇切换，身体短暂变得直立，尚未达到职业球员式连续蓄力；
- 踢后没有倒地或持续后退，最终能稳定站住；
- 射门进门但没有命中声明的下角目标，所以视觉“进球”不能替代精度失败。

## 为什么仍不算统一端到端小脑

SONIC 已经是真正的全身神经小脑，但当前整体仍是两个技能模型：

1. SONIC planner + encoder/decoder 负责 APPROACH/ALIGN_BRAKE；
2. RoboNaldo motion-conditioned policy 负责 LOAD/SWING/CONTACT/FOLLOW_THROUGH/RECOVERY；
3. 中间仍靠一个连续物理、速度匹配的 0.35 s bridge。

所以“跑得像人”已经明显改善，“跑—刹—踢一气呵成”尚未根治。视频中短暂直立就是架构边界的可见表现。

## 下一步：真正需要训练的突破点

### P8-A：接触前 0.8 s 统一 transition actor

冻结 SONIC body foundation 和现有射门/恢复专家，训练一个只覆盖 `ALIGN_BRAKE → PLANT_BRIDGE → LOAD → early SWING` 的 residual actor。输入包括：

- 10 帧本体感；
- 球相对位置/速度；
- 支撑脚接触、COM、capture point；
- SONIC token；
- 射门专家 future target；
- 目标球门点。

输出先限制为 C1：COM shift、步幅、相位速率、骨盆/脚 yaw 和两侧摆臂残差；通过后再扩大到 C2 关节目标残差。目标是让 bridge 从手工 quintic 变成可学习的动态动作，而不是直接上 29 关节无保护 torque RL。

### P8-B：接触事件分支课程

围绕当前敏感区采样起始 x/y、相位、摩擦、球半径和球初速，按 CONTACT 事件切片，训练：

- 触球模式分类器：脚背/脚内侧/擦碰/失触；
- 落点 critic；
- mixture-of-experts transition policy；
- 对模式切换边界的熵和 margin 约束。

这比继续把 aim bias 调到小数点后三位更可能得到可泛化的死角命中。

### P8-C：安全 authority projection 与蒸馏

当前 73 个上游力矩需求触顶步必须保留为失败。下一候选要加入可审计的 joint-wise target/torque authority projection，并把 projection penalty 纳入训练；再将 SONIC 动作蒸馏到本项目 G1 模型上，使未投影需求本身也回到硬边界内，而不是只在最后 clip。

### P8-D：连续学习和保留集

只在上述 transition actor 上做在线 actor-critic；SONIC 和通过基线先冻结。每次更新必须同时通过：

- 当前 contact curriculum；
- v4 通过基线保留集；
- 摩擦/延迟/质量随机化；
- 力矩、跌倒、最终稳定和严格复跑门控。

这把 stability-plasticity dilemma 限制在一个可回滚的小模块内，避免为了学会“跑着踢”而遗忘已经通过的射门精度与踢后恢复。

## 证据位置

- v2 严格证据：`/code/rosclaw/phase8_evidence/g1-sonic-player-v2/g1-free-kick.json`
- v2 轨迹：`/code/rosclaw/phase8_evidence/g1-sonic-player-v2/g1-free-kick-trajectory.npz`
- v2 拒绝候选诊断视频：`/code/rosclaw/phase8_evidence/g1-sonic-player-v2/g1-sonic-player-v2-rejected-diagnostic.mp4`
- v2 视频 manifest：`/code/rosclaw/phase8_evidence/g1-sonic-player-v2/g1-sonic-player-v2-rejected-diagnostic.json`
- v1 严格证据：`/code/rosclaw/phase8_evidence/g1-sonic-player-v1/g1-free-kick.json`

所有证据均为 `SIM_ONLY`；没有连接真实机器人，没有发送硬件命令，也没有停止或占用用户已有 GPU 任务。
