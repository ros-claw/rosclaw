# Phase 8：目标无关的有状态柔性球网 v1 实施与验证报告

日期：2026-08-11
证据上限：`SIM_ONLY`（不授权真实机器人、在线热切换或产品晋升）

## 1. 结论先行

本轮修复了视频中“足球进入球网后像碰到一堵墙，或像掉进水里”的真实动力学问题。旧模型只按当前穿透深度施加前后方向阻尼，球网没有接触记忆，横向和竖向速度也不能形成一个自然的网兜捕获过程。新模型在球第一次触及斜背网时绑定一个**由物理接触决定、与请求目标无关**的三维锚点，随后用有界三轴弹簧阻尼吸收球的动量，并留下可审计的接触、受力、形变和收敛轨迹。

最终结果：

- 24/24 条训练探针完成严格重放，只有一个实现哈希；
- 24 条轨迹均只发生一次球网捕获，0 次跌倒、0 m 触球后倒退；
- 球相对首次接触锚点的最终位移中位数为 `8.143 mm`，最大为 `176.796 mm`；最大值来自高位球在网兜内受重力自然下沉，不是向目标点吸附；
- 在同版本物理下重建 episodic memory 后，AIRBORNE 留出误差 `42.497 -> 15.279 mm`，BOUNCE 留出误差 `15.704 -> 3.848 mm`；
- 已知不支持的 seed 7 稳定锚点仍被 fail-closed 保留；评估 verdict 为 `DEVELOPMENT`；
- 两段 1920x1080、20.9 秒严格物理回放视频已导出，但由于测试目标不是完整验收要求中的真实球门死角，视频明确保留 `REJECTED / DIAGNOSTIC ONLY` 水印，未通过改标签伪装宣传结果。

## 2. 根因与设计约束

旧网模型是无状态的单向软约束：

1. 球越过斜背网接触平面后，只根据 x 方向穿透量和速度施力；
2. y/z 方向只有返回运动时的弱阻尼，没有网兜形变中心；
3. “网内最终点接近射门目标”曾参与通过条件，容易把球网物理和射门精度混为一谈。

本轮采用三个约束：

- **目标独立**：网兜锚点只能来自首次物理接触，不能读取或趋近 `target_y/z`；
- **物理职责单一**：射门精度只在球门平面交点计算，球网只负责耗散和留球；
- **fail-closed 可审计**：完整通过要求一次捕获、最终锚点位移不大于 `0.25 m`，并继续保留门线精度、真实死角、稳定性和动作权威等原有严格门槛。

## 3. 实现

### 3.1 可复用球网状态

`G1CompliantGoalNetState` 保存：

- 是否已接触以及首次接触三维锚点；
- 捕获次数；
- 峰值三轴合力和峰值锚点形变；
- 当前物理子步施力。

首次越过斜背网接触平面时只绑定锚点，不在事件边界制造人工冲量。后续 2 ms MuJoCo 子步围绕该锚点计算三轴弹簧阻尼，并把每轴限制在 `[-250, 250] N`。如果球从球门前方退出，状态会复位；再次进入会增加捕获计数，使严格通过门拒绝异常反复穿网。

不传入状态的调用仍保持旧无状态行为，避免破坏已有第三方调用；单人任意球与多 G1 传射/守门流程则统一复用新状态模型。

### 3.2 评分与遥测

result/evidence schema 升级到 v30，新增：

- `goal_net_anchor_xyz_m`；
- `goal_net_final_anchor_error_m`；
- `goal_net_peak_force_n`；
- `goal_net_peak_anchor_displacement_m`；
- `goal_net_engagement_count`。

逐帧轨迹新增 `goal_net_force_world`、`goal_net_engaged` 和 `goal_net_anchor_xyz`。非有限值、维度错误或“已接触但没有锚点”会立即失败，不生成貌似有效的证据。

## 4. 同版本证据重建

最终 24 条证据均绑定实现哈希：

`sha256:edff0e8e0b205708bfb03c125c28890a6c6da7f32c5d64be0e7594d4a941fef8`

证据目录：

- `/code/rosclaw/rosclaw_football/evidence/age10-goal-plane-v3-net-seed0-v1`：8 条；
- `/code/rosclaw/rosclaw_football/evidence/age10-goal-plane-v3-net-seed6-v1`：8 条；
- `/code/rosclaw/rosclaw_football/evidence/age10-goal-plane-v3-net-seed7-v1`：6 条；
- `/code/rosclaw/rosclaw_football/evidence/age10-goal-plane-v3-net-seed21-v1`：2 条。

四组编排映射到本地 A6000 0--3；物理真值仍来自 CPU MuJoCo，不声称 GPU 加速了动力学。汇总指标：

| 指标 | 最小值 | 中位数 | 最大值 |
|---|---:|---:|---:|
| 最终锚点位移 | 0.001 mm | 8.143 mm | 176.796 mm |
| 峰值球网合力 | 3.432 N | 90.200 N | 118.168 N |
| 峰值锚点形变 | 12.551 mm | 207.290 mm | 339.180 mm |

24 条均为一次捕获；严格重放 `24/24`，跌倒 `0/24`，触球后最大倒退 `0.0 m`。

## 5. Growth 闭环复验

新球网会改变严格 evidence 的实现哈希，因此没有复用旧 memory，而是从上述 24 条证据重新派生：

- memory：`/code/rosclaw/rosclaw_football/evidence/age10-episodic-goal-plane-net-memory-v1.json`
- 文件 SHA-256：`3018762394046d7e4d057a433b29d58be14337d4092bf83ed094caaeac8bee1f`
- memory hash：`sha256:02643a41fab009845b1b496dac406870f30a1ff657a5fd8807dd535dba225ebb`
- 安全/拒绝探针：`21 / 3`
- 支持原型：seed 0 AIRBORNE、seed 6 BOUNCE；拒绝上下文：seed 7、21。

留出闭环：

| 状态 | 目标 (y,z) m | 无记忆基线 | memory 候选 | 相对改善 | 网兜最终锚点位移 |
|---|---:|---:|---:|---:|---:|
| seed 0 AIRBORNE | (2.155305, 0.781930) | 42.497 mm | 15.279 mm | 64.05% | 176.796 mm |
| seed 6 BOUNCE | (1.293466, 0.121721) | 15.704 mm | 3.848 mm | 75.49% | 3.163 mm |

两个候选均严格重放、只在支持窗口输出一次动作、无跌倒、无后退。seed 7 稳定锚点没有获得非零学习力，保持 fail-closed。

评估 artifact：

- `/code/rosclaw/rosclaw_football/evidence/age10-episodic-goal-plane-net-evaluation-v1.json`
- 文件 SHA-256：`8c8f7e568482272c68d3745eca382b77d763f8b310e8d870335304ce94f33998`
- evaluation hash：`sha256:02358471970714918954221749b58ded37c9af5520957d62159d1b0b4f722a2a`
- 平均门线误差：`29.100 -> 9.564 mm`
- verdict：`DEVELOPMENT`，`promotion_authorized=false`。

## 6. 视频

- AIRBORNE：`/code/rosclaw/rosclaw_football/evidence/age10-net-heldout-video-v1/g1-airborne-net-pocket-1080p.mp4`
  - SHA-256：`a29922c64160182ee2ecf2cfb17d0e78f956543157012d721e90e2373c5da524`
- BOUNCE：`/code/rosclaw/rosclaw_football/evidence/age10-net-heldout-video-v1/g1-bounce-net-pocket-1080p.mp4`
  - SHA-256：`3783e705c77de05cf8f0caaa84267b42dcc74e590c5da0809e62167fb23f04f2`

两段均为 1920x1080、30 fps、20.9 秒。连续主镜头来自严格物理轨迹；慢动作只做插值展示，像素不参与评分。它们用于核查新网兜，不是可对外宣称“死角挑战已通过”的发布视频。

## 7. 验证

- `ruff check`：通过；
- 本轮 3 个源码文件 `mypy`：通过；
- 任意球与多 G1 耦合流程测试：`45 passed, 3 deselected`；
- 24 条训练证据和 5 条闭环/稳定性证据：全部严格重放；
- 两段视频经本地抽帧复查，球门保持开放几何，球进入斜背网后在网兜内耗散并受重力下沉，没有目标吸附。

## 8. 边界与下一步

本轮解决的是球网物理与学习证据的一致性，没有宣称已经获得全局球星策略：

1. seed 0/6 仍是两个局部状态岛；
2. 这两个留出目标不是标准球门左/右上死角，所以完整 showcase 正确地保持 rejected；
3. 高位死角不能靠继续放大附加力解决，需要主动学习脚背姿态、支撑腿、躯干倾角和触球时刻；
4. 下一轮应把 MotionDecode 动作先验转成接触姿态候选，通过权威约束主动采样扩展高球可达域，再以严格 replay 和稳定锚点选择 growth 候选；
5. 新网兜状态应继续扩展到传球、双脚、守门员和多 G1 统一场景，禁止各 demo 私自使用不同的球网动力学。
