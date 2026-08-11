# Phase 8：状态路由球门平面世界模型 v2 实施与验证报告

日期：2026-08-11
证据上限：`SIM_ONLY`（不授权真实机器人、在线热切换或产品晋升）

## 1. 结论先行

本轮把此前的“附加力 → 瞬时出球速度 → 理想单段抛体”改造成了“附加力 → 球门平面落点与到达时间”的局部世界模型。后者直接从严格 MuJoCo 回放学习球在触球后经历的飞行、落地、反弹和滚动结果，不再假设球只在空中飞一次。

最终自动评估结论为 `DEVELOPMENT`：

- seed 0（空中球）held-out 落点误差从 `0.042497 m` 降至 `0.015279 m`，降低 `64.05%`；
- seed 6（落地反弹球）held-out 落点误差从 `0.015704 m` 降至 `0.003848 m`，降低 `75.49%`；
- 两个候选都仅在一个控制帧内产生动作，动作子步的状态路由距离均为 `0.0 < 0.5`；
- 两个候选的最小任务权威均为 `1.0`，后退距离均为 `0.0 m`，无跌倒、关节越界、力矩越界或执行器饱和；
- seed 7 是已知安全但二维不可控的滚动状态，运行时激活 `0` 帧、拒绝 `2` 帧，保持原动作；
- 两个支持原型全部覆盖，稳定锚点保留，评估器退出码为 `0`。

这证明的是“两个已排练局部状态岛内的精度学习”，不是全局球星策略，也不是实机安全证明。

## 2. 为什么上一版会卡住

上一版先预测触球后的横向/竖直初速度，再用单段抛体公式反算目标。离线复盘发现：

- seed 0 的球从触球到过门线全程离地，可近似为空中球；
- seed 4 在过门前有约 `16%` 的采样处于地面接触区；
- seed 6 约有 `9.8%` 的采样处于地面接触区；
- seed 7 约有 `97.4%` 的采样贴地，属于长距离滚动。

因此 seed 4/6/7 的终点不能由单段抛物线可靠描述。原控制器安全拒绝这些目标是正确行为，但它也无法利用反弹和滚动结果学习。

宽动作域实验进一步给出了真实边界：

- seed 0 在竖直附加力 `150 N` 时仍保持权威 `1.0`，`250 N` 时降至 `0.785`，必须拒绝；
- seed 4 在 `100 N` 时权威已降至 `0.926`，更大动作不可作为安全训练标签；
- seed 7 在高力下仍几乎贴地，二维响应条件数约 `4273`，不能把一个不可控方向硬反演成 actor。

这说明“继续增大 RL 动作”不会自然得到高球，反而会先突破动作权威边界。

## 3. v2 架构

### 3.1 因果状态

每个局部原型使用真实施力前 `2 ms` 子步采集的 11 维状态：脚踝相对球位置、脚踝线速度、骨盆线速度、躯干 roll/pitch。planner seed 只作为审计标签，运行时不能参与路由。

### 3.2 世界模型

每个合格状态岛拟合：

`[1, lateral_force, vertical_force] -> [goal_y, goal_z, arrival_time]`

模型同时保存：

- 训练中实际出现的安全力包络；
- 安全落点构成的二维凸包；
- 空中、反弹或滚动接触类型；
- 落点 RMSE、到达时间 RMSE 和允许的最大预测残差；
- 每条训练 evidence 的内容哈希。

### 3.3 Stability–Plasticity 门

控制器只有同时满足以下条件才输出直接关节力矩：

1. 当前 11 维状态位于某个原型的归一化半径 `0.5` 内；
2. 请求目标位于该原型实测安全落点凸包内；
3. 反演动作位于实测安全横向/竖直力包络内；
4. 正向回代后的目标误差不超过该原型的模型阈值；
5. 脚与球足够接近、尚未触球、且处于绑定的策略时间窗；
6. 最终附加力矩仍经过既有任务权威投影和关节边界保护。

任一条件不满足都输出全零附加力矩。artifact 固定为 `SIM_ONLY`，`promotion_authorized=false`、`hardware_authorized=false`、`online_hot_swap_allowed=false`。

## 4. 最终训练证据

最终使用单一实现哈希 `sha256:0963617f6443b4f7f145efda8fdce9b0ec677c2b22394fd40513daa712de7167` 的 24 条严格双回放：

- `/code/rosclaw/rosclaw_football/evidence/age10-goal-plane-v2-seed0-release-v1`：8 条；
- `/code/rosclaw/rosclaw_football/evidence/age10-goal-plane-v2-seed6-release-v1`：8 条；
- `/code/rosclaw/rosclaw_football/evidence/age10-goal-plane-v2-seed7-release-v1`：6 条；
- `/code/rosclaw/rosclaw_football/evidence/age10-goal-plane-v2-seed21-release-v1`：2 条。

四路工作负载分别映射到 A6000 0–3。最终物理标签来自 CPU MuJoCo 严格双回放；GPU 映射用于隔离和并行编排，不宣称 GPU 加速了 MuJoCo 物理真值。

记忆 artifact：

- 路径：`/code/rosclaw/rosclaw_football/evidence/age10-episodic-goal-plane-memory-release-v1.json`
- 文件 SHA-256：`sha256:1cea8e460018863517e92847e7464ffdf5600b91bc8f868a56a8e55ddf19e1b9`
- memory hash：`sha256:023a186561ed91015e87d3a9074558babac599876f151552f2a017decd1b09e7`
- 安全/拒绝探针：`21 / 3`
- 支持原型：seed 0、seed 6；拒绝上下文：seed 7、seed 21
- 原型最小距离：`1.381987`，大于路由半径 `0.5`

局部模型：

| 状态岛 | 接触类型 | 落点 RMSE | 到达时间 RMSE | 安全力包络 Fy/Fz |
|---|---:|---:|---:|---:|
| seed 0 | AIRBORNE | 5.518 mm | 约 0 s | 0–30 N / 0–150 N |
| seed 6 | BOUNCE | 1.833 mm | 3.599 ms | 0–30 N / 0–100 N |

## 5. Held-out 闭环

两个测试目标从模型安全凸包内部插值得到，但从未作为训练任务目标出现。每个目标分别运行无记忆基线和带记忆候选，并对每次运行做严格重放。

| 状态岛 | 目标 (y,z) m | 基线误差 | 候选误差 | 绝对改善 | 相对改善 | 动作 |
|---|---:|---:|---:|---:|---:|---:|
| seed 0 | (2.155305, 0.781930) | 42.497 mm | 15.279 mm | 27.218 mm | 64.05% | Fy 15 N, Fz 80 N |
| seed 6 | (1.293466, 0.121721) | 15.704 mm | 3.848 mm | 11.855 mm | 75.49% | Fy 10 N, Fz 80 N |

两次候选的最小任务权威均为 `1.0`，动作时状态距离均为 `0.0`，后退均为 `0.0 m`。控制帧后续子步离开状态支持域时，控制器立即输出零附加力矩；新增的 active-only 距离字段使这一点可以直接审计。

稳定锚点 seed 7：状态到最近原型的最大评估距离 `1.0021 > 0.5`，激活 `0` 帧、拒绝 `2` 帧，权威 `1.0`，零后退且无安全事件。

评估 artifact：

- 路径：`/code/rosclaw/rosclaw_football/evidence/age10-episodic-goal-plane-evaluation-release-v1.json`
- 文件 SHA-256：`sha256:99b18e899edaf88083b470dee1c8ef8483bd5832d5548eb0d56cdda9094b8695`
- evaluation hash：`sha256:8e5801749bb0aa27b2505cd27ff66f1877f92267e3b0612eccc4821f0057fdfe`
- 平均误差：`29.100 mm -> 9.564 mm`
- 平均绝对改善：`19.537 mm`
- 支持状态覆盖：完整；安全候选：`2/2`；改善候选：`2/2`
- verdict：`DEVELOPMENT`

## 6. 代码变化

- `growth/episodic_contact_memory.py`
  - schema v2；
  - 二维凸包构建与目标域判断；
  - 飞行/反弹/滚动结果提取；
  - 力到球门平面三输出世界模型；
  - 状态、目标、力和预测残差四重 fail-closed；
  - 通过足端 Jacobian 直接输出 29 关节附加力矩。
- `simforge/g1_free_kick_showcase.py`
  - flow/result/evidence/request schema 升级；
  - 新增目标支持、预测落点/时间和 active-only 路由距离轨迹；
  - 将世界模型动作继续置于权威投影与边界保护之后；
  - 严格重放绑定新轨迹。
- `growth/episodic_contact_evaluation.py`
  - v2 评估 schema；
  - 使用 active-only 路由距离，避免把同一控制帧内后续拒绝子步误写成越域激活。
- CLI 与测试
  - 保持 `growth episodic-contact-memory` 和 `--episodic-contact-memory` 产品入口；
  - 更新帮助文案；
  - 增加凸包内激活、目标 OOD 零力矩、状态 OOD 零力矩、artifact round-trip 和不可晋升测试。

## 7. 代码验证

- 本轮聚焦测试：`33 passed, 2 deselected`；
- 本轮 5 个改动源码文件的 `mypy`：`Success: no issues found`；
- 本轮 6 个改动 Python 文件的 `ruff check` 与 `ruff format --check`：通过；
- `compileall` 与 `git diff --check`：通过；
- 仓库全量非慢速测试：`6842 passed, 114 skipped, 19 deselected, 6 failed`。6 项失败均为既有环境假设：2 项假定系统找不到 Codex CLI，4 项在收集阶段发现用户目录中的 LeRobot 配置、执行阶段却切换到隔离目录。隐藏已安装的 Codex CLI 后前 2 项 `2 passed`；显式指定本机 LeRobot 0.6.1 隔离解释器后后 4 项 `4 passed`。

仓库级 `mypy src` 仍有 `145` 个既有文件、`655` 项类型错误，本轮改动文件为零错误；为避免把目标扩张成全仓类型债务治理，这些错误未在本轮顺带修改。

## 8. 证据审计说明

开发中曾出现两次“代码功能正确但证据哈希/遥测表达不够严格”的情况：一次是采集中途静态类型修正改变实现哈希，另一次是帧级聚合混合了激活子步和随后拒绝子步。派生器拒绝了混合哈希；最终报告只使用上述 `release-v1` 单一哈希数据。早期 evidence 保留在外部目录用于审计，但不进入最终 artifact。

## 9. 仍未解决的问题

- 这是两个局部状态岛，不是任意跑姿、左右脚、移动球和对抗条件下的全局策略。
- seed 0 的安全高度覆盖目前到约 `0.808 m`，远未覆盖标准球门左/右上角；单纯增大附加力会先损失动作权威。
- seed 6 的毫米级结果是低位反弹目标，证明了接触结果建模有效，但视觉宣传价值有限。
- 当前 actor 是一次性局部任务力，不是端到端神经网络小脑，也不是持续在线更新的 actor-critic。
- 所有结果均为 MuJoCo 仿真，不能外推到真实 G1。

## 10. 下一步

1. 用权威约束的主动采样扩展更多 SONIC 接近状态，优先寻找能安全产生高出球角的接触姿态，而不是盲目增加力；
2. 将 AIRBORNE/BOUNCE/ROLLING 作为 mixture-of-experts 接触模式，分别学习局部动力学；
3. 把 MotionDecode/足球动作先验用于改变脚背姿态、支撑腿和躯干协同，扩大高球可达域；
4. 在世界模型内做保守 actor-critic 更新，候选只进入 SHADOW/严格回放，达到新状态支持和稳定性门后再固化；
5. 对左脚、移动球、传球来球、守门员干扰分别建立独立 evidence profile，禁止跨 profile 宣称泛化。
