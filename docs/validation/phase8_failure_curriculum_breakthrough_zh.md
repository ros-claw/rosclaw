# Phase 8 G1 失败课程、自感知执行门与 Growth 闭环报告

日期：2026-08-06
范围：Unitree G1 + RoboNaldo MuJoCo，纯仿真（`SIM_ONLY` / `SHADOW`），未连接 ROS、DDS 或真实机器人。

## 结论先行

本阶段打通了一个可审计的 ROSClaw 自进化闭环：轨迹质量审计 → 失败课程 → 上下文肌肉记忆 → 实时关节边界反射 → 预踢环境/本体辨识 → 风险执行门 → 密封 validation/holdout → Growth/agentd 知识沉淀。

第三代冻结候选在全新密封验证上获得：

- 2/2 允许执行的 validation 射门成功；总 validation 成功率 2/3。
- validation 执行覆盖率 2/3；危险场景的 abstention 不计成功。
- 0 次摔倒、0 次关节/力矩越界、0 次执行器饱和、0 次非有限状态。
- 3 个 validation 与 1 个 holdout 全部严格重放一致。
- 高风险 validation 和私有 holdout 均在运动前返回 `ROBOT_NOT_STABLE`，没有产生运动或力矩。
- 决策：`SIM_CANDIDATE`；这不是硬件部署许可。

证据报告：`/code/rosclaw/phase8_evidence/g1-failure-curriculum-v3/g1-failure-curriculum-report.json`
报告哈希：`sha256:b076639a5c6c3c16a09937585160b7455a11ab1913a4e026b0a1ad33319cb5ad`

## 为什么前两代失败

| 代次 | validation 结果 | holdout 结果 | 决策 | 学到的经验 |
|---|---|---|---|---|
| v1 | 0/2 成功；1 次后踢摔倒 | 1 次关节越界 | REJECTED | 单一 g7 成功过拟合扰动时序 |
| v2 | 1/3 成功；2 次关节越界 | 1 次关节越界 | REJECTED | 固定的摩擦/时延 belief 使机器人无法知道脚下有多滑 |
| v3 | 2/3 成功；危险例预踢拒绝 | 危险例预踢拒绝 | SIM_CANDIDATE | 先辨识自身与环境，再决定是否行动 |

v2 的一个失败样本已经把球踢入目标区，但机器人随后倒地。门控仍将它判为临界失败，避免把“球进、人倒”包装成成功。

## 关键开发内容

### 1. 非因果数据隔离

`anticipatory-dev-00-g4` 中没有踢球脚接触，但球速和高度异常增长。新审计器根据“无踢球脚接触 + 球速超过 16 m/s 或球高超过 2 m”将其标记为 `NONCAUSAL_BALL_MOTION_WITHOUT_KICK_CONTACT`。

该轨迹不进入 actor，也不进入风险 critic。这样 MotionDecode、RL 或离线训练不会学习模拟器碰撞伪影。

### 2. 上下文肌肉记忆

参数搜索没有继续追求更大摆腿，而是在低偏置区域形成低冲量技能：

- 摆腿幅度 0.85，速度比例 0.90；
- COM 向支撑侧预置到 -0.065 m；
- 站姿横移 -0.035 m；
- 恢复步长 0.11 m。

它在两个独立开发种子上从临界线附近提升到 2/2 成功，COM 最小裕量分别为 +0.013 m 和 +0.003 m。全新 v3 validation 的两个可执行样本也都成功，目标误差为 0.466 m 和 0.450 m，COM 最小裕量为 +0.008 m 和 +0.016 m。

### 3. 局部、速度感知的关节边界反射

新增的安全投影读取当前关节位置与速度，预测短时未来位置。只有受审计关节即将继续越过边界时，才修正父控制器力矩；其余时刻逐元素保持父控制器原输出。

默认保护左膝、左踝滚转和腰俯仰。它具有配置哈希、父策略哈希、投影次数、投影关节数、最大修正量及最大预测越界量收据，并硬限制为 `SIM_ONLY`。它不是第二个 actuator writer，也不是“学会了直接输出力矩”的神经 actor。

全关节动态扩展和下肢整链强保护也做过物理 A/B，但会与原策略对抗甚至恶化倒地，因此没有进入最终候选。

### 4. 预踢自感知与执行门

此前 `support_friction_belief` 与 `control_latency_belief_ms` 是常数，无法区分安全和危险地面。v3 增加 SIM 预踢辨识收据，输出：

- 摩擦估计、显式不确定度和保守下界；
- 控制时延估计、显式不确定度和保守上界；
- 机体零位校准偏差；
- `safe_to_execute` 与拒绝原因；
- 内容哈希与 `SIM_ONLY` 上限。

只有摩擦下界至少 0.74、时延上界不超过 15 ms、机体校准偏差不超过 0.023 rad 时才允许踢球。高风险场景返回 `ROBOT_NOT_STABLE`，不运行物理运动。validation 至少要有 2/3 实际执行，且总成功率至少 50%，所以不能靠全部 abstain 骗过门控。

### 5. Growth/agentd 自进化知识闭环

Growth 桥接器新增对失败课程 v1–v3 的严格支持。它会重新计算报告哈希、密封门控、执行覆盖率和 abstention 语义，而不是相信文件中的决策字段。

统一 agentd 数据库保存了：

- v3：`HOW` 候选 `lc_f9865bb488dc4feab0921e35`；
- v1：负面 `MEMORY` `lc_db3afe52d9b54232bfc4829e`；
- v2：负面 `MEMORY` `lc_b73b43f574ac4bf7a0f4e8a1`。

三条记录均为 `CANDIDATE / measured`，`deployable=false`，且 activation、promotion、hardware authorization 全为 false。数据库位于 `/code/rosclaw/phase8_evidence/g1-failure-curriculum-v3/agentd/growth.db`。

## ROSClaw 模块验证

- `simforge`：RoboNaldo G1 资产资格验证、MuJoCo 物理执行、轨迹记录和严格重放。
- `feedback/contracts`：配置、策略、报告和辨识收据的内容哈希。
- `GoalForge`：场景分区、隐藏 validation/holdout、成功和临界失败语义。
- `growth`：通过证据成为 HOW、拒绝证据成为 MEMORY，且不自动晋升。
- `agentd`：内容寻址候选、证据等级、幂等知识沉淀和禁止部署字段。
- ROSClaw CLI：`simforge validate ... --profile failure-curriculum` 与 `growth stage-agentd-evaluation` 均完成真实运行。

## 正式运行命令

```bash
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile failure-curriculum \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output /code/rosclaw/phase8_evidence/g1-failure-curriculum-v3
```

```bash
.venv/bin/python -m rosclaw.entrypoint growth stage-agentd-evaluation \
  --evaluation /code/rosclaw/phase8_evidence/g1-failure-curriculum-v3/g1-failure-curriculum-report.json \
  --agentd-db /code/rosclaw/phase8_evidence/g1-failure-curriculum-v3/agentd/growth.db \
  --receipt /code/rosclaw/phase8_evidence/g1-failure-curriculum-v3/agentd/bridge-receipt.json \
  --source-checkout /code/rosclaw/rosclaw_phase8_exploration
```

## 诚实边界与下一步

本阶段没有宣称已经训练出端到端神经小脑，也没有解决低摩擦、高时延、强扰动下仍能漂亮射门的问题。当前突破是：系统能识别自己何时没有能力安全踢球，并把成功、失败、伪数据和拒绝动作正确分流。

下一阶段应优先开发真正的实时支撑策略，而不是继续调静态踢球参数：

1. 用预踢辨识和在线本体状态训练接触相位 actor-critic，动作限于支撑腿踝/髋与恢复步残差；
2. 高风险 episode 保留为 critic 负样本，abstention 作为安全行为，不作为 actor 成功示范；
3. 在 0.70–0.78 摩擦边界做更密的分层 curriculum，目标是逐步降低 abstention、保持零临界失败；
4. 只有新一套密封 validation/holdout 再次通过后，才把接触策略并入现有 HOW 候选；
5. 视频应只渲染真实执行的成功样本，并同时展示 belief、COM、躯干角和关节保护触发，避免宣传视频隐藏安全边界。

## Moving-ball Gauntlet 可视化复验

在上述冻结候选之上又增加了三个只属于 development 的组合挑战，并为每一关执行严格重放：

| 关卡 | 新增难度 | 结果 | 安全结果 |
|---|---|---|---|
| Moving Ball Intercept | 来球 -0.10 m/s、横漂 +0.02 m/s、30 N 推力、7 ms 时延 | SUCCESS，误差 0.457 m | 未摔倒、无越界 |
| Fast Moving Ball | 来球 -0.18 m/s、横漂 -0.04 m/s、32 N 推力 | SUCCESS，误差 0.474 m | 未摔倒、无越界 |
| Friction-edge Combo | 移动球、摩擦 0.78、8 ms 时延、0.018 rad 零偏、32 N 推力 | SUCCESS，误差 0.469 m | 未摔倒、无越界 |

三关均为真实 CPU MuJoCo 物理轨迹，3/3 严格重放一致，关节边界反射分别投影 3、7、2 次。证据在
`/code/rosclaw/phase8_evidence/g1-self-aware-showcase-v1/g1-self-aware-showcase.json`。

47.1 秒、1280×720、30 fps 的视频位于
`/code/rosclaw/phase8_evidence/g1-self-aware-showcase-v1/g1-self-aware-moving-ball-gauntlet.mp4`，SHA-256 为
`2ee170bb906d766797f6cbf354955eb4197f53349d58c8ec75aba3847324bd15`。前三段展示三条严格物理重放；第四段把 v2 的“球进入目标区但机器人倒地”与 v3 的“危险条件下不发送运动命令”并排展示；最后一段是验证记分牌。

这条视频是证据下游的可视化，像素不回流到训练或晋升。三关是开发阶段挑选的 showcase，不冒充新的密封泛化结论，也没有使用或展示 holdout。0.75–0.90 m 高目标的探索暴露出当前技能仍以低平球为主；强行提高摆腿曾造成跌倒，因此该失败没有剪成成果，而是保留为下一轮抬脚接触策略和高球课程的开发目标。
