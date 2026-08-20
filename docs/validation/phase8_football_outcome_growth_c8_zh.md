# Phase 8 C8：G1 必须射门的成功/失败成长闭环

日期：2026-08-07
范围：ROSClaw Growth + GoalForge + G1 MuJoCo，SIM_ONLY
结论：完成一次可复现的“行动—判分—反思—巩固—新种子再验证”闭环。恢复不再被计为足球成功；每个评估状态都必须选择并实际计分一次射门。

## 1. 这轮解决了什么

此前 readiness gate 可以在高风险状态选择 abstain，再执行身体恢复。这对通用机器人安全有价值，但不满足本任务“最终必须踢球”的目标。本轮把任务语义改为：

1. 足球成功必须有球接触，并按落点、身体安全和稳定性计分。
2. `RECOVER_AND_RETRY` 只是失败后的中间状态，不能成为足球终态。
3. outcome model 无论置信度高低都必须给出一个击球相位；低置信只产生重试建议，不产生终止弃权。
4. 新策略必须在未见种子上同时守住硬安全和执行器饱和，不能只靠更猛烈的动作换取命中率。

## 2. 工程实现

### 2.1 成功/失败 outcome memory

新增 `src/rosclaw/growth/football_outcome_model.py`：

- 输入为 SONIC 跑动结束时的六维本体状态：骨盆 yaw/roll/pitch、骨盆 x/y、关节速度 RMS。
- 训练数据不是只保留成功轨迹，而是对同一身体状态完整执行 P190、P205、P214 三种击球相位，保存成功与失败结果。
- 每个候选动作同时记忆：硬安全、0.16 m 精度命中、完整稳定合格、惩罚落点误差、执行器饱和、击球倾角和终态速度。
- 采用逐种子 leave-one-out 选择邻域和多目标权重，避免用当前种子的答案选择当前种子的动作。
- 决策始终返回一个相位；`retry_recommended` 不等于 abstain。

### 2.2 Stability–Plasticity 护栏

第一代模型证明了精度可学习，但也暴露出塑性挤压稳定裕量的问题。因此第二代加入硬性资格条件：

- 硬安全不得低于最佳固定相位；
- 精度至少增加 3 次；
- 平均惩罚落点误差至少下降 0.02 m；
- 交叉验证饱和分数不得高于固定相位。

只有同时满足四项的超参数才优先成为可接受模型。该约束写入 v2 artifact 的 objective 和 loader，篡改或缺失会 fail closed。

### 2.3 Sealed 反事实评估

新增 `src/rosclaw/growth/football_outcome_evaluation.py`：

- 开发种子与 holdout 种子强制不相交；
- 每个新状态必须具备三个相位的完整严格重放；
- Body、实验上下文、trajectory hash、implementation hash 均绑定；
- 每个种子必须选择并计分一次射门，`terminal_abstentions=0`；
- 同时比较固定基线、学习选择和反事实 oracle；
- v2 评估将饱和不回归作为 acceptance 条件。

### 2.4 接入实际连续射门运行时

`goalforge free-kick-showcase run` 新增 `--football-outcome-model`：

- 模型与旧 contextual/router 互斥；
- 校验 Body hash、model hash 和实验上下文；
- 在连续 MuJoCo 世界的 run-up handoff 读取真实本体状态；
- 模型选择 P190/P205/P214 后继续完成 bridge、击球、随动和恢复；
- 证据记录预测安全概率、预测命中概率、预测误差和 retry 建议；
- claims 明确 `football_success_requires_ball_contact=true`、`recovery_only_is_task_success=false`。

视频渲染器也新增 `SUCCESS-FAILURE OUTCOME MEMORY` / `LEARNED OUTCOME EXPERT` 真值标签，避免把 outcome 选择误写为 high-yaw router。

## 3. 第一代：先学会提高射门结果

开发记忆：种子 0–47，每个种子三个相位，共 144 个严格反事实 episode。

| 指标 | 固定 P190 | v1 leave-one-out | 变化 |
|---|---:|---:|---:|
| 硬安全 | 45/48 | 46/48 | +1 |
| 0.16 m 命中 | 18/48 | 26/48 | +8 |
| 平均惩罚误差 | 0.645 m | 0.521 m | -19.3% |
| 完整稳定合格 | 0/48 | 1/48 | +1 |

Artifact：`/code/rosclaw/phase8_evidence/g1-growth-c8-football-outcome-model-v1.json`
Model hash：`sha256:09252b4f0015f2772ad50956e3062da88f066f86b8560c70a72faaa1fea86a14`

### 3.1 未见种子 56–71

对 16 个全新状态生成 48 个 strict counterfactual episode：

| 指标 | 固定 P190 | v1 选择 | 结果 |
|---|---:|---:|---|
| 必须射门 | 16 | 16 | 0 次终止弃权 |
| 硬安全 | 15/16 | 15/16 | 无回归 |
| 0.16 m 命中 | 6/16 | 10/16 | +4 |
| 平均惩罚误差 | 0.674 m | 0.205 m | -69.6% |
| 饱和步数 | 245 | 369 | **恶化** |

旧的精度合同会接受这个结果；加入 Stability–Plasticity 审计后，v1 被明确拒绝，failure code 为 `SEALED_SATURATION_REGRESSION`。这次拒绝很重要：系统没有把“更准但更伤身体”包装成全面成长。

封存目录：`/code/rosclaw/phase8_evidence/g1-growth-c8-football-outcome-sealed-v1`
稳定性审计 hash：`sha256:e15cbef772aff29aeae74099e42f6cc0e678ebe4981adfa372b679ad77b7c51f`

## 4. 第二代：吸收失败并守住稳定性

把种子 56–71 的成功和失败全部巩固到记忆，形成 64 个身体状态、192 个反事实 episode。第二代不是只记成功，而是显式记住哪些相位会射失、饱和或失稳。

| 指标 | 固定 P190 | v2 leave-one-out | 变化 |
|---|---:|---:|---:|
| 硬安全 | 60/64 | 61/64 | +1 |
| 0.16 m 命中 | 24/64 | 27/64 | +3 |
| 平均惩罚误差 | 0.652 m | 0.508 m | -22.1% |
| 完整稳定合格 | 0/64 | 1/64 | +1 |
| 平均饱和分数 | 0.06914 | 0.06891 | -0.3% |

Artifact：`/code/rosclaw/phase8_evidence/g1-growth-c8-football-outcome-model-v2-stable.json`
Model hash：`sha256:996635d8fc08fb1cbaf5f68963b6f636ee34ddba025b2361ea77c5f38bc45277`

### 4.1 第二次未见种子 72–79

这些种子未参与 v2 训练，共执行 24 个 strict counterfactual episode：

| 指标 | 固定 P190 | v2 选择 | 变化 |
|---|---:|---:|---:|
| 必须射门 | 8 | 8 | 0 次终止弃权 |
| 硬安全 | 8/8 | 8/8 | 无回归 |
| 0.16 m 命中 | 2/8 | 3/8 | +1 |
| 平均惩罚误差 | 0.511 m | 0.293 m | -42.7% |
| 饱和步数 | 84 | 73 | -13.1% |
| 完整稳定合格 | 0/8 | 1/8 | +1 |

评估被 v2 合同接受，failure codes 为空。
封存目录：`/code/rosclaw/phase8_evidence/g1-growth-c8-football-outcome-v2-sealed-v1`
Report hash：`sha256:a8812349702f661e160ba566a7ed2e37bc4fa3529ad8d9acba4c63001250abef`

## 5. 直观案例：seed 76 从射失变为命中

相同 SONIC 本体状态、球门、物理参数和种子下：

| 方案 | 相位 | 结果 | 球门平面误差 | 饱和步数 |
|---|---:|---|---:|---:|
| 固定基线 | P190 | 未过球门平面 | 2.000 m 惩罚 | 10 |
| v2 outcome memory | P205 | 命中 | 0.026 m | 10 |

模型直控证据：`/code/rosclaw/phase8_evidence/g1-growth-c8-outcome-v2-runtime-seed76/g1-free-kick.json`
视频：`/code/rosclaw/phase8_evidence/g1-growth-c8-outcome-v2-runtime-seed76/g1-growth-outcome-v2-seed76.mp4`

视频是 21 秒、1280×720、30 fps 的 strict physics replay；它是 visualization-only / development evidence。该 episode 仍有 10 个饱和步、run-up 最低骨盆高度 0.659 m、kick peak tilt 0.412 rad，因此没有伪装成 promotion evidence。

## 6. 通俗解释

可以把三个击球相位理解成同一个球员的三种出脚时机。以前无论跑到球前时身体是偏左、偏右、快还是慢，都按同一个时机踢；有时正好，有时射偏，甚至更晃。

现在 ROSClaw 会记住：过去在“类似身体状态”下，三种时机各自踢到了哪里、身体有没有危险、关节是否过度用力。下次再遇到相似状态，它不是背诵一个固定补丁，而是查阅成功和失败记忆，选更合适的时机。第一代变准但用力过猛，系统把这个问题判成失败；第二代重新学习后，在新题上既更准，又减少了饱和。这就是本轮可验证的“成长”。

## 7. 验证与边界

已通过：

- `ruff check src tests`
- `mypy src/rosclaw`（1168 个 source files）
- outcome model/evaluator 与 free-kick/video 定向测试：`21 passed, 2 deselected`
- 72 个 strict counterfactual episode（两轮 sealed holdout）均绑定轨迹 hash 和实现 hash
- 模型直控 seed 76 两次仿真轨迹 digest 一致

全仓 `pytest` 还不能标记为全绿：运行到发布安装门禁时，宿主机
`/usr/bin/python3.10` 因缺少需要 sudo 安装的 `python3.10-venv` 而无法创建
虚拟环境；当时结果为 `1 failed, 317 passed, 15 skipped, 36 deselected`，后续慢速
构建尚未完成即停止。为区分环境故障与产品故障，使用项目现有、带
`venv/ensurepip` 的 Python 3.12 对同一个 237 MB 签名离线包复验：6168 个文件
验签/哈希检查通过、离线安装和健康检查通过，ROSClaw Agent 在 PTY 中显示品牌
并以 `/quit` 返回 0。故该失败归因于主机系统依赖，不能冒充“全量 pytest 通过”，
也没有据此修改发布安全代码。

仍未解决：

1. 这不是端到端神经小脑，也不是持续更新 actor-critic；它是数据驱动的局部 outcome memory / skill selector。
2. v2 改善的是群体统计，单次击球仍可能有饱和、倾角偏高、跑动最低骨盆高度不足。
3. `RECOVER → REASSESS → KICK` 已完成两个低置信状态的成对开发验证，但还没有足够多的 sealed 低置信种子，不能外推为普遍有效。
4. 三个离散相位仍限制动作上限。后续应学习连续 phase/bridge residual，再进入受约束 actor-critic，而不是直接放开关节力矩在线探索。
5. 当前只有 MuJoCo SIM_ONLY 证据，未授权真实机器人。

## 8. 下一步闭环

1. 将 P190/P205/P214 扩展为连续相位与 bridge 参数的 bounded residual，先离线 conservative RL，再 sealed 验证。
2. 将执行器饱和、支撑多边形、ZMP/CoM 轨迹和落脚质量纳入 critic，直接优化“进球后仍像球员一样站稳”。
3. 将当前连续世界 `RECOVER → REASSESS → KICK` 扩展为必要时主动重新落脚/重新接近，并积累至少 8 个 sealed 低置信种子。
4. 在多目标/高球门角、来球初速和左右脚条件下构建同样的完整反事实 outcome memory。
5. 只有经过 no-regression gate 的新模型才能替换上一代；未通过的模型和视频保留为 rejected evidence，防止灾难性遗忘。

## 9. 追加突破：连续恢复后仍必须射门

本轮后续又把 retry 从“一个标志”推进到物理动作：当初次 outcome decision 低置信时，SONIC 在同一个 MuJoCo 世界继续神经反馈，短时卸掉平移和关节速度；系统随后重新读取本体状态、重新选择击球相位，并继续完成真实球接触和射门。全程无 teleport、无 state reset。

对恢复时长 0.6/0.8/1.0/1.2/1.5 s 和随动增益 0.70/0.80/0.85/0.90/0.95 做了扫描。0.8 s + 0.80 gain 是当前两个状态的 Pareto 候选；0.70 gain 会落入低反馈失稳区，出现数百个饱和步和超过 2.6 rad 的倾角，因此被拒绝。

同实现 hash `sha256:a231b54016e7bc652b231f41538197aa67605dcd551c0b6190ea7291d0570e7a` 下的严格重放对照：

| seed | 方案 | 相位 | 结果 | 惩罚误差 | 交接速度 | 关节速度 RMS | kick tilt | 饱和步数 |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 65 | 无恢复 | P214 | 射偏 | 0.840 m | 0.399 m/s | 1.103 rad/s | 0.309 rad | 14 |
| 65 | 0.8 s 恢复 + 0.80 gain | P190 | 命中 | 0.011 m | 0.069 m/s | 0.217 rad/s | 0.264 rad | 7 |
| 94（未见） | 无恢复 | P190 | 未过门线 | 2.000 m | 0.530 m/s | 0.680 rad/s | 0.411 rad | 1 |
| 94（未见） | 0.8 s 恢复 + 0.80 gain | P190 | 命中 | 0.137 m | 0.077 m/s | 0.298 rad/s | 0.388 rad | 0 |

这两组都不是“恢复后结束”，而是恢复后完成球接触、球门判分、随动和终态测量。seed 94 是在 80–95 共 16 个新状态的 retry scout 中唯一被 v2 标记为低置信的状态，因此没有挑选多个结果后只展示最好的一个。

候选目录：`/code/rosclaw/phase8_evidence/g1-growth-c8-retry-followthrough-sweep-v1`
视频：`/code/rosclaw/phase8_evidence/g1-growth-c8-retry-followthrough-sweep-v1/seed94-g0p8/g1-continuous-recover-reassess-kick.mp4`
视频 hash：`sha256:e3bf19dd340992c748448cb48a24315d09409d92afc39b7bef28065b0b35d7ad`

边界仍然明确：只有 2 个低置信状态，其中 1 个是开发记忆中的 seed 65，另 1 个是未见 seed 94；因此这是“物理闭环已跑通”的开发证据，不是统计充分的 promotion evidence。
