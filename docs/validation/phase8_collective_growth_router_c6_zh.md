# Phase 8 C6：Collective 足球动作先验与本体三专家 Growth 闭环

日期：2026-08-07
边界：`SIM_ONLY / CPU_MUJOCO / 4×A6000 REPRESENTATION TRAINING`。没有真实机器人、ROS/DDS 或电机命令。

## 1. 结论先行

本轮打通了两条之前断开的链：

1. **众生经验链**：真实盘点 MotionDecode/OmniContact/GEAR-SONIC 等本地数据，许可证内容寻址，MotionDecode 足球动作分层抽样、运动学修复、足底相位推断、MuJoCo 物理资格门、四卡本体动态先验训练；
2. **自我成长链**：从连续 SONIC 助跑结束时的真实本体状态学习 phase 190/205/214 三专家路由，在从未用于拟合的 seeds 16–31 上与固定 phase 214 做严格成对验证，并由独立评估器决定是否接受。

结果不是“完美晋升”，而是一次可测但被安全门拒绝的成长：

| sealed holdout，16 seeds | 固定 phase 214 | 本体三专家路由 | 变化 |
|---|---:|---:|---:|
| 平均惩罚落点误差 | 1.2811 m | 0.8708 m | **-32.0%** |
| 未过门线 | 10 | 6 | **-40.0%** |
| 0.16 m 精度圈命中 | 5 | 5 | 持平 |
| 不安全 episode | 5 | 3 | 改善，但未归零 |
| actuator saturation physics steps | 1199 | 324 | **-73.0%** |
| strict replay | 16/16 | 16/16 | 一致 |

评估器因此记录：

```text
measurable_improvement = true
accepted = false
failure_codes = [ROUTED_UNSAFE_EPISODE]
```

这正是 Stability–Plasticity Dilemma 的工程化处理：允许新能力被测量和保留为 development memory，但只要独立保留集仍有不安全 episode，就不覆盖稳定技能、不进入 Promotion。

## 2. Dataset Doctor：先判断“能不能用”，再谈训练

### 2.1 本地快照

审计根目录：`/code/rosclaw/rosclaw_football/datasets`。`dataset_inventory.json` 的语义报告承诺：

`sha256:11981af87673d0021395aad207e08fa86903a5f18db46f2e39f39132948e3f71`

外层 JSON 文件 SHA-256：`6203495d0234bfe75f39e079b3935266944d7a546906c135f76d83cf019b7903`。前者排除自身 commitment 字段并用于协议校验，后者用于逐字节文件校验。

关键盘点：

| 数据集 | 文件数 | 大小 | 足球匹配 | 主要问题 |
|---|---:|---:|---:|---|
| GEAR-SONIC | 134 | 34.054 GiB | 0 | 分卷归档已识别为有意 multipart，不再误报下载残片 |
| MOSAIC | 10,582 | 42.740 GiB | 0 | 本地无许可证文件 |
| MotionDecode | 209,410 | 156.273 GiB | 1,204 | 空 `LICENSE`，有效条款在 `LICENSE.md`；快照仍按传输中处理 |
| OmniContact | 9,100 | 26.396 GiB | 1,045 | 一个零字节可视化脚本；本地无许可证文件 |
| g1-retargeted-motions | 354 | 0.229 GiB | 0 | 本地无许可证文件 |

Dataset Doctor 的修复：

- 分卷文件 `.tar.part_aa/.part_ab/...` 不再误判为未完成下载；
- metadata 模式下，许可证仍强制做内容 SHA-256，而不是只记 mtime/size；
- `football_match_count` 记录真实总数，另记 retained sample 是否截断；
- 非操作员声明的快照只要出现零字节文件就保持 PARTIAL；
- HTML/CSV/JSON 使用相同真实计数；
- schema 升级并保留向后加载。

用户说明数据仍在传输，因此这份 inventory 是**传输中快照**，不能外推为完整上游清单。

## 3. MotionDecode：有帮助，但不是可执行足球策略

### 3.1 来源与许可绑定

本地 Hugging Face metadata 给出的固定 revision：

`f71451a3e3285e83f11fe8738fc1d4750cab84f2`

研究非商业条款快照：

`sha256:ab46dc8e0994ea94fd6f66c0ec0cd33fa3d309702297cc5e220eda58bfc1d60e`

登记哈希：

`sha256:a4d5c82a2ae1ffceff8a46b17b5ee39f0e08689738fa4af42680851cbc710e16`

原 source registration 直接按路径排序截断，64 条全部来自 Short Pass。这会把“足球先验”偷换成“短传先验”。本轮新增确定性两层 round-robin：先动作族、再叶子技能。新的 64 条 pilot 为：

- Short Pass 13；
- Long Pass 13；
- Shooting 13；
- Ball Control 13；
- Others 12。

### 3.2 数据资格结果

| 阶段 | 结果 |
|---|---|
| 原始运动学 Q1 | 3/64 |
| 受约束尾段 reset 修复后 Q1 | 55/64 |
| 被拒绝 | 9/64（含 ambiguous reset、无合法 terminal reset、joint limit） |
| 足底/支撑相位候选 | 24/64 |
| CPU MuJoCo 物理步 | 147,205 |
| Q3 physics-trackable | **0/64** |

大量 CSV 在最后拼接“回到初始姿态”的 loop reset。ROSClaw 只在内存中重放唯一、位于尾窗、可证明的 trim；不修改原始 CSV，也不把 repair 当成训练许可。

24 条候选虽然进入真实 MuJoCo step，但在当前 G1 position-reference tracker 下出现跌倒、非脚触地、root tracking error、support slip 或 torque saturation，因此全部停在 Q1。原 CSV 也没有同步球位姿、接触、action、reward、transition 或 torque，不能包装成 offline RL 足球数据。

### 3.3 四卡动作表征先验

55 个 Q1 episode 构成只读表征包：

- 61 维：29 joint position + 29 joint velocity + projected gravity；
- 5,084 training windows；
- 1,423 validation windows；
- split 按 source episode 隔离；
- 52 个 episode 来自可重放尾段修复；
- 原始 motion 不导出，action/reward semantics 明确为 ABSENT。

4 张 A6000 各跑独立 seed，4/4 worker 成功。最佳 GPU 1 / seed 8601：

| 指标 | 值 |
|---|---:|
| persistence baseline loss | 0.0230741 |
| validation loss | 0.0183574 |
| improvement | **20.44%** |
| 2% quality gate | PASS |

决定是 `REPRESENTATION_CANDIDATE`，永久字段仍为：

```text
action_semantics = ABSENT
activation_ceiling = SIM_ONLY_REPRESENTATION_INITIALIZATION
promotion_evidence_eligible = false
hardware_authorized = false
```

所以 MotionDecode 的真实帮助是“更会预测足球动作里的下一步本体变化”，不是已经学会踢球，更不是可直接输出力矩的小脑。

## 4. 从固定补丁到本体三专家

### 4.1 输入不是 seed，而是身体状态

助跑结束、速度匹配桥开始前读取：

1. `abs pelvis yaw`；
2. `abs pelvis roll`；
3. `abs pelvis pitch`；
4. pelvis x；
5. pelvis y；
6. 29 关节速度 RMS。

每个 seed 对 phase 190/205/214 都执行同一世界、同一参数的严格反事实。特征从 `.npz` 中最后一个 SONIC approach state 重算，三个专家的 handoff 特征必须逐元素一致，否则拒绝标定。

模型是可审计的 robust centroid router：

- development 中位数/IQR 归一化；
- 每个赢家专家一个 centroid；
- 最近与次近中心 margin 小于 0.05 时不信任；
- 最近中心距离大于 2.5 时视为 OOD；
- 不确定/OOD 退回 development 中 16/16 安全的 phase 190；
- 不输出 torque，只选择受限 motion expert。

### 4.2 Development 留一 seed 门

seeds 0–15 的 leave-one-seed-out：

| 指标 | 固定 214 | 路由 |
|---|---:|---:|
| 平均惩罚误差 | 0.4750 m | 0.2759 m |
| 精度命中 | 5/16 | 9/16 |
| 不安全 episode | 4 | 0 |

Artifact 的语义 `router_hash`：

`sha256:8a6b914ca2f00133b616ae10392e1cc5f62f0455dad9200ccc7b322d75aa3b96`

外层 artifact 文件 SHA-256：`f9e77b924940999caed220e2887a194781fc1b1906b7c2a9e3e8a7732b0c2405`。

运行时 loader 会重算 artifact hash，检查 Body、安全边界、development acceptance，并拒绝与 legacy yaw router 同时启用。

### 4.3 Sealed holdout

artifact 冻结后才运行 seeds 16–31。路由相位分布：

- phase 190：5；
- phase 205：9；
- phase 214：2；
- OOD/low-confidence fallback：3。

评估器的语义 `report_hash` 为 `sha256:ba9c19e65829aa497cf8b46b6b70eff2c7d4717d95e9b2e077130b87e528dbf2`；外层评估 JSON 文件 SHA-256 为 `0faa9648552e605574b9e5aa1eeb37dfeb0936ae020404f69f110461ab469a6d`。

局部突破：

- seed 26，phase 205：门线目标误差 **0.001696 m**；
- seed 30，OOD fallback phase 190：门线目标误差 **0.013241 m**；
- seed 19：固定 214 未过门线，路由 205 达到 0.1654 m；
- seed 26：固定 214 不安全且未过门线，路由 205 安全并达到 1.7 mm。

真实失败：

- seed 16/20/29 相比固定 214 回归；
- seed 23/24 的三个专家都不安全，说明问题在助跑/交接 readiness，而不是再选一个 kick phase；
- seed 27 路由到 205 不安全，而反事实 phase 190 安全且 5.1 mm；
- run-up peak tilt 最差仍达 0.9454 rad，SONIC 小脑的 balance foundation 尚未合格。

因此下一代必须新增“无安全专家则拒绝射门并执行受验证刹停恢复”的 readiness/abstention expert；仅靠 OOD 时仍踢 phase 190 不够安全。

## 5. ROSClaw 新增/加固模块

- `rosclaw.dataset`：真实足球计数、截断语义、license content hash、multipart 识别、zero-byte fail closed；
- `rosclaw.collective.sources.motiondecode.manifest`：动作族 + 叶子技能的确定性分层 pilot；
- `rosclaw.growth.proprioceptive_expert_router`：轨迹/Body/实现/实验上下文绑定的三专家学习与 loader；
- `rosclaw.growth.proprioceptive_router_evaluation`：成对 sealed holdout 评估、可测成长与绝对安全分离；
- `g1_free_kick_showcase`：运行时真实本体特征、router hash、选择距离/margin/fallback 进入 strict evidence；
- CLI：`growth proprioceptive-expert-router`、`growth evaluate-proprioceptive-router`、`goalforge free-kick-showcase --proprioceptive-expert-router`。

所有新路径保持：

```text
promotion_truth_allowed = false
activation_authorized = false
hardware_authorized = false
```

## 6. 视频证据

42.30 秒连续双案例：

`/code/rosclaw/phase8_evidence/g1-growth-c6-router-holdout-v1/g1-growth-c6-two-breakthroughs.mp4`

SHA-256：

`35d974c50dde5328ff514f4694a9c32d40d6fe5e88d8bf56a39302faf6153ebf`

组成：

1. seed 26 / phase 205 / 1.7 mm；
2. seed 30 / OOD fallback phase 190 / 1.3 cm。

视频明确水印 `REJECTED CANDIDATE / DIAGNOSTIC ONLY / NOT PROMOTED`。像素不参与评分，数值来自对应 strict physics trajectory。

## 7. 回归验证与环境缺口

代码质量与本轮定向回归：

- `ruff check src tests`：PASS；
- `mypy src/rosclaw`：1160 个源码文件 PASS；
- Dataset Doctor、MotionDecode source、Growth router/evaluator、contextual/showcase 定向测试：64 passed，2 deselected；
- `git diff --check`：PASS。

完整仓库第一次执行结果为 `6431 passed, 89 skipped, 23 failed, 36 deselected`。失败均不在本轮修改模块：15 项来自测试虚拟环境缺少 Pillow，1 项缺少 `rank_bm25`，4 项来自 LeRobot collection-time 配置与隔离 HOME 不一致，2 项来自本机已安装 Codex CLI 而旧测试假定该二进制不存在，1 项是离线发行安装失败。

没有把这些失败隐藏为 skip。补装 Pillow/`rank_bm25` 后，相关复测为 `76 passed, 4 failed`；再显式复用现有 `/code/rosclaw/lerobot-runtime/bin/python`（LeRobot 0.6.1，runtime ready）后，剩余 4 个真实 LeRobot export/dataloader 案例为 `4 passed`。因此 23 个原始失败中 20 个已实证消除，仍有 3 个仓库/机器基线问题：

1. `test_installed_artifact_pty_quit_clean`：离线安装 prefix 中 `.venv/bin/python3` 不存在；
2. 两个 external-pack 测试：用例要求 Codex CLI 缺失/T0，但本机实际存在 Codex CLI，分别得到版本门和 trusted-directory 错误。

这 3 项与 Dataset/Collective/Growth/SimForge 改动无交集，本轮不扩大范围篡改测试假设。它们仍是后续仓库加固项，因此不能宣称“全仓零失败”。

## 8. 下一阶段闭环

优先级按失败证据决定：

1. **Readiness/abstention expert**：预测“当前是否存在安全 kick expert”；无安全专家时不射门，执行受物理验证的短刹停/卸力恢复；
2. **SONIC balance residual actor-critic**：奖励以 run-up tilt、pelvis height、joint boundary、saturation、terminal speed 为主，先解决 seed 23/24 和 0.945 rad tilt；
3. **MotionDecode latent skill distillation**：用已训练的 61 维 prior 初始化低层 actor trunk，但只在 MuJoCo torque/safety critic 闭环中学习 residual authority；
4. **OmniContact adapter**：利用明确的足球接触轨迹补 MotionDecode 缺失的接触语义，但仍需本地 license snapshot 与 G1 retarget/physics gate；
5. **上角射门**：等独立高球 motion/contact expert 合格后再挑战 1.35 m 左/右上角，禁止用长时间外力 teacher 伪装能力；
6. **下一保留集**：新模型冻结后用 seeds 32+，要求 `unsafe=0`、precision hits 不回归、miss/饱和显著下降，才进入下一层 Growth memory。

“见众生”提供动作分布；“见自己”判断当前身体处于哪个吸引域以及是否应当出脚；“见天地”仍由独立 MuJoCo 世界决定成长是否真实。Dream/Collective 可以提出先验，Growth 可以学习路由，但清醒世界的安全门拥有最终否决权。

## 9. 证据索引

- Dataset Doctor：`/code/rosclaw/phase8_evidence/dataset-doctor-c6-transfer-v2/`
- MotionDecode Collective：`/code/rosclaw/phase8_evidence/motiondecode-c6-social-dream/`
- 三专家 artifact：`/code/rosclaw/phase8_evidence/g1-growth-c6-proprioceptive-router-v1.json`
- 16-seed paired evaluation：`/code/rosclaw/phase8_evidence/g1-growth-c6-router-holdout-v1/g1-proprioceptive-router-evaluation.json`
- 全部 routed/baseline/counterfactual trajectories：`/code/rosclaw/phase8_evidence/g1-growth-c6-router-holdout-v1/`

## 10. 公开参考的使用边界

- MotionDecode 官方 samples 树用于核对 `3.3.Ball_Game_Interaction` 的公开存在；实际资格结论只来自本地固定 revision 和哈希文件；
- OmniContact 官方 README 声明 200 条 soccer processed trajectories 和 NPZ/contact 字段，当前只用于下一阶段 adapter 设计，未作为本轮 G1 成果；
- NVLabs GR00T WholeBodyControl/SONIC 官方仓库提供运行/训练结构和已发布 checkpoint；ROSClaw 只在 SIM_ONLY adapter 内使用本地模型，未声称复现其大规模训练结论。
