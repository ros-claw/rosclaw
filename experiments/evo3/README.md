# PR-EVO-3: Matched-Regime Real-Hardware Validation（数据库优化v4 §11–§12）

真机验证设计与分析机器。**当前状态：协议 + 决策机器 + 离线证明完成；
真机 A/B/C 战役待操作员排期执行**（每组 12 会话 × 100 回合 × 3 臂，
跨多天、左右手平衡 —— 属于多日硬件占用，不在代码提交内虚报完成）。

## 实验（`protocols.py`）

| 实验 | 目标 | 状态 |
|---|---|---|
| Exp1 健康工况拒绝干预 | C 组低覆盖、ABSTAIN 正确率高、Memory Hurt ≈ 0 | 协议就绪，待真机 |
| Exp2 匹配热退化工况 | 工况 crossover（3 选 2 条件）后随机入臂，C 组 Recovery SR 提升 | 协议就绪，待真机 |
| Exp3 反工况负测试 | 热退化记忆放入健康工况 → Retrieve=true, Applicable=false, ABSTAIN | **已通过（离线）** |
| Exp4 编舞保护 | run1 有害 Patch → 100% BLOCK，0 真机执行 | **已通过（离线）** |

安全不变量（v4 §17.12）：绝不主动超过已验证温度/电流/保护阈值；
实验二只响应自然退化（slope ≥ 0.15 °C/min、p95 误差 ≥ 15、invalid
≥ 6% 三选二触发），不主动升温。

## 运行

```bash
# 离线证明（无硬件）
.venv/bin/python experiments/evo3/run_experiment.py validate-offline

# 决策行为回放（真实遥测，无硬件；不测反事实 invalid rate）
.venv/bin/python experiments/evo3/run_experiment.py replay --sessions 6

# 真机战役计划（不驱动硬件）
.venv/bin/python experiments/evo3/run_experiment.py live --protocol evo3_exp1_healthy_abstain
```

## 回放结果（真实 7×24 会话，决策行为）

- 4 会话：COLD_HEALTHY ×1、TRACKING_DEGRADATION ×1、THERMAL_DRIFT ×2
  （工况模型直接读真实会话，缺失特征诚实降置信）。
- 28 个失败样本：A 干预 0/28、B 干预 28/28、**C 干预 0/28（ABSTAIN
  100%）** —— 无 validated 证据时选择性管线一律停手，正是健康工况
  Memory Hurt ≈ 0 的决策形态。

## 统计（`stats_analysis.py`，v4 §12）

- 只以 **Session 为推断单元**（§12.1）：Round/Session/Seed/Day/Hand/
  Gesture 多层输出，绝不把 round 总数当独立样本（§17.9）。
- 晋升报告（§12.5）：effect size（Cohen's d，零方差时如实 None）、
  配对 bootstrap 95% CI、McNemar 精确检验、Kaplan-Meier + RMST
  生存增益、Session 分布、unsafe action 计数。
- 混合效应模型（§12.2）：statsmodels 可用时
  `invalid ~ arm + temperature + slope + (1|day)`；不可用时给出诚实
  fallback 说明，绝不拿 OLS 冒充混合模型。
- 4%→2% 这类幅度永不单独作为"统计显著"结论（§17.10）。

## 真机战役接入点

工作区 stress harness 逐会话产出 practice runs → 提取 rounds 为
`SessionRecord` → `promotion_report(records, arm_a="A_no_memory",
arm_b="C_regime_aware")` 出晋升报告。C 臂接线与生产一致：
facade(HOW_INTERVENTION) → regime builder → matcher → 选择性决策 →
编舞门 →（真机时再经 sandbox/许可证门）。
