# Regime Applicability Benchmark (PR-BENCH-4, 数据库优化v4 §10)

下一版 Benchmark：不再优化已饱和的 CJK/手势车道，转向**工况适用性**——
同症不同工况、同工况不同故障、跨任务、人类自由查询、未见 Session。

## 结构

- `fixture_corpus.py` — session-holdout 语料（dev: 36–50°C 健康会话;
  test holdout: 56–58°C 热退化 + 48–50°C 短会话）。**Ground truth 是物理
  构造的，不是 matcher 推导的**（热会话记忆在冷查询下 BY CONSTRUCTION
  不适用）—— 杜绝循环论证。
- `human_queries.jsonl` — 103 条手写自由查询（非模板生成），覆盖 RH56 /
  D435i / CH340-FTDI / LeRobot / 移动底盘 / VLN / Sandbox / Provider /
  GPU 九类，中英混合、口语化、带缩写。
- `run_regime_benchmark.py` — facade 检索 → RegimeMatcher 适用性门 →
  选择性干预决策 → 逐查询指标 + dev/test 分组汇总。
- `evaluate_regime.py` — v4 §10.3 指标。

## 运行

```bash
.venv/bin/python benchmarks/memory/regime/run_regime_benchmark.py --out /tmp/regime_bench
```

## 指标（当前基线，确定性 3 连跑一致）

| 指标 | 值 | 说明 |
|---|---|---|
| retrieval_recall_at_k | 0.963 | 相关记忆召回 |
| applicable_recall_at_k | 0.741 | 适用记忆召回且门判定一致 |
| inapplicable_top1_rate | 0.204 | 检索把不适用记忆排第一（检索职责，门负责拦） |
| contraindicated_top1_rate | 0.148 | 检索把禁忌记忆排第一（同上） |
| abstention_accuracy | 1.000 | 108/108 决策正确（该停则停，该行才行） |
| apply_precision | 1.000 | 17 次 APPLY 全部落在 validated 适用记忆 |
| regime_confusion_rate | 0.000 | 门从未把不适用判为适用 |

## 判定口径

- ABSTAIN 正确 ⇔ 不存在 validated 适用的记忆（薄证据本来就应停手）。
- APPLY 正确 ⇔ 存在 validated 适用的记忆且其被选中。
- SUGGEST/ESCALATE 永远是安全动作（不产生自主运动）。
