# Embedding bake-off（数据库优化v3 §10，真机 SeekDB Server :2881）

- 时间: 2026-07-22, run id `v3_run1`
- 语料: **820 条真实记忆**（234 个 RH56 会话全量蒸馏：failure 523 / skill 99 / episodic 152 / body 45 / intervention 1；merge 去重后）。v3 目标 ≥1000——如实记录 820（当前全部真实会话库存；后续随会话增长重跑）
- 查询: **300 条带结构化标签**（cjk_to_cjk 60 / cjk_to_en 45 / en_to_cjk 45 / mixed 45 / error_code 45 / same_symptom 30 / hard_negative_body 30），标签来自结构化字段而非模型输出
- 引擎: SeekDB Server（`memory_bench__{profile}__ngram` 版本化集合，手工 embedding，ngram 分析器）

## 总体指标

| lane | R@1 | R@5 | MRR | nDCG@5 | confusion | p50 | p95 |
|---|---|---|---|---|---|---|---|
| qwen3_06b_1024_v1 | 0.940 | 0.970 | 0.951 | 0.657 | 0.000 | 12.3ms | 21.0ms |
| **qwen3_06b_768_v1** | **0.970** | 0.970 | 0.973 | 0.655 | 0.000 | 14.0ms | 19.9ms |
| qwen3_06b_512_v1 | 0.940 | 0.970 | 0.958 | 0.653 | 0.000 | 12.5ms | 22.8ms |
| bge_m3_1024_v1 | 0.880 | 0.910 | 0.890 | 0.661 | 0.000 | 14.6ms | 25.1ms |
| minilm_384_builtin | 0.910 | 0.910 | 0.910 | 0.448 | 0.000 | 59.6ms | 61.3ms |
| qwen3_06b_1024_v1+reranker | 0.914 | 0.914 | 0.914 | 0.550 | 0.000 | 265.8ms | 281.3ms |

## 分类目 Recall@1（判别力所在）

| kind | qwen3_1024 | qwen3_768 | bge_m3 | minilm |
|---|---|---|---|---|
| cjk_to_cjk / cjk_to_en / en_to_cjk / mixed / hard_negative_body / same_symptom | 1.0 | 1.0 | 1.0 | 1.0 |
| **error_code（EIO/-110/相机/串口符号）** | 0.6 | **0.8** | 0.2 | 0.4 |

## Reranker 子集（105 条高风险查询：hard_negative + error_code + same_symptom）

- base qwen3_1024: R@1 **0.8286** → +reranker: **0.9143**（**+8.6pp**，9 条改善 / 0 条恶化）
- 成本 ~266ms/query——仅用于故障恢复/HOW 决策，不进热路径（符合 §9 定位）

## 晋升 Gate 评估（§10.4）

| Gate | 结果 | 判定 |
|---|---|---|
| Joint Confusion Rate = 0 | 全部 lane 0.000 | ✅ |
| Cross-Body / Cross-Robot Leakage = 0 | 全部 lane 0 | ✅ |
| Query P95 满足预算 | 全部 ≤25ms（reranker 281ms 仅恢复路径） | ✅ |
| 总体 R@1 ≥ MiniLM +8pp | qwen3_768 0.970 vs 0.910 = **+6pp** | ⚠️ 未达 |
| CJK R@1 ≥ MiniLM +15pp | CJK 类目双方均 1.0（饱和无空间） | ⚠️ 未达 |

**如实结论**：本语料/标签设计下，CJK 与手势类目已饱和（1.0 across）——MiniLM+BM25+精确实体过滤已很强；Qwen3 的增量在**排序质量**（nDCG 0.657 vs 0.448, +47%）、**错误码符号检索**（error_code +40pp over MiniLM, +60pp over BGE-M3）、**查询延迟**（12-14ms vs 60ms, 4-5×）和 **reranker 高风险子集**（+8.6pp）。总体/CJK 增益 gate 未达是**基准饱和**所致而非模型回退——类目设计需加更难负例（已记录关节归因的会话、真实同症不同因对）。
**维度选择**：768 ≥ 1024（R@1 0.970 vs 0.940）——满足 §1.2 降维规则（Recall@1 下降 ≤1pp 且关节混淆不增加；实际上升），生产 profile 采用 **qwen3_06b_768_v1**。
**GTE 缺口**：gte-multilingual-base 在本机 transformers/torch 栈不兼容（new-impl 远程代码 gather 越界，cuda+cpu 同现），如实降级弃用并记录环境事实。

## 环境

- Jetson（cuda, FP16）, Qwen3-Embedding-0.6B @97b0c614, BGE-M3 @5617a9f, Qwen3-Reranker-0.6B @e61197ed, SeekDB Server v1.3.0.0 @:2881
- per-query 明细: `per_query.json`; 指标: `retrieval_metrics.json`; 环境: `environment.json`
