# ROSClaw Data Flywheel 重构实施报告

- 日期：2026-08-14
- 依据：《当前 rosclaw seekdb 情况》+《ROSClaw SeekDB-Centric Data Flywheel 重构实施总纲 v1》
- Baseline commit：`cfa671ad3510b7b0a53cb02cc0c98a394c59ea04`（main 同步点）
- Baseline 测试：6831 collected（39 deselected）；31 failed / 10 errors 全部为环境预置
  （venv 缺 `rosclaw_know` 包、agentd/firstboot/readme 的 PATH 依赖等，均在未改动的
  main 上逐名复现）

## 1. PR 序列与状态

| PR | 分支 | 内容 | 验证 |
|---|---|---|---|
| #344 DF-00 | pr-df-00-architecture-freeze | ADR-0009/0010 + DATA_FLYWHEEL.md，零行为 | docs-only |
| #345 DF-01 | pr-df-01-storage-naming | Storage 命名 12 组 canonical + identity 别名 | 别名恒等 7 项；全量套件逐名比对（1 个 mock 目标回归已修） |
| #346 DF-02 | pr-df-02-config-v2 | Config v2 + normalize_legacy_config + 镜像 | 8 项新测试 |
| #347 DF-03 | pr-df-03-dataplane-context | DataPlaneContext 单次装配、逐项故障隔离 | 4 项新测试；335 套件绿 |
| #348 DF-04 | pr-df-04-practice-facts | EpisodeFactExtractor/PracticeFactIngestor + close 时 verify→extract→ingest | 4 项；practice 194 绿 |
| #349 DF-05 | pr-df-05-memory-write-path | Critic → MemoryItem(WriteGate→Repository) 主写链 | 4 项；529 套件绿 |
| #350 DF-06 | pr-df-06-memory-migrate | `rosclaw memory migrate`（真实库实证幂等 11→0） | 2 项 |
| #352 DF-07 | pr-df-07-retrieval-projection | 检索投影装配 + target 过滤 worker + lag 可观测 | 5 项；511 套件绿 |
| #353 DF-08 | pr-df-08-recovery-loop | RecoveryLoop 入 Runtime 生命周期 | 3 项；239 绿 |
| #354 DF-09 | pr-df-09-knowledge-consolidation | knowledge canonical + legacy shim + 联邦坐标 | 4 项 |
| #355 DF-10 | pr-df-10-knowledge-feedback | KnowledgeUsageTracker 保守自动反馈 | 5 项 |
| #356 DF-11 | pr-df-11-memory-insight | MemoryInsightService（insight 生产者） | 5 项；342 绿 |
| #359 DF-12 | pr-df-12-evolution-repository | EvolutionRepository，LocalStore 降级 spool | 7 项；414 绿 |
| #360 DF-13 | pr-df-13-darwin-runtime | Darwin 入 Runtime（默认关） | 2 项 |
| #361 DF-14 | pr-df-14-receipt-lineage | execution_receipts + lineage_edges + LineageRepository + ReceiptProjector | 5 项；349 绿 |
| #362 DF-15 | pr-df-15-observability | doctor 飞轮检查 + /api/data-flywheel + 全栈 ruff 清零 | 5 项；443 绿 |

分支为栈式叠加（每个 PR 的 base 是前一个分支）；按序合并即可，或全部合并后
统一 rebase 到 main。git transport 不稳定期间，DF-01 修复、DF-07、DF-12、DF-13、
DF-15 经 Git Data API 推送，内容与本地的对应提交一致。

## 2. Canonical naming mapping（落地版）

见 ADR-0010 表。落地情况：`StructuredStore`/`InMemoryStructuredStore`/
`SQLiteStructuredStore`/`SeekDBSQLStore`/`ROSCLAW_STRUCTURED_SCHEMAS`/
`StoreFactory.create_structured_store`/`SeekDBRetrievalStore`(+Embedded/Server)/
`MemoryRetrievalProjection`(+Committer)/`EpisodeFactExtractor`/`EpisodeFactBundle`/
`PracticeFactIngestor` —— 全部 canonical 为真实定义，旧名 identity 别名，
`tests/storage/test_structured_naming.py` 逐项钉死。

`rosclaw.knowledge` canonical、`rosclaw.know` 标记 DEPRECATED shim、
`rosclaw.knowledge.legacy` 再导出（`LegacyKnowledgeRuntime`）。`rosclaw.evolution`
获得 `repository.py`；`rosclaw.auto` 未动（orchestrator 迁移留给 DF-16）。

## 3. 新 Data Plane 架构

```
Runtime._create_data_plane()  (一次)
  └─ DataPlaneContext
      ├─ structured_store   SQLite(边缘) / SeekDB SQL(生产) / InMemory(测试)
      ├─ retrieval_store    SeekDB Native（retrieval 启用且 structured=SQLite 时）
      ├─ memory_projection  MemoryRetrievalProjection（注入 DF-05 的 Repository）
      ├─ memory_retrieval   RetrievalFacade（PR-MEM-5 已有）
      ├─ outbox             OutboxStore（启用时）
      └─ practice_sink      SeekDBBridge（HTTP，配置时）

Runtime 装配（全部 best-effort、stop 时逆序卸载）：
  MemoryRepository+WriteGate (DF-05) · RecoveryLoop (DF-08) ·
  KnowledgeUsageTracker (DF-10) · MemoryInsightService (DF-11) ·
  ReceiptProjector+LineageRepository (DF-14) · DarwinPlugin (DF-13)
```

## 4. Config migration

`normalize_legacy_config()`：`runtime.seekdb_backend/url/path`、`memory.backend`、
`storage.vector_enabled`、`know`、`auto` → canonical section，每条一次性
`DEPRECATED CONFIG` 警告；canonical 值镜像回 legacy 键，DF-03 前的读者零影响。
structured 默认路径未动（`~/.rosclaw/data/memory/knowledge.sqlite`）——路径迁移
需要带数据搬运的独立 PR。

## 5. DB/schema migration

新增表：`evolution_records`（DF-12）、`execution_receipts`（DF-14 §21）、
`lineage_edges`（DF-14 §36）。`rosclaw memory migrate`（DF-06）完成
experience_graph → memory_items 迁移，本机真实库实证幂等。

## 6-15. 各管线落地

- **Practice**：close 时 `verify → extract → ingest` 一条链；verify 可观测不阻塞（DF-04）。
- **Memory**：Critic 判决走 WriteGate→Repository 主写，experience_graph 降为兼容投影（DF-05）；
  迁移 CLI（DF-06）。
- **Retrieval**：投影注入 Repository，outbox target 过滤 worker，status/lag/doctor（DF-07/15）。
- **How**：RecoveryLoop 订阅/退订入 Runtime（DF-08）；物理验证门原样保留。
- **Knowledge**：联邦坐标实传（DF-09）；UsageTracker 保守反馈（DF-10）。
- **Insight→Evolution**：publisher 落地，载荷兼容 Auto 现读字段（DF-11）。
- **Evolution**：EvolutionRepository 结构化权威 + spool（DF-12）。
- **Darwin**：入 Runtime，默认关（DF-13）。
- **Lineage**：link/parents/children/ancestors/descendants/trace + ReceiptProjector 自动连边（DF-14）。

## 16. 向后兼容

全部旧类名/配置键/CLI 命令可用；identity 别名有测试钉死；`rosclaw know`/
`rosclaw auto` CLI 未动；`local` storage_backend 默认不变（standalone 零影响）。

## 17-18. Tests / E2E evidence

新增测试 60+ 项分布各 PR。覆盖的 E2E 要素：幂等重放（DF-05/06）、
store  outage 降级（DF-03/12/14）、恢复学习门（DF-08 沿用既有物理验证测试）、
保守反馈判定（DF-10）、跨本体安全沿用 MEM-6/HOW-3 的既有 regime/purpose 测试。

## 19. SeekDB outage evidence

`test_runtime_survives_structured_store_failure`（DF-03）、
`test_store_down_falls_back_to_spool`（DF-12）、
`test_receipt_projector_failure_never_raises`（DF-14）、
`test_doctor_survives_dead_store`（DF-15）。

## 20. 已知限制

- Evolution 实体（proposal/patch/experiment/evaluation/champion）尚未自动写入
  lineage_edges——目前只有 receipt→action 边由 ReceiptProjector 自动产生；
  验收六问之六需要 Evolution 侧 link 调用（小增量，见后续建议）。
- Retrieval projection 的 outbox 路径与 practice bridge 共享 OutboxStore 但各自
  target 过滤；真正多 sink 的 RoutingCommitter 未做（暂不需要）。
- DF-16（物理迁移 know/auto/memory.v2 目录）按总纲纪律刻意未做——行为主线刚收敛，
  物理移动应在合并后独立 PR 进行。
- `tests/knowledge/test_contracts.py` 在本 venv 因缺 `rosclaw_know` 包无法收集
  （预置环境问题）。

## 21. 尚未完成项目

- DF-16 物理迁移与 deprecated 清理。
- Evolution→lineage 自动连边。
- Dashboard Data Flywheel 的可视化页面（当前为 JSON 端点）。
- Config v2 在 RuntimeConfig 层的完整消费（当前经镜像键兼容）。

## 22. 后续建议

1. 按栈序合并 #344..#362；合并后跑全量套件 + `rosclaw db doctor`。
2. 合并后立即做 DF-16（窗口期最短，避免兼容层长期滞留）。
3. Evolution 实体的 lineage 自动连边 + `rosclaw data lineage champion:<id>` CLI。
4. 在 7×24 stress rig 上开启 retrieval 投影观察 lag 一周，再考虑默认启用。
