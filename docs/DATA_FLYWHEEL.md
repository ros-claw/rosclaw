# ROSClaw Data Flywheel（数据飞轮）

> 状态：架构目标文档（PR-DF-00 冻结，不改变行为）
> 依据：ADR-0009（Data Plane 分层）、ADR-0010（Canonical Vocabulary）、
> 《ROSClaw SeekDB-Centric Data Flywheel 重构实施总纲》

## 一句话

ROSClaw 让具身 Agent 不仅能够行动，还能够把每一次物理行动变成可验证的经验；
把经验沉淀为记忆；把记忆抽象为知识；把知识转化为干预和技能；再通过真实或
仿真的物理反馈验证改变，完成持续自进化。

## 飞轮全图

```text
                         ┌────────────────────┐
                         │   Physical World   │
                         │ Real / Sim / Human │
                         └─────────┬──────────┘
                                   │
                     Observation / Action / Receipt
                                   │
                                   ▼
┌────────────────────────────────────────────────────────┐
│                    PRACTICE / EVIDENCE                 │
│ episode · praxis_event · failure · telemetry metadata  │
│ receipt · critic_result · artifact pointer · hash      │
└─────────────────────────┬──────────────────────────────┘
                          │  EpisodeFactExtractor (session close trigger)
                          ▼
┌────────────────────────────────────────────────────────┐
│                     MEMORY                             │
│ episodic · failure · intervention · procedural · body  │
│ spatial · skill · sim2real · human_feedback            │
│        MemoryItem ←──── evidence_refs ──── Evidence    │
└─────────────────────────┬──────────────────────────────┘
                          │  Structured Store (SeekDB SQL / SQLite)
                          │  → Outbox → Retrieval Projection
                          ▼
              SeekDB Hybrid Retrieval
             vector + BM25 + metadata + body-aware
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
┌─────────────────────┐     ┌─────────────────────┐
│     KNOWLEDGE       │     │         HOW         │
│ generalized priors  │────▶│ runtime intervention│
│ reference packs     │     │ recovery strategies │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           └────────────┬──────────────┘
                        ▼
                 Agent / Skill / Policy
                        │
                        ▼
              Sandbox / Physical Execute
                        │
             Verifier / Critic / Reward
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Memory Feedback          EVOLUTION
                      Proposal → Patch → Experiment
                                │
                                ▼
                              DARWIN
                         baseline / candidate
                                │
                         Promotion Gate
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
                 Champion               DeadEnd
                     │
                     ▼
               Skill Registry ──────→ 再次执行
```

## 数据 ownership（与 ARCHITECTURE.md 一致）

| 模块 | 拥有 | 一句话 |
|---|---|---|
| Practice | 事实记录 | 发生了什么 |
| Memory | 长期具身回忆 | 应该记住什么 |
| Knowledge | 编译后的工程先验 | 可以泛化出什么 |
| How | 运行时干预 | 当前应该怎么修 |
| Evolution | 自进化编排 | 应该尝试改变什么 |
| Darwin | 评估压力 | 改变究竟好不好 |
| Learning | 模型/权重更新 | 真正训练 |
| Structured Store | 结构化事实 source of truth | SeekDB SQL / SQLite |
| Retrieval Store | 检索投影（可重建） | SeekDB Native |
| rosclawd | 物理执行与授权边界 | 唯一权威，永不等数据库 |

## 经验血缘链（Lineage）

任意一次行为必须可沿 Structured Store 追溯：

```text
trace_id → action_id → receipt_id → practice_id → session_id → episode_id
→ evidence_id → memory_id → reference_pack_id → advice_id → proposal_id
→ patch_id → experiment_id → evaluation_id → champion_id → skill_version
```

验收六问（Definition of Done 的北极星）：

1. 给我一个失败 Episode —— 发生了什么、为什么失败、证据在哪里？
2. 给我一个 Memory —— 它来自哪些真实 Practice/Receipt？
3. 给我一个 How Recovery —— 成功/失败多少次、在哪些 Body/Regime 下有效？
4. 给我一个 Knowledge Unit —— 何时真帮助过、何时被证明 incompatible/stale？
5. 给我一个 Evolution Proposal —— 为什么提出？来自什么 Failure/Memory/Insight？
6. 给我一个 Champion Skill —— 能否追到最初的 Physical Receipt？

## 实施序列（每 PR 独立可审）

| PR | 内容 | 行为改变 |
|---|---|---|
| DF-00 | 本文档 + ADR-0009/0010 | 无 |
| DF-01 | Storage 命名（StructuredStore 等 + 兼容别名） | 无 |
| DF-02 | Config v2（storage.structured/retrieval/outbox + legacy 归一化） | 仅配置层 |
| DF-03 | DataPlaneContext，Runtime 单次初始化 | 结构 |
| DF-04 | EpisodeFactExtractor / PracticeFactIngestor + session-close 自动 verify→extract→ingest | 行为 |
| DF-05 | Memory canonical 写入主链（Critic → MemoryItem） | 行为 |
| DF-06 | `rosclaw memory migrate`（experience_graph → memory_items） | 工具 |
| DF-07 | Retrieval projection 正式接线 + watermark/lag/doctor | 行为 |
| DF-08 | RecoveryLoop 挂入 Runtime | 行为 |
| DF-09 | knowledge 包合并（know shim）+ federation 配置接通 | 结构 |
| DF-10 | KnowledgeUsageTracker 自动反馈闭环 | 行为 |
| DF-11 | MemoryInsightService（publisher） | 行为 |
| DF-12 | EvolutionRepository（StructuredStore 为权威，LocalStore 降级 cache） | 行为 |
| DF-13 | Darwin 挂入 Runtime | 行为 |
| DF-14 | execution_receipts + lineage_edges + LineageRepository | 新增 |
| DF-15 | Dashboard Data Flywheel 页 + storage doctor 增强 | 观测 |
| DF-16 | 物理移动与清理（最后做） | 无 |

## 绝对禁止（与实施总纲 §61 一致）

1. 删除 rosclawd 本地安全 ledger；
2. REAL execution 等待 SeekDB；
3. raw video / MCAP 进 SeekDB；
4. 完整 Chain-of-Thought 进长期 Memory；
5. 数据库写失败时伪造 "memory saved / knowledge learned / evolution completed"；
6. Retrieval Store 当唯一 source of truth（投影必须可删除可重建）；
7. Evolution 绕过 Sandbox / Darwin / Promotion Gate / Human Approval。
