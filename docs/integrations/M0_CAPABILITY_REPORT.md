# M0 Capability Report — m0.seekdb.ai（数据库优化v2.md §9.1）

**Date:** 2026-07-16 · **Method:** 公开文档（SKILL.md）+ 只读探针 + 最小合成写入探针（已全部清理）
**Verdict 图例：** ✅ supported · ❌ unsupported · ❓ unknown · 📞 requires vendor confirmation

## 1. 结论摘要

M0 是面向 OpenClaw 的托管云端记忆服务（Experience + Skill 两层，向量+BM25 混合召回，自动蒸馏，共享经验池）。**读路径（health/instance status/search/list）全部实测可用；写路径（capture 蒸馏写入、直接 create）在当前时点实测失败或静默无操作**，需要厂商确认后才能进入任何业务依赖。按 v2 文档约束，M0 只能做 **Optional Cloud Memory Projection**，永不在机器人热路径上。

## 2. API 能力矩阵（实测）

| 能力 | Endpoint | 状态 | 实测证据 |
|---|---|---|---|
| Service health | `GET /health` | ✅ | `{"status":"ok"}` (2026-07-16T22:33Z) |
| Authentication | `X-API-Key: ak_*` | ✅ | 无 key/错 key → 404/拒绝；实例即租户 |
| Instance create | `POST /api/instances/` | ✅ | 返回 `ak_58173aa1...`（探针实例，已弃置） |
| Instance status | `GET /api/instances/{AK}/status` | ✅ | `{status:"ready", memory_count:0}` |
| Memory capture（对话蒸馏写入） | `POST /api/memories/capture` | ❌ | 三种 messages 变体均返回 `{add:0,update:0,skip:0}`，status 的 memory_count 恒为 0 |
| Memory search（向量+全文+rewritten query） | `POST /api/memories/search` | ✅（接口可用） | 返回结构 `{memories, total, rewritten_queries, experiences}`；因写入不可用无法验证召回质量 |
| Memory list | `GET /api/memories/` | ✅ | 结构正常（空） |
| Memory get/update/delete (by id) | `GET/PUT/DELETE /api/memories/{id}` | ❓ | 无可写入的记忆，无法实测 |
| Memory delete all | `DELETE /api/memories/` | ✅ | HTTP 200 |
| Memory 直接创建 | `POST /api/memories/` | ❌ | HTTP 500 Internal Server Error |
| Memory export（批量导出） | — | ❓ | 公开文档未见导出端点 📞 |
| Tenant isolation（租户隔离） | 实例 = AK 即隔离边界 | ✅（设计） | 每个 AK 一个独立实例；跨实例互不可见（按文档，未深测） |
| Retention / TTL | — | ❓ | 无公开说明 📞 |
| Encryption at rest | — | ❓ | 无公开说明 📞 |
| Rate limit | — | ❓ | 探针期间未触发；无公开配额 📞 |
| Embedding 模型 | 服务端内建 | ❓ | 未公开型号；hybrid = 向量+BM25（官方博客） |
| Distillation（自动蒸馏 Experience+Skill） | capture 管线 | ❓ | 官方博客详述（skills-first, progressive loading, dedup>0.75 merge），但当前 capture 实测不产数 📞 |
| Skill extraction | 同上 | ❓ | 同上 |
| Data region（数据驻留） | — | ❓ | 无公开说明 📞 |
| Pricing | — | ❓ | 无公开定价 📞 |
| SLA | — | ❓ | 无公开 SLA 📞 |

## 3. OpenClaw 集成面（来自 SKILL.md）

- 插件安装：`openclaw plugins install clawhub:m0`（≥2026.3.22）或本地编译。
- 配置项：`apiKey / baseUrl / autoCapture / autoRecall / recallLimit`。
- 宿主要求：OpenClaw ≥ 2026.2.2。**ROSClaw 不是 OpenClaw**——要接 M0 必须自研 `M0CloudMemoryAdapter`（HTTP 直连），不能用其插件。

## 4. 对 ROSClaw 的建议（§9.2 原则约束）

1. **暂不实现 `M0CloudMemoryAdapter`**：写路径实测失败，等厂商确认 capture/create 契约后再启动 PR-M0-2。
2. 若未来接入，只能走：`Local Memory → Privacy Filter → Redaction → M0 Outbox → Cloud Upload`，默认禁止上传 Raw RGB-D / MCAP / 完整 CoT / 用户敏感信息 / API Key / 精确地图 / 未确认真机数据。
3. 重验清单（给厂商的问题）：
   - `POST /api/memories/capture` 的 messages schema 与蒸馏触发条件；为何返回 `{add:0,update:0,skip:0}`？
   - `POST /api/memories/` 直接创建 500 的原因与正确 schema；
   - export/retention/encryption/rate limit/data region/pricing/SLA 的正式契约。

## 5. 探针记录（已全部清理）

- 创建探针实例 `rosclaw-capability-spike-probe`（ak_58173aa1…，零记忆，已弃置）。
- 合成非敏感测试内容 3 次 capture（均未产数）+ 1 次直接 create（500）。
- `DELETE /api/memories/` 返回 200，实例 memory_count 保持 0 —— 无测试数据残留。
