# Architecture final

Status: PASS.

The final implementation preserves the module authority boundaries:

| Concern | Owner | Final invariant |
|---|---|---|
| Current robot state | Body | Know/How do not manufacture runtime state |
| Past robot experience | Memory | Never ingested into the world-knowledge index |
| External world knowledge | Know | Canonical snapshots, claims, Wiki, units, retrieval |
| Contextual engineering advice | How | Advisory only; no Action authority |
| Executable capability | Skill/MCP registry | Separate from relevance and knowledge truth |
| Physical effect | Core Action chain | Operator/safety authority remains authoritative |

Know now follows `Source → SourceSnapshot → Document → deterministic facts →
KnowledgeClaim → Wiki/KnowledgeUnit → ReferencePack`. Claims separate truth,
utility, compatibility, and retrieval scores. Cross-source conflicts create a
review record and fail closed at pack construction.

How consumes only the versioned Reference Pack wire contract. Its explanation
contains structured candidate decisions and explicitly fixes
`private_reasoning_disclosed=false`. It cannot invoke tools or actions.

Core contains routing, clients, budgets, and session workspace state only. It
does not duplicate indexing, ranking, claim compilation, or advice selection.

Memory and Practice schemas were not modified by this acceptance round.
