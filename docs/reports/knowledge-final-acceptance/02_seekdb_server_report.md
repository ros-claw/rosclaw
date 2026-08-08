# SeekDB server report

Status: PASS.

The acceptance database ran in a real SeekDB 1.3 server container on port
2881. The server, database, container name, and temporary volume were scoped
only to this campaign.

## Migration and durability matrix

| Operation | Result |
|---|---|
| Empty database → migrations 1..7 | PASS |
| Apply 1..7 again | PASS, idempotent |
| Down to 6 → up to 7 | PASS |
| Synthetic migration 8 failure | PASS, rollback left version 7 and no partial rows |
| Container restart | PASS, migration rows 1..7 and two native records remained |
| Logical backup → delete → restore | PASS, 2/2 records restored |

Snapshots remain immutable through store validation. Transactions are native
in server mode.

## Native retrieval

The live table created and queried:

- BENG full-text index for English content;
- IK full-text index for Chinese content;
- NGRAM indexes for error, symbol, API, and path surfaces;
- HNSW vector index;
- `DBMS_HYBRID_SEARCH.SEARCH` and `GET_SQL`;
- RRF across keyword and semantic lanes;
- status and compatibility filters in both lanes.

The final 20-query native error-profile sample returned only `current`, never
the incompatible/superseded row. Observed p95 was 9.35 ms. The saved SQL
contained both `_keyword_rank` and `_semantic_rank`, and applied
`status='active'` plus `compatibility_status='compatible'` to both subqueries.

`AI_RERANK` was probed with an unavailable model key. Capability reporting was
honest: `available=false`, reason recorded, fallback `deterministic_rrf`.
