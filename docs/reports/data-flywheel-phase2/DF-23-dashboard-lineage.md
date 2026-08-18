# DF-23 — Data Flywheel lineage graph dashboard (implementation note)

Phase-II P1 per `seekdb优化v2.md` §九: the dashboard's most valuable view
is not more cards but ONE real ancestry graph:
Episode → Failure → Memory → Insight → Proposal → Patch → Experiment →
Darwin → Champion — with clickable nodes.

## What landed

- **`src/rosclaw/dashboard/lineage_view.py`** —
  `build_lineage_payload(client, type, id)`: the DF-17 `trace_graph` DAG
  plus a per-node detail card: evidence refs, body_id, execution mode,
  artifact, score/metrics, and the why — `why_promoted` for champions
  (level, evaluation id, promotion_verified), `why_rejected` for dead
  ends (reason + direction) and rejected evaluations (decision + delta),
  `dead_end_guard` on proposals (DF-22).
- **Routes**: `GET /api/lineage/{type}/{id}` (best-effort like the rest
  of the dashboard — a down database returns an error field, never a
  500) and `GET /lineage` — an interactive SVG viewer (vanilla JS, no
  build step): BFS-depth layout, relation labels on edges, type-colored
  nodes, click-to-detail side panel with a parents section.

## Tests — `tests/dashboard/test_lineage_view.py`

Full-chain payload (champion→receipt, every node type), detail
resolution per type (memory evidence/artifacts, receipt mode/domain,
champion why-promoted, episode artifact), dead-end why-rejected,
missing-entity empty graph (no error), page HTML wiring.  Dashboard
suite: 21 passed, ruff/mypy clean.
