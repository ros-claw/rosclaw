"""Data Flywheel lineage graph view (PR-DF-23 / phase-II §9 Dashboard).

Not another card page: ONE real ancestry graph —

    Episode → Failure → Memory → Insight → Proposal → Patch
            → Experiment → Darwin → Champion

``build_lineage_payload`` resolves the trace_graph DAG plus per-node
detail: evidence, body, execution mode, artifact, score, parents, and
the why (promoted / rejected).  ``LINEAGE_PAGE_HTML`` is the interactive
viewer (vanilla JS + SVG, no build step).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("rosclaw.dashboard.lineage_view")

_EVOLUTION_NS_BY_TYPE = {
    "failure": "failures",
    "diagnosis": "diagnoses",
    "proposal": "proposals",
    "patch": "patches",
    "experiment": "experiments",
    "evaluation": "evaluations",
    "champion": "champions",
    "dead_end": "deadends",
    "task": "tasks",
}


def _load_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _evolution_record(client: Any, entity_type: str, entity_id: str) -> dict[str, Any] | None:
    namespace = _EVOLUTION_NS_BY_TYPE.get(entity_type)
    if not namespace:
        return None
    try:
        rows = client.query("evolution_records", {"id": f"{namespace}:{entity_id}"}, limit=1)
        if rows:
            return _load_json(rows[0].get("data")) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("evolution record lookup failed for %s: %s", entity_id, exc)
    # the SQL auto_* tables keep a copy for some types
    table = {"proposal": "auto_proposals", "patch": "auto_patches"}.get(entity_type)
    if table:
        try:
            rows = client.query(table, {"id": entity_id}, limit=1)
            if rows:
                return dict(rows[0])
        except Exception:  # noqa: BLE001
            pass
    return None


def _node_detail(client: Any, entity_type: str, entity_id: str) -> dict[str, Any]:
    """The clickable per-node card: evidence/body/mode/artifact/score/why."""
    detail: dict[str, Any] = {"type": entity_type, "id": entity_id}
    try:
        if entity_type == "memory":
            rows = client.query("memory_items", {"id": entity_id}, limit=1)
            if rows:
                row = rows[0]
                detail.update(
                    {
                        "title": row.get("title"),
                        "outcome": row.get("outcome"),
                        "body_id": row.get("body_id"),
                        "skill_id": row.get("skill_id"),
                        "quality_score": row.get("quality_score"),
                        "evidence_refs": _load_json(row.get("evidence_refs")) or [],
                        "artifact_refs": _load_json(row.get("artifact_refs")) or [],
                    }
                )
        elif entity_type == "episode":
            rows = client.query("episodes", {"id": entity_id}, limit=1)
            if rows:
                row = rows[0]
                detail.update(
                    {
                        "outcome": row.get("outcome"),
                        "task_id": row.get("task_id"),
                        "robot_id": row.get("robot_id"),
                        "artifact_uri": row.get("artifact_uri"),
                    }
                )
        elif entity_type == "receipt":
            rows = client.query("execution_receipts", {"id": entity_id}, limit=1) or client.query(
                "execution_receipts", {"action_id": entity_id}, limit=1
            )
            if rows:
                row = rows[0]
                detail.update(
                    {
                        "execution_mode": row.get("execution_mode"),
                        "final_state": row.get("final_state"),
                        "evidence_level": row.get("evidence_level"),
                        "evidence_domain": row.get("evidence_domain"),
                        "body_id": row.get("body_id"),
                        "capability_id": row.get("capability_id"),
                    }
                )
        elif entity_type == "darwin_benchmark":
            rows = client.query("darwin_benchmarks", {"id": entity_id}, limit=1)
            if rows:
                detail.update(_load_json(rows[0].get("data")) or dict(rows[0]))
        elif entity_type in _EVOLUTION_NS_BY_TYPE:
            record = _evolution_record(client, entity_type, entity_id)
            if record:
                detail.update(
                    {
                        "status": record.get("status"),
                        "score": record.get("metrics") or record.get("candidate_metrics"),
                        "decision": record.get("decision"),
                        "body_id": record.get("body_id"),
                        "evidence": record.get("evidence") or record.get("evidence_refs"),
                    }
                )
                if entity_type == "champion":
                    detail["why_promoted"] = {
                        "level": record.get("level"),
                        "evaluation_id": record.get("evaluation_id"),
                        "promotion_verified": (record.get("validation_summary") or {}).get(
                            "promotion_verified"
                        ),
                    }
                elif entity_type == "dead_end":
                    detail["why_rejected"] = record.get("rejection_reason")
                    detail["direction"] = record.get("direction")
                elif entity_type == "evaluation":
                    detail["why_rejected"] = (
                        record.get("decision")
                        if record.get("decision") not in (None, "promote", "approved")
                        else None
                    )
                    detail["delta"] = record.get("delta")
                elif entity_type == "proposal":
                    detail["hypothesis"] = record.get("hypothesis_statement")
                    detail["dead_end_guard"] = record.get("dead_end_guard")
    except Exception as exc:  # noqa: BLE001 — detail is best-effort
        detail["error"] = str(exc)
    return detail


def build_lineage_payload(client: Any, entity_type: str, entity_id: str) -> dict[str, Any]:
    """trace_graph DAG + per-node detail for the viewer."""
    from rosclaw.storage.lineage import LineageRepository

    graph = LineageRepository(client).trace_graph(entity_type, entity_id)
    details = {
        f"{node['type']}:{node['id']}": _node_detail(client, node["type"], node["id"])
        for node in graph.get("nodes", [])
    }
    return {**graph, "details": details}


LINEAGE_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ROSClaw — Data Flywheel Lineage</title>
<style>
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 0;
         background: #0d1117; color: #c9d1d9; }
  header { padding: 12px 20px; border-bottom: 1px solid #30363d; display: flex;
           gap: 8px; align-items: center; }
  input, button { background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
                  border-radius: 6px; padding: 6px 10px; font: inherit; }
  button { cursor: pointer; background: #21262d; }
  #main { display: flex; height: calc(100vh - 57px); }
  #graph { flex: 1; overflow: auto; }
  #detail { width: 380px; border-left: 1px solid #30363d; padding: 16px;
            overflow-y: auto; font-size: 13px; }
  .node rect { fill: #161b22; stroke: #58a6ff; rx: 6; cursor: pointer; }
  .node.champion rect { stroke: #d2a8ff; }
  .node.memory_insight rect { stroke: #ffa657; }
  .node.memory rect { stroke: #7ee787; }
  .node text { fill: #c9d1d9; font-size: 11px; pointer-events: none; }
  .edge { stroke: #30363d; fill: none; }
  .edge-label { fill: #8b949e; font-size: 9px; }
  .kv { margin: 2px 0; word-break: break-all; }
  .k { color: #8b949e; }
  .why { border: 1px solid #30363d; border-radius: 6px; padding: 8px;
         margin-top: 10px; background: #161b22; }
</style>
</head>
<body>
<header>
  <strong>Lineage</strong>
  <input id="entity" size="38" placeholder="champion:champ_xxx / memory:mem_xxx / proposal:prop_xxx">
  <button onclick="load()">Trace</button>
  <span id="status" style="color:#8b949e"></span>
</header>
<div id="main">
  <div id="graph"></div>
  <div id="detail"><em>Click a node for evidence / body / mode / score / why.</em></div>
</div>
<script>
let DATA = null;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function load() {
  const entity = document.getElementById("entity").value.trim();
  if (!entity.includes(":")) { document.getElementById("status").textContent = "use type:id"; return; }
  const [type, id] = entity.split(":", 2);
  document.getElementById("status").textContent = "loading…";
  const resp = await fetch(`/api/lineage/${encodeURIComponent(type)}/${encodeURIComponent(id)}`);
  DATA = await resp.json();
  document.getElementById("status").textContent =
    `${DATA.nodes.length} nodes, ${DATA.edges.length} edges` + (DATA.truncated ? " (truncated)" : "");
  render();
}

function layout() {
  const byDepth = {};
  for (const n of DATA.nodes) (byDepth[n.depth] = byDepth[n.depth] || []).push(n);
  const pos = {};
  const dy = 86, dx = 200;
  for (const [depth, nodes] of Object.entries(byDepth)) {
    nodes.forEach((n, i) => {
      pos[`${n.type}:${n.id}`] = { x: 60 + i * dx, y: 40 + depth * dy };
    });
  }
  return pos;
}

function render() {
  const pos = layout();
  const W = Math.max(...Object.values(pos).map(p => p.x)) + 220;
  const H = Math.max(...Object.values(pos).map(p => p.y)) + 90;
  let svg = `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">`;
  for (const e of DATA.edges) {
    const a = pos[e.from], b = pos[e.to];
    if (!a || !b) continue;
    svg += `<path class="edge" d="M ${a.x+70} ${a.y+22} L ${b.x+70} ${b.y+22}"/>`;
    svg += `<text class="edge-label" x="${(a.x+b.x)/2+62}" y="${(a.y+b.y)/2+18}">${esc(e.relation)}</text>`;
  }
  for (const n of DATA.nodes) {
    const p = pos[`${n.type}:${n.id}`];
    const label = `${n.type} ${n.id}`.slice(0, 26);
    svg += `<g class="node ${esc(n.type)}" onclick="show('${esc(n.type)}','${esc(n.id)}')">
      <rect x="${p.x}" y="${p.y}" width="140" height="44" rx="6"/>
      <text x="${p.x+8}" y="${p.y+18}">${esc(n.type)}</text>
      <text x="${p.x+8}" y="${p.y+34}" style="fill:#8b949e">${esc(n.id.slice(0, 20))}</text></g>`;
  }
  svg += "</svg>";
  document.getElementById("graph").innerHTML = svg;
}

function show(type, id) {
  const d = (DATA.details || {})[`${type}:${id}`] || {type, id};
  let html = `<h3>${esc(type)} ${esc(id)}</h3>`;
  const skip = new Set(["type", "id"]);
  for (const [k, v] of Object.entries(d)) {
    if (skip.has(k) || v === null || v === undefined || v === "") continue;
    const cls = k.startsWith("why") ? "kv why" : "kv";
    html += `<div class="${cls}"><span class="k">${esc(k)}:</span> ${esc(typeof v === "object" ? JSON.stringify(v) : v)}</div>`;
  }
  const parents = DATA.edges.filter(e => e.from === `${type}:${id}`);
  if (parents.length) {
    html += `<div class="why"><span class="k">parents:</span>` +
      parents.map(e => `<div class="kv">—${esc(e.relation)}→ ${esc(e.to)}</div>`).join("") + "</div>";
  }
  document.getElementById("detail").innerHTML = html;
}

const q = new URLSearchParams(location.search).get("entity");
if (q) { document.getElementById("entity").value = q; load(); }
</script>
</body>
</html>
"""
