"""Reader-facing GoalForge showcase manifest builder."""

from __future__ import annotations

import json
from typing import Any

from rosclaw.simforge.backends.g1_visual_backend import trajectory_overlay
from rosclaw.simforge.tasks.g1_goalforge.task import SameSeedKickPair


def build_showcase_manifest(pair: SameSeedKickPair) -> dict[str, Any]:
    return {
        "schema_version": "rosclaw.g1_goalforge.showcase.v1",
        "title": "G1 自进化点球挑战赛",
        "slogan": "第一脚踢偏，第二脚修正，第三脚换个位置也能进。",
        "causal_passed": pair.causal_passed,
        "shots": [
            {
                "label": "baseline",
                "color": "red",
                "result": pair.baseline.result.summary_dict(),
                "trajectory": trajectory_overlay(pair.baseline.trajectory),
            },
            {
                "label": "same-seed retry",
                "color": "green",
                "result": pair.retry.result.summary_dict(),
                "trajectory": trajectory_overlay(pair.retry.trajectory),
            },
        ],
        "evidence": {
            "same_seed": pair.same_seed,
            "same_scenario": pair.same_scenario,
            "only_candidate_changed": pair.only_candidate_changed,
            "baseline_receipt": (
                pair.baseline.receipt.receipt_hash if pair.baseline.receipt else None
            ),
            "retry_receipt": (pair.retry.receipt.receipt_hash if pair.retry.receipt else None),
        },
    }


def render_showcase_html(manifest: dict[str, Any]) -> str:
    """Render a dependency-free, evidence-labelled three-panel dashboard."""

    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    return _HTML.replace("__GOALFORGE_MANIFEST__", payload)


_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 GoalForge</title>
<style>
:root{color-scheme:dark;--bg:#071018;--panel:#0e1d29;--line:#294657;--muted:#93aaba;--cyan:#44d7ff;--green:#38e68d;--red:#ff5f67;--blue:#599cff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -20%,#193a4a,var(--bg) 48%);font:14px/1.45 ui-sans-serif,system-ui;color:#eaf8ff}
header{padding:26px 28px 14px}h1{margin:0;font-size:28px;letter-spacing:.03em}header p{margin:5px 0;color:var(--cyan)}
main{display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:14px;padding:12px 22px 24px}.panel{background:linear-gradient(160deg,#102532,#0b1721);border:1px solid var(--line);border-radius:15px;padding:16px;min-height:500px;box-shadow:0 16px 50px #0006}
h2{font-size:14px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);margin:0 0 14px}canvas{width:100%;height:310px;background:#07151e;border:1px solid #284858;border-radius:10px}
.legend,.metrics{display:grid;gap:8px;margin-top:13px}.shot,.metric{display:flex;align-items:center;justify-content:space-between;padding:8px 10px;background:#0a1620;border-radius:8px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:8px}.chain{display:flex;flex-direction:column;gap:7px}.module{padding:7px 10px;border:1px solid #285065;border-radius:8px;color:#cae8f7;background:#0a1822}.module.on{border-color:#2b9b71;box-shadow:inset 3px 0 var(--green)}
.badge{padding:2px 7px;border-radius:12px;background:#173343;color:var(--cyan)}.pass{color:var(--green)}.fail{color:var(--red)}footer{padding:0 24px 25px;color:var(--muted);font-size:12px}
@media(max-width:1000px){main{grid-template-columns:1fr}.panel{min-height:auto}}
</style>
</head>
<body>
<header><h1 id="title"></h1><p id="slogan"></p></header>
<main>
<section class="panel"><h2>G1 MuJoCo trajectory</h2><canvas id="field" width="720" height="420"></canvas><div class="legend" id="legend"></div></section>
<section class="panel"><h2>ROSClaw causal module chain</h2><div class="chain" id="chain"></div></section>
<section class="panel"><h2>Verifier & receipt</h2><div class="metrics" id="metrics"></div></section>
</main>
<footer>Simulation-only SHADOW evidence. CUDA screening is not physical truth. No real robot transport is opened.</footer>
<script>
const data=__GOALFORGE_MANIFEST__;
document.querySelector("#title").textContent=data.title;
document.querySelector("#slogan").textContent=data.slogan;
const colors={red:"#ff5f67",green:"#38e68d",blue:"#599cff"};
const canvas=document.querySelector("#field"),ctx=canvas.getContext("2d");
ctx.strokeStyle="#294657";ctx.lineWidth=2;ctx.strokeRect(55,45,610,320);
ctx.setLineDash([7,7]);ctx.beginPath();ctx.moveTo(610,45);ctx.lineTo(610,365);ctx.stroke();ctx.setLineDash([]);
for(const shot of data.shots){
 const points=shot.trajectory?.ball_xyz||[];ctx.beginPath();ctx.strokeStyle=colors[shot.color]||shot.color;ctx.lineWidth=3;
 points.forEach((p,i)=>{const x=55+Math.max(0,Math.min(5.5,p[0]))/5.5*610;const y=205-Math.max(-1.2,Math.min(1.2,p[1]))/2.4*280;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();
 const row=document.createElement("div");row.className="shot";row.innerHTML=`<span><i class="dot" style="background:${colors[shot.color]||shot.color}"></i>${shot.label}</span><b class="${shot.result.success?"pass":"fail"}">${shot.result.status}</b>`;document.querySelector("#legend").append(row);
}
for(const name of data.module_chain||[]){const el=document.createElement("div");el.className="module on";el.textContent=name;document.querySelector("#chain").append(el)}
const retry=data.shots.find(s=>s.label==="same-seed retry")||data.shots[1],base=data.shots[0];
const items=[["Causal pair",data.causal_passed],["Baseline error",`${base.result.target_error_m.toFixed(3)} m`],["Retry error",`${retry.result.target_error_m.toFixed(3)} m`],["Retry fall",retry.result.post_kick_fall],["Torque violation",retry.result.torque_limit_violation],["Strict receipts",Boolean(data.evidence?.baseline_receipt&&data.evidence?.retry_receipt)]];
for(const [k,v] of items){const el=document.createElement("div");el.className="metric";el.innerHTML=`<span>${k}</span><span class="badge">${v}</span>`;document.querySelector("#metrics").append(el)}
</script>
</body>
</html>
"""


__all__ = ["build_showcase_manifest", "render_showcase_html"]
