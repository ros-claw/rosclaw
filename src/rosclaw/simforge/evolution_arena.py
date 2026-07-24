"""Self-contained Evolution Arena showcase export and dashboard report loading."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def export_evolution_showcase(
    *,
    run_report: Path,
    output_dir: Path,
    source_checkout: Path,
) -> dict[str, Any]:
    """Export complete trace videos, evidence HTML, receipts, and hashes."""

    report_path = run_report.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("Evolution showcase output must stay outside the checkout")
    if output.exists():
        raise FileExistsError(output)
    report = _load_phase3_report(report_path)
    run_root = report_path.parent
    baseline_path = _one_path(
        run_root / "02-causal-loop" / "flagship" / "baseline",
        "trajectory_states.json",
    )
    recovery_path = _one_path(
        run_root / "02-causal-loop" / "flagship" / "recovery",
        "trajectory_states.json",
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    _atomic_json(output / "technical-report.json", report)
    _render_videos(
        baseline=baseline,
        recovery=recovery,
        output=output,
    )
    receipts_root = output / "receipts"
    receipts_root.mkdir()
    key_receipts = [
        (
            _one_path(
                run_root / "02-causal-loop" / "flagship" / "baseline",
                "simulation_receipt.json",
            ),
            "baseline_simulation_receipt.json",
        ),
        (
            _one_path(
                run_root / "02-causal-loop" / "flagship" / "recovery",
                "simulation_receipt.json",
            ),
            "recovery_simulation_receipt.json",
        ),
        *[
            (path, path.name)
            for path in sorted((run_root / "09-registry" / "receipts").glob("*.json"))
        ],
    ]
    for path, name in key_receipts:
        shutil.copy2(path, receipts_root / name)
    embedded = {
        "report": report,
        "baseline": baseline,
        "recovery": recovery,
    }
    (output / "evidence.html").write_text(
        render_evolution_arena_html(embedded),
        encoding="utf-8",
    )
    dashboard_command = (
        f"ROSCLAW_EVOLUTION_ARENA_REPORT={report_path} "
        "rosclaw dashboard --host 127.0.0.1 --port 8765\n"
        "open http://127.0.0.1:8765/evolution-arena\n"
    )
    (output / "dashboard-command.txt").write_text(
        dashboard_command,
        encoding="utf-8",
    )
    hashes = _hash_tree(output)
    manifest = {
        "schema_version": "rosclaw.evolution_showcase.v1",
        "evolution_id": report["final_proof_bundle"]["run_id"],
        "candidate_hash": report["candidate_hash"],
        "dataset_snapshot_hash": report["dataset_snapshot_hash"],
        "proof_bundle_hash": report["final_proof_bundle"]["bundle_hash"],
        "complete_trace_video": True,
        "before_after_split_screen": True,
        "files": hashes,
    }
    _atomic_json(output / "hashes.json", manifest)
    return {
        **manifest,
        "output_dir": str(output),
        "evidence_html": str(output / "evidence.html"),
    }


def load_dashboard_report_from_environment() -> dict[str, Any]:
    value = os.environ.get("ROSCLAW_EVOLUTION_ARENA_REPORT")
    if not value:
        return {
            "schema_version": "rosclaw.evolution_arena.unconfigured.v1",
            "configured": False,
            "message": "Set ROSCLAW_EVOLUTION_ARENA_REPORT to a Phase 3 run report.",
        }
    report = _load_phase3_report(Path(value))
    return {"configured": True, **report}


def render_evolution_arena_html(embedded: dict[str, Any] | None = None) -> str:
    payload = (
        json.dumps(embedded, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        if embedded is not None
        else "null"
    )
    source = (
        "window.__ARENA__"
        if embedded is not None
        else "await fetch('/api/evolution-arena').then(r=>r.json())"
    )
    return _ARENA_HTML.replace("__EMBEDDED_PAYLOAD__", payload).replace(
        "__DATA_SOURCE__",
        source,
    )


def _render_videos(
    *,
    baseline: dict[str, Any],
    recovery: dict[str, Any],
    output: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    with tempfile.TemporaryDirectory(prefix="rosclaw-arena-frames-") as temporary:
        root = Path(temporary)
        before_root = root / "before"
        after_root = root / "after"
        split_root = root / "split"
        for item in (before_root, after_root, split_root):
            item.mkdir()
        before_states = list(baseline["states"])
        after_states = list(recovery["states"])
        frame_count = max(len(before_states), len(after_states))
        for index in range(frame_count):
            before = before_states[min(index, len(before_states) - 1)]
            after = after_states[min(index, len(after_states) - 1)]
            before_image = _draw_episode(
                Image,
                ImageDraw,
                font,
                state=before,
                result=baseline["result"],
                title="BEFORE · fixed baseline",
                accent=(239, 68, 68),
            )
            after_image = _draw_episode(
                Image,
                ImageDraw,
                font,
                state=after,
                result=recovery["result"],
                title="AFTER · Memory + How",
                accent=(45, 212, 191),
            )
            before_image.save(before_root / f"frame_{index:05d}.png")
            after_image.save(after_root / f"frame_{index:05d}.png")
            split = Image.new("RGB", (1920, 540), (7, 11, 20))
            split.paste(before_image, (0, 0))
            split.paste(after_image, (960, 0))
            split.save(split_root / f"frame_{index:05d}.png")
        _encode_video(before_root, output / "before.mp4")
        _encode_video(after_root, output / "after.mp4")
        _encode_video(split_root, output / "split-screen.mp4")


def _draw_episode(
    image_module: Any,
    draw_module: Any,
    font: Any,
    *,
    state: dict[str, Any],
    result: dict[str, Any],
    title: str,
    accent: tuple[int, int, int],
) -> Any:
    image = image_module.new("RGB", (960, 540), (7, 11, 20))
    draw = draw_module.Draw(image)
    draw.rounded_rectangle((34, 28, 926, 512), radius=22, fill=(14, 22, 38))
    draw.text((60, 52), title, font=font, fill=accent)
    draw.text(
        (60, 82),
        f"t={float(state['time_sec']):.3f}s · force={float(state['contact_force_n']):.2f}N "
        f"· velocity={float(state['object_velocity_mps']):.3f}m/s",
        font=font,
        fill=(173, 188, 211),
    )
    x0, x1 = 100, 860
    floor_y = 350
    draw.line((x0, floor_y, x1, floor_y), fill=(91, 110, 140), width=4)
    scale = (x1 - x0) / 0.65
    target = float(result["target_x_m"])
    tolerance = float(result["target_tolerance_m"])
    target_left = x0 + int((target - tolerance + 0.12) * scale)
    target_right = x0 + int((target + tolerance + 0.12) * scale)
    draw.rounded_rectangle(
        (target_left, floor_y - 82, target_right, floor_y + 6),
        radius=8,
        outline=(34, 197, 94),
        width=4,
    )
    draw.text((target_left, floor_y + 20), "target", font=font, fill=(74, 222, 128))
    object_x = x0 + int((float(state["object_x_m"]) + 0.12) * scale)
    pusher_x = x0 + int((float(state["pusher_x_m"]) + 0.12) * scale)
    draw.rounded_rectangle(
        (object_x - 30, floor_y - 60, object_x + 30, floor_y),
        radius=8,
        fill=(245, 158, 11),
    )
    draw.rounded_rectangle(
        (pusher_x - 18, floor_y - 90, pusher_x + 18, floor_y),
        radius=7,
        fill=accent,
    )
    status = str(result["status"])
    status_color = (34, 197, 94) if status == "SUCCESS" else (239, 68, 68)
    draw.text((60, 450), status, font=font, fill=status_color)
    draw.text(
        (250, 450),
        f"final error={float(result['final_error_m']):+.4f}m · "
        f"peak force={float(result['peak_contact_force_n']):.2f}N",
        font=font,
        fill=(226, 232, 240),
    )
    return image


def _encode_video(frames: Path, output: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        "30",
        "-i",
        str(frames / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError("ffmpeg showcase export failed: " + completed.stdout[-2000:])


def _load_phase3_report(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if value.get("schema_version") != "rosclaw.contact_push_phase3_run.v1":
        raise ValueError("Evolution Arena requires a ContactPush Phase 3 report")
    required = (
        "candidate_hash",
        "dataset_snapshot_hash",
        "final_proof_bundle",
        "activation",
    )
    if any(name not in value for name in required):
        raise ValueError("Phase 3 report is incomplete")
    return value


def _one_path(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {filename} below {root}, found {len(matches)}")
    return matches[0]


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _hash_bytes(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    }


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


_ARENA_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROSClaw Failure-to-Success Arena</title>
<style>
:root{color-scheme:dark;--bg:#070b14;--panel:#0e1626;--line:#263653;--text:#e8eef8;--muted:#91a2bd;--good:#2dd4bf;--bad:#fb7185;--warn:#fbbf24}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 8% 0,#17233c 0,var(--bg) 40%);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui}
header{padding:28px 34px 18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:20px;align-items:flex-end}.eyebrow{letter-spacing:.18em;color:var(--good);font-size:11px}.title{font-size:28px;font-weight:700}.hash{font:11px ui-monospace;color:var(--muted);max-width:480px;overflow-wrap:anywhere}
main{padding:24px 34px 40px;display:grid;grid-template-columns:1.15fr 1fr .9fr;gap:18px}.panel{background:linear-gradient(145deg,#111b2d,#0b1220);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 18px 45px #0005}.wide{grid-column:1/-1}.span2{grid-column:span 2}h2{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#a9bad4;margin:0 0 15px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{padding:13px;border:1px solid #263b5e;border-radius:12px}.metric b{font-size:24px;display:block}.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.flow{display:flex;gap:7px;flex-wrap:wrap}.node{border:1px solid #2b4268;border-radius:999px;padding:7px 10px}.node.e5{border-color:#2dd4bf;color:#5eead4}.node.e4{border-color:#60a5fa;color:#93c5fd}
.bar{height:12px;border-radius:8px;background:#1b2942;overflow:hidden;margin:7px 0 12px}.bar i{display:block;height:100%;background:linear-gradient(90deg,#22c55e,#2dd4bf)}
pre{white-space:pre-wrap;word-break:break-word;background:#08101d;border-radius:10px;padding:12px;color:#b9c8dd;max-height:340px;overflow:auto}video{width:100%;border-radius:12px;background:#030712}.timeline{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}.event{border-left:3px solid #4f6b97;padding:8px;background:#0a1323;border-radius:6px}.event.goodline{border-color:var(--good)}.event.badline{border-color:var(--bad)}
@media(max-width:1100px){main{grid-template-columns:1fr 1fr}.span2,.wide{grid-column:1/-1}}@media(max-width:700px){main{grid-template-columns:1fr;padding:16px}.panel,.span2,.wide{grid-column:1}.metrics{grid-template-columns:1fr 1fr}header{padding:20px 16px}}
</style></head><body><header><div><div class="eyebrow">PHYSICAL AI · CAUSAL EVIDENCE</div><div class="title">ROSClaw Failure-to-Success Arena</div></div><div id="identity" class="hash">loading…</div></header>
<main><section class="panel span2"><h2>Before / After complete trace</h2><video controls loop muted src="split-screen.mp4"></video></section>
<section class="panel"><h2>Promotion status</h2><div id="status"></div></section>
<section class="panel wide"><h2>Core outcomes</h2><div id="metrics" class="metrics"></div></section>
<section class="panel span2"><h2>Causal module chain</h2><div id="proofs" class="flow"></div></section>
<section class="panel"><h2>Failure → intervention</h2><div id="failure"></div></section>
<section class="panel"><h2>Validation / Hidden Holdout</h2><div id="evaluation"></div></section>
<section class="panel"><h2>Four-GPU stress</h2><div id="stress"></div></section>
<section class="panel"><h2>Canary / rollback</h2><div id="canary"></div></section>
<section class="panel wide"><h2>Episode timeline</h2><div id="timeline" class="timeline"></div></section>
<section class="panel wide"><h2>Evidence identity</h2><pre id="evidence"></pre></section></main>
<script>window.__ARENA__=__EMBEDDED_PAYLOAD__;
(async()=>{const d=__DATA_SOURCE__;const r=d.report||d;if(!r.configured&&r.message){document.body.innerHTML='<pre>'+r.message+'</pre>';return}
const q=s=>document.querySelector(s),pct=x=>(100*x).toFixed(1)+'%';q('#identity').textContent=r.candidate_hash+' · '+r.body_snapshot_hash;
q('#status').innerHTML='<b class="good" style="font-size:28px">'+r.candidate_b.promotion.decision+'</b><p>active after rollback</p><div class="hash">'+r.activation.final_active_candidate_hash+'</div>';
q('#metrics').innerHTML=[['Memory attempts',r.memory_attempts.off+' → '+r.memory_attempts.on,'good'],['Validation',pct(r.candidate_b.validation_success.candidate),'good'],['Hidden Holdout',pct(r.candidate_b.holdout_success.candidate),'good'],['Stress worlds',r.four_gpu.worlds,'good']].map(x=>'<div class="metric"><span>'+x[0]+'</span><b class="'+x[2]+'">'+x[1]+'</b></div>').join('');
q('#proofs').innerHTML=r.final_proof_bundle.proofs.map(p=>'<span class="node '+p.level.toLowerCase()+'">'+p.module+' · '+p.level+'</span>').join('');
q('#failure').innerHTML='<b class="bad">'+r.failure_class+'</b><p>Same-seed retry: <span class="good">'+r.same_seed_retry_passed+'</span></p><p>Know invalid candidates reduced: '+r.know.invalid_candidates_reduced+'<br>Safety overrides admitted: '+r.know.safety_override_admitted+'</p>';
q('#evaluation').innerHTML='<div>Baseline → Candidate</div><p>Validation '+pct(r.candidate_b.validation_success.baseline)+' → <b class="good">'+pct(r.candidate_b.validation_success.candidate)+'</b></p><p>Holdout '+pct(r.candidate_b.holdout_success.baseline)+' → <b class="good">'+pct(r.candidate_b.holdout_success.candidate)+'</b></p>';
q('#stress').innerHTML='<b>'+r.four_gpu.worlds+' unique worlds</b><p>GPUs '+r.four_gpu.physical_gpus.join(', ')+'</p><p>critical disagreements <span class="good">'+r.four_gpu.critical_backend_disagreements+'</span><br>force violations <span class="good">'+r.four_gpu.cpu_force_violations+'</span></p>';
const c=r.activation.canary;q('#canary').innerHTML='<b class="bad">Canary '+(c.passed?'PASS':'REGRESSION')+'</b><p>success '+pct(c.success_rate)+' · frozen '+c.frozen+'</p><p class="good">rollback retry '+r.activation.rollback_retry.status+'</p>';
q('#timeline').innerHTML=[['Baseline','OVERSHOOT','badline'],['Practice','Dataset snapshot',''],['Darwin','SIM_CHAMPION','goodline'],['Canary','Regression','badline'],['Rollback','SUCCESS','goodline']].map(x=>'<div class="event '+x[2]+'"><b>'+x[0]+'</b><br>'+x[1]+'</div>').join('');
q('#evidence').textContent=JSON.stringify({dataset:r.dataset_snapshot_hash,candidate:r.candidate_hash,proof:r.final_proof_bundle.bundle_hash,rollback:c.rollback_receipt_hash,ledger:r.activation.ledger_verified},null,2);
})()</script></body></html>"""


__all__ = [
    "export_evolution_showcase",
    "load_dashboard_report_from_environment",
    "render_evolution_arena_html",
]
