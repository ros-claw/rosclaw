"""Self-contained lightweight ROSClaw Trace dashboard page."""

TRACE_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROSClaw Trace</title>
<style>
:root { color-scheme: dark; --bg:#080b12; --panel:#101622; --line:#253047;
  --text:#e8eefb; --muted:#8fa0ba; --ok:#42d392; --err:#ff657a; --warn:#ffc857; }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--text);
  font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; }
header { height:58px; padding:0 18px; border-bottom:1px solid var(--line); display:flex;
  align-items:center; justify-content:space-between; background:#0c111b; }
h1 { font-size:18px; margin:0 } button { color:var(--text); background:#182237;
  border:1px solid #344564; border-radius:6px; padding:7px 11px; cursor:pointer; }
.layout { display:grid; grid-template-columns:280px minmax(420px,1fr) 390px;
  height:calc(100vh - 58px); }
.panel { overflow:auto; border-right:1px solid var(--line); padding:14px; }
.panel:last-child { border-right:0 }.hint,.meta { color:var(--muted); font-size:12px }
.trace { padding:10px; margin-bottom:8px; border:1px solid var(--line); border-radius:7px;
  cursor:pointer; background:var(--panel) }.trace:hover,.trace.active { border-color:#6889c6 }
.trace-id { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:5px }
.badge { font-size:11px; padding:2px 6px; border-radius:9px; background:#26344c }
.OK { color:var(--ok) }.ERROR { color:var(--err) }.BLOCKED { color:var(--warn) }
.timeline { position:relative; min-width:560px }.track { display:grid; grid-template-columns:115px 1fr;
  min-height:40px; border-bottom:1px solid #182033 }.track-name { color:var(--muted); padding:11px 8px }
.lane { position:relative }.span { position:absolute; top:8px; height:24px; min-width:3px;
  border-radius:4px; padding:3px 6px; overflow:hidden; white-space:nowrap; cursor:pointer;
  color:#071019; background:#68a6ff; font-size:11px }.span.ERROR { background:var(--err) }
.span.BLOCKED { background:var(--warn) }.span.OK { background:var(--ok) }
.tree { margin-top:18px }.tree-row { padding:5px 7px; border-left:2px solid #344564;
  margin:2px 0; cursor:pointer }.tree-row:hover { background:#182237 }
pre { white-space:pre-wrap; word-break:break-word; background:#090d15; border:1px solid var(--line);
  padding:10px; border-radius:6px; max-height:36vh; overflow:auto; }
.section { margin-bottom:14px }.section h2 { font-size:13px; color:#b9c9e5; margin:0 0 7px }
@media(max-width:1050px){.layout{grid-template-columns:240px 1fr}.inspector{display:none}}
</style>
</head>
<body>
<header><div><h1>ROSClaw Trace</h1><div class="hint">Physical intelligence decision & runtime evidence</div></div>
<button id="refresh">Refresh</button></header>
<main class="layout">
  <aside class="panel"><div id="trace-list"><span class="hint">No traces yet.</span></div></aside>
  <section class="panel"><div id="summary" class="meta">Select a trace.</div>
    <div id="timeline" class="timeline"></div><div id="tree" class="tree"></div></section>
  <aside class="panel inspector"><div id="inspector"><span class="hint">Select a span to inspect redacted I/O.</span></div></aside>
</main>
<script>
const q = s => document.querySelector(s); let selectedTrace = null;
const el = (tag, cls, text) => { const n=document.createElement(tag); if(cls)n.className=cls;
  if(text!==undefined)n.textContent=text; return n; };
const fmtMs = n => `${Number(n||0).toFixed(1)} ms`;
async function loadList(){
  const data=await fetch('/api/traces?limit=100').then(r=>r.json()); const root=q('#trace-list'); root.replaceChildren();
  if(!data.traces.length){root.append(el('span','hint','No completed traces yet.'));return;}
  data.traces.forEach(t=>{const card=el('div','trace'+(t.trace_id===selectedTrace?' active':''));
    card.onclick=()=>loadTrace(t.trace_id); card.append(el('div','trace-id',t.trace_id));
    const m=el('div','meta'); m.append(el('span','badge '+t.status,t.status));
    m.append(document.createTextNode(`  ${t.span_count} spans · ${fmtMs(t.duration_ms)}`)); card.append(m); root.append(card);});
}
async function loadTrace(id){selectedTrace=id; const data=await fetch('/api/traces/'+encodeURIComponent(id)).then(r=>r.json());
  q('#summary').textContent=`${id} · ${data.span_count} spans`; renderTimeline(data.spans); renderTree(data.tree); loadList();}
function renderTimeline(spans){const root=q('#timeline');root.replaceChildren();if(!spans.length)return;
  const start=Math.min(...spans.map(s=>s.started_at)), end=Math.max(...spans.map(s=>s.ended_at||s.started_at));
  const total=Math.max(end-start,.001); const groups=Object.groupBy?Object.groupBy(spans,s=>s.span_kind):spans.reduce((a,s)=>((a[s.span_kind]??=[]).push(s),a),{});
  Object.entries(groups).forEach(([kind,items])=>{const track=el('div','track'),name=el('div','track-name',kind),lane=el('div','lane');
    items.forEach(s=>{const bar=el('div','span '+s.status,s.name);bar.style.left=`${(s.started_at-start)/total*100}%`;
      bar.style.width=`${Math.max(1,(s.duration_ms||0)/(total*10))}%`;bar.title=`${s.name} · ${fmtMs(s.duration_ms)}`;bar.onclick=()=>inspect(s);lane.append(bar);});
    track.append(name,lane);root.append(track);});}
function renderTree(roots){const root=q('#tree');root.replaceChildren(el('h3','', 'Span tree'));
  const walk=(nodes,depth)=>nodes.forEach(s=>{const row=el('div','tree-row',`${s.span_kind}  ${s.name}  ${fmtMs(s.duration_ms)}`);
    row.style.marginLeft=`${depth*16}px`;row.onclick=()=>inspect(s);root.append(row);walk(s.children||[],depth+1);});walk(roots,0);}
function inspect(s){const root=q('#inspector');root.replaceChildren();const title=el('div','section');
  title.append(el('h2','',s.name),el('div','meta',`${s.span_kind} · ${s.status} · ${fmtMs(s.duration_ms)}`));root.append(title);
  [['Attributes',s.attributes],['Input (redacted)',s.input],['Output (redacted)',s.output],['Error',s.error],['Evidence',s.evidence_refs]].forEach(([name,value])=>{
    if(value===null||value===undefined)return;const sec=el('div','section');sec.append(el('h2','',name),el('pre','',JSON.stringify(value,null,2)));root.append(sec);});}
q('#refresh').onclick=()=>{loadList();if(selectedTrace)loadTrace(selectedTrace)};
const initialTrace=new URLSearchParams(location.search).get('trace_id');
if(initialTrace)loadTrace(initialTrace);else loadList(); setInterval(loadList,3000);
</script>
</body></html>"""
