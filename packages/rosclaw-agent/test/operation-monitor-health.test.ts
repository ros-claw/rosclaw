import test from "node:test";
import assert from "node:assert/strict";
import { OperationWatcher } from "../src/native/operation-watcher.js";

function fixture() {
 let time = 0;
 let fail = false;
 let idle = true;
 let revision = 1;
 let events: Record<string, unknown>[] = [];
 let rpc: ((method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>) | undefined;
 const calls: { method: string; params: Record<string, unknown> }[] = [];
 const widgets = new Map<string, string[]>();
 const updates: { key: string; lines: string[] | undefined }[] = [];
 const messages: unknown[] = [];
 const notices: string[] = [];
 const watcher = new OperationWatcher({
  now: () => time,
  sink: () => ({ isIdle: idle, api: { sendMessage: (...args) => { messages.push(args); } },
   notify: text => { notices.push(text); },
   setWidget: (key, lines) => { updates.push({ key, lines }); if (lines) widgets.set(key, lines); else widgets.delete(key); } }),
  call: async (method, params) => {
   calls.push({ method, params });
   if (rpc) return rpc(method, params);
   if (fail && (method === "pi.kernel.events" || method === "pi.op.get")) throw new Error("SECRET https://private.invalid /private/key");
   if (method === "pi.op.get") return { operation: { operation_id: params.operation_id, task_id: "t", revision: 1, state: "RUNNING" } };
   if (method === "pi.kernel.events") return { events };
   if (method === "pi.kernel.get") return { task: { task_id: params.task_id, state: "RUNNING", active_revision: revision } };
   throw new Error("unexpected inert RPC");
  }
 });
 const tick = () => (watcher as unknown as { tick(): Promise<void> }).tick();
 return { watcher, tick, calls, widgets, updates, messages, notices,
  time: (n: number) => { time = n; }, fail: (b: boolean) => { fail = b; }, idle: (b: boolean) => { idle = b; },
  revision: (n: number) => { revision = n; }, events: (e: Record<string, unknown>[]) => { events = e; },
  rpc: (fn: typeof rpc) => { rpc = fn; },
  health: (id = "t") => widgets.get(`monitor:task:${id}`)?.join(" ") ?? "" };
}
const output = (seq: number, text: string, operation_id = "o") => ({ seq, operation_id, event_type: "operation.output", payload: { text } });
function deferred() {
 let resolve!: (v: Record<string, unknown>) => void;
 let reject!: (e: Error) => void;
 const promise = new Promise<Record<string, unknown>>((a, b) => { resolve = a; reject = b; });
 return { promise, resolve, reject };
}

test("monitor_health/sustained_events_failure", async () => {
 const f = fixture(); f.watcher.track("o"); f.events([output(1, "first")]); await f.tick();
 f.fail(true);
 for (let i = 1; i <= 3; i++) { f.time(i * 2000); await f.tick(); assert.match(f.health(), /monitor unavailable/); }
 assert.match(f.health(), /6 seconds ago/);
 assert.deepEqual(f.calls.filter(c => c.method === "pi.kernel.events").map(c => c.params.last_seq), [0, 1, 1, 1]);
 assert.equal(f.messages.length, 0); assert.equal(f.notices.length, 0);
 f.fail(false); f.events([output(2, "second")]); await f.tick(); f.events([]); await f.tick();
 assert.deepEqual(f.updates.filter(u => u.key === "op:o" && u.lines?.includes("⠋ Operation o… second")),
  [{ key: "op:o", lines: ["⠋ Operation o… second"] }]);
 assert.equal(f.calls.at(-1)?.params.last_seq, 2);
});

test("monitor_health/quiet_recovery", async () => {
 const f = fixture(); f.watcher.track("o"); f.events([output(1, "real progress")]); await f.tick();
 f.fail(true); f.time(2000); await f.tick();
 const before = f.widgets.get("op:o"); f.fail(false); f.events([]); f.time(4000); await f.tick();
 assert.match(f.health(), /monitor connected\/no new output/); assert.match(f.health(), /0 seconds ago/);
 assert.deepEqual(f.widgets.get("op:o"), before); assert.equal(f.messages.length, 0);
 f.fail(true); f.time(6000); await f.tick(); assert.match(f.health(), /2 seconds ago/);
});

test("monitor_health/pending_resolution_failure", async () => {
 const f = fixture(); f.watcher.track("o"); f.fail(true);
 for (let i = 0; i < 3; i++) await f.tick();
 assert.match(f.widgets.get("monitor:pending:o")!.join(" "), /monitor unavailable\/no successful observation yet/);
 assert.equal(f.calls.filter(c => c.method === "pi.kernel.events").length, 0);
 f.fail(false); await f.tick(); assert.equal(f.widgets.has("monitor:pending:o"), false);
 assert.match(f.health(), /connected/); assert.equal(f.calls.at(-1)?.params.task_id, "t");
});

test("monitor_health/independent_tasks", async () => {
 const f = fixture(); f.watcher.trackTask("a"); f.watcher.trackTask("b");
 f.rpc(async (_m, p) => { if (p.task_id === "a") throw new Error("private"); return { events: [{ seq: 7, event_type: "ignored" }] }; });
 await f.tick(); f.time(4000); await f.tick();
 assert.match(f.health("a"), /no successful observation yet/); assert.match(f.health("b"), /connected/);
 assert.deepEqual(f.calls.map(c => [c.params.task_id, c.params.last_seq]), [["a", 0], ["b", 0], ["a", 0], ["b", 7]]);
 assert.equal(f.messages.length, 0);
});

test("monitor_health/single_flight_deferred", async () => {
 const f = fixture(); f.watcher.trackTask("t"); const d = deferred(); let active = 0; let max = 0;
 f.rpc(async () => { active++; max = Math.max(max, active); try { return await d.promise; } finally { active--; } });
 f.watcher.start(); const a = f.tick(); await Promise.resolve(); const b = f.tick();
 assert.equal(a, b); assert.equal(f.calls.length, 1);
 d.resolve({ events: [{ seq: 3, event_type: "ignored" }] }); await Promise.all([a, b]);
 assert.equal(max, 1); assert.equal(f.updates.filter(u => u.lines).length, 1);
 f.rpc(undefined); f.fail(true); await f.tick(); assert.match(f.health(), /unavailable/);
 assert.equal(f.calls.at(-1)?.params.last_seq, 3); f.watcher.stop();
});

test("monitor_health/stop_during_await", async () => {
 for (const reject of [false, true]) {
  const f = fixture(); f.watcher.trackTask("t"); await f.tick(); const d = deferred(); f.rpc(async () => d.promise);
  const pending = f.tick(); await Promise.resolve(); f.watcher.stop(); const count = f.updates.length;
  assert.equal(f.widgets.has("monitor:task:t"), false);
  if (reject) d.reject(new Error("SECRET")); else d.resolve({ events: [] });
  await pending; await f.tick(); assert.equal(f.updates.length, count); assert.equal(f.calls.length, 2);
 }
});

test("monitor_health/terminal_proof_regression", async () => {
 const terminal = { seq: 1, operation_id: "o", event_type: "operation.completed", payload: { state: "SUCCEEDED" } };
 const proof = { ok: true, status: "SUCCEEDED", operation: { operation_id: "o", task_id: "t", revision: 1, state: "SUCCEEDED" } };
 const f = fixture(); f.watcher.track("o"); f.idle(false); f.events([terminal]); await f.tick();
 assert.equal(f.messages.length, 0); f.watcher.observeToolResult(proof); f.idle(true); f.events([]); await f.tick();
 assert.equal(f.messages.length, 0);
 const g = fixture(); g.watcher.track("o"); g.idle(false); g.events([terminal]); await g.tick();
 g.watcher.observeToolResult({ ...proof, operation: { ...proof.operation, revision: 2 } });
 g.idle(true); g.events([]); await g.tick(); await g.tick(); assert.equal(g.messages.length, 1);
 const h = fixture(); h.watcher.track("o"); h.revision(2); h.events([terminal]); await h.tick(); await h.tick();
 assert.equal(h.messages.length, 0); assert.equal(h.notices.length, 1);
});

test("monitor_health/redaction_and_bounded_updates", async () => {
 const f = fixture(); f.watcher.trackTask("t"); f.fail(true);
 for (let i = 0; i < 12; i++) { f.time(i * 2000); await f.tick(); }
 assert.equal(f.calls.length, 12); assert.equal(f.updates.length, 12);
 assert.equal(f.messages.length, 0); assert.equal(f.notices.length, 0);
 assert.doesNotMatch(JSON.stringify(f.updates.map(u => [u.key, ...(u.lines ?? [])])), /SECRET|https:|private|key/);
 assert.match(f.health(), /worker state unknown/); f.watcher.stop();
});
