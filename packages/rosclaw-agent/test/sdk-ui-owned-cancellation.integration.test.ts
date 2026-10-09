import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { attachOwnedUIAbort } from "../src/ui-owned-cancellation.js";
import { createRosclawExtension, ownedUIState } from "../src/extension/index.js";

// The production adapter calls the production bridge client over this local,
// in-process JSONL fixture. No copied cancellation implementation or hardware.
async function fixture(reply: (request: any) => object = () => ({ ok: true, code: "CANCELLED" })) {
 const home = mkdtempSync(join(tmpdir(), "owned-ui-"));
 mkdirSync(join(home, "run"));
 const requests: any[] = [];
 const server = createServer(socket => {
  let text = "";
  socket.on("data", chunk => {
   text += chunk.toString();
   const line = text.indexOf("\n");
   if (line < 0) return;
   const request = JSON.parse(text.slice(0, line));
   requests.push(request);
   socket.write(JSON.stringify(reply(request)) + "\n");
  });
 });
 await new Promise<void>(resolve => server.listen(join(home, "run", "pi-bridge.sock"), resolve));
 return { home, requests, close: async () => {
  await new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve()));
  rmSync(home, { recursive: true, force: true });
 } };
}
const owner = { mission_id: "mission", session_ref: "session", task_id: "task", operation_id: "operation", turn_id: "turn" };
function modeWith(abort: () => Promise<void> | void) {
 const session = { abort, isStreaming: true };
 const mode = { session, defaultEditor: { handleInput(data: string) { if (data === "\u001b" || data === "\u0003") return session.abort(); } } };
 return mode;
}

test("ESC and Ctrl-C submit exact owner tuple once while preserving original abort", async () => {
 const f = await fixture();
 try {
  let calls = 0;
  const mode = modeWith(async () => { calls++; });
  const binding = attachOwnedUIAbort(mode, { rosclawHome: f.home, current: () => owner, turn: () => "turn" });
  mode.defaultEditor.handleInput("\u001b");
  mode.defaultEditor.handleInput("\u0003");
  assert.deepEqual(await binding.drain(), [{ ok: true, code: "CANCELLED" }]);
  assert.equal(calls, 2);
  assert.equal(f.requests.length, 1);
  assert.equal(f.requests[0].method, "pi.op.cancel_owned");
  assert.deepEqual(Object.fromEntries(Object.entries(f.requests[0].params).filter(([key]) => key !== "token")), owner);
  mode.defaultEditor.handleInput("\u001b");
  assert.deepEqual(await binding.drain(), []);
  assert.equal(calls, 3);
 } finally { await f.close(); }
});

test("provider abort, ordinary input, missing and cross-turn owners never submit UI cancellation", async () => {
 const f = await fixture();
 try {
  let calls = 0;
  let current: typeof owner | undefined = owner;
  let turn = "turn";
  const mode = modeWith(() => { calls++; });
  const binding = attachOwnedUIAbort(mode, { rosclawHome: f.home, current: () => current, turn: () => turn });
  await mode.session.abort();
  mode.defaultEditor.handleInput("a");
  turn = "other"; mode.defaultEditor.handleInput("\u001b");
  turn = "turn"; current = undefined; mode.defaultEditor.handleInput("\u0003");
  assert.deepEqual(await binding.drain(), []);
  assert.equal(f.requests.length, 0);
  assert.equal(calls, 3);
 } finally { await f.close(); }
});

test("late prototype hook is called once with original this and sync/rejected errors propagate", async () => {
 const f = await fixture();
 try {
  let original = 0, late = 0;
  const mode = modeWith(() => { original++; });
  const binding = attachOwnedUIAbort(mode, { rosclawHome: f.home, current: () => owner, turn: () => owner.turn_id });
  const proto: { abort(this: typeof mode.session): void | Promise<void> } = { abort() { assert.equal(this, mode.session); late++; } };
  Object.setPrototypeOf(mode.session, proto);
  mode.defaultEditor.handleInput("\u001b");
  assert.equal(original, 0); assert.equal(late, 1);
  assert.deepEqual(await binding.drain(), [{ ok: true, code: "CANCELLED" }]);
  const sync = new Error("sync abort");
  proto.abort = () => { throw sync; };
  assert.throws(() => mode.defaultEditor.handleInput("\u0003"), error => error === sync);
  const rejected = new Error("rejected abort");
  proto.abort = () => Promise.reject(rejected);
  await assert.rejects(mode.session.abort() as Promise<void>, error => error === rejected);
 } finally { await f.close(); }
});

test("already completed exact operation is quiescent, not reported as cancelled", async () => {
 const f = await fixture(() => ({ ok: true, code: "ALREADY_SUCCEEDED", operations_cancelled: 0 }));
 try {
  const mode = modeWith(() => {});
  const binding = attachOwnedUIAbort(mode, { rosclawHome: f.home, current: () => owner, turn: () => owner.turn_id });
  mode.defaultEditor.handleInput("\u001b");
  assert.deepEqual(await binding.drain(), [{ ok: true, code: "ALREADY_SUCCEEDED" }]);
  assert.equal(f.requests.length, 1);
  assert.equal(f.requests[0].method, "pi.op.cancel_owned");
  assert.deepEqual(Object.fromEntries(Object.entries(f.requests[0].params).filter(([key]) => key !== "token")), owner);
 } finally { await f.close(); }
});

// Exercise the registered original extension callback, not a reimplementation of
// its abort classifier. These messages can be emitted after editor abort, provider
// failure, or programmatic watchdog abort; none is evidence of editor input.
test("original message_end cannot escalate assistant abort into broad or duplicate owned cancellation", async () => {
 const f = await fixture();
 try {
  const handlers: Array<(event: { message: { role: string; stopReason: string; errorMessage?: string } }, ctx: { hasUI: boolean; isIdle: () => boolean; ui: { notify: (text: string) => void } }) => Promise<unknown>> = [];
  const calls: string[] = [];
  const notifications: string[] = [];
  const active = { current: { missionId: "mission", sessionId: "session", mode: "SIMULATION" } };
  const center = {
   call: async (method: string) => { calls.push(method); if (method === "pi.session.interrupt" || method === "pi.op.cancel_owned") throw new Error(`unexpected ${method}`); return { ok: true }; },
   snapshot: () => ({}), noteWorkspace: () => {}, noteProviderOk: () => {}, noteProviderPaused: () => {},
  };
  const pi = {
   on: (name: string, cb: (typeof handlers)[number]) => { if (name === "message_end") handlers.push(cb); },
   registerCommand: () => {}, registerShortcut: () => {}, registerTool: () => {},
   registerMessageRenderer: () => {}, registerEntryRenderer: () => {}, appendEntry: () => {},
  };
  createRosclawExtension({
   profile: "developer", version: "test", systemPrompt: "", rosclawHome: f.home,
   active, center, locale: { effective: "en" }, coordinator: {},
   taskContext: { workspaceRoot: f.home, workspaceSource: "default" },
  } as never)(pi as never);
  // Select the real provider/error callback among the registered message_end
  // handlers; call it with the same event shape as the SDK does.
  const callback = handlers.find(cb => cb.toString().includes("classifyAssistantFailure"));
  assert.ok(callback, "production provider message_end callback registered");
  const state = ownedUIState(f.home);
  const mode = modeWith(() => {});
  const binding = attachOwnedUIAbort(mode, {
   rosclawHome: f.home, current: () => state.owners.get("session"),
   turn: () => state.turns.get("session"),
  });
  const ctx = { hasUI: true, isIdle: () => false, ui: { notify: (text: string) => notifications.push(text) } };
  // This exercises the original message_end handler only. No session_start
  // initializes its UI context here, so provider UI notifications are unproven.
  for (const current of [owner, { ...owner, task_id: "stale" }, undefined]) {
   state.owners.clear(); state.turns.clear();
   if (current) { state.owners.set("session", current); state.turns.set("session", current.turn_id); }
   await callback({ message: { role: "assistant", stopReason: "aborted" } }, ctx);
   await callback({ message: { role: "assistant", stopReason: "error", errorMessage: "model request cancelled" } }, ctx);
   await callback({ message: { role: "user", stopReason: "aborted" } }, ctx);
   assert.deepEqual(await binding.drain(), []);
  }
  state.owners.set("session", owner); state.turns.set("session", owner.turn_id);
  mode.defaultEditor.handleInput("\u001b");
  assert.deepEqual(await binding.drain(), [{ ok: true, code: "CANCELLED" }]);
  await callback({ message: { role: "assistant", stopReason: "aborted" } }, ctx);
  await callback({ message: { role: "assistant", stopReason: "error", errorMessage: "model request cancelled" } }, ctx);
  assert.deepEqual(await binding.drain(), []);
  assert.deepEqual(f.requests.map(r => r.method), ["pi.op.cancel_owned"]);
  assert.ok(!calls.includes("pi.session.interrupt"));
  assert.ok(!notifications.some(text => text.includes("后台操作已停")));
 } finally { await f.close(); }
});

test("foreign, conflicting, delayed and changed ownership never confirm a cancellation", async () => {
 for (const response of [{ ok: false, code: "CALLER_MISMATCH" }, { ok: false, code: "OWNERSHIP_STALE" }, { ok: false, code: "OWNERSHIP_MISMATCH" }, { ok: true, code: "CANCELLED" }]) {
  const f = await fixture(() => response);
  try {
   let current = owner;
   const mode = modeWith(() => {});
   const binding = attachOwnedUIAbort(mode, { rosclawHome: f.home, current: () => current, turn: () => current.turn_id });
   mode.defaultEditor.handleInput("\u001b");
   if (response.ok) current = { ...owner, turn_id: "later" };
   const results = await binding.drain();
   assert.equal(results.length, 1);
   assert.equal(results[0].ok, false);
   assert.equal(f.requests.length <= 1, true);
  } finally { await f.close(); }
 }
});
