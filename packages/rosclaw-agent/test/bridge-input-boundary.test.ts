import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer, type Socket } from "node:net";
import test from "node:test";
import { bridgeCall } from "../src/bridge/bridge-client.js";
import { createRosclawExtension } from "../src/extension/index.js";
import { runPiPrint } from "../src/harness/pi/pi-sessions.js";
import { InputController } from "../src/native/input-controller.js";

type Envelope = Record<string, unknown>;
type Call = (method: string, params: Envelope) => Promise<Envelope>;
type Hook = (event: any, ctx: any) => Promise<any>;
type Message = { role?: string; customType?: string; content?: unknown; [key: string]: unknown };
type Event = { type: string; message?: Message };

// Offline host fixture, not an AgentSession/provider or a replacement print
// implementation. The real extension factory, InputController, bridge client,
// repo runPiPrint and pinned upstream runPrintMode all execute unchanged.
function fixture(home: string, call: Call, prior: Message[] = [], idle = true) {
 const hooks = new Map<string, Hook[]>(), commands = new Map<string, any>();
 const listeners = new Set<(event: Event) => void>();
 const notices: string[] = [], sent: Array<{ message: Message; options: Envelope }> = [];
 let deliveries = 0, aborts = 0, disposals = 0, bound = false;
 const ctx = {
  hasUI: false, isIdle: () => idle, abort: () => { aborts++; },
  ui: { notify: (text: string) => notices.push(text), setWidget() {}, setWorkingMessage() {} },
  sessionManager: { getSessionId: () => "offline-session" },
 };
 const emit = async (name: string, event: unknown) => {
  for (const hook of hooks.get(name) ?? []) await hook(event, ctx);
 };
 const session = {
  state: { messages: [...prior] },
  subscribe(listener: (event: Event) => void) {
   listeners.add(listener); return () => { listeners.delete(listener); };
  },
  async bindExtensions() {
   if (!bound) { bound = true; await emit("session_start", { reason: "startup" }); }
  },
  async prompt(text: string) {
   // Commands are intentionally handled before input, as in the public host.
   if (text.startsWith("/")) {
    const [name, ...args] = text.slice(1).split(" ");
    const command = commands.get(name); assert.ok(command, `missing command ${name}`);
    await command.handler(args.join(" "), ctx); return;
   }
   for (const hook of hooks.get("input") ?? []) {
    const result = await hook({ text, source: "interactive" }, ctx);
    if (result?.action === "handled") return;
   }
   // Count the delivery boundary, without invoking or fabricating a model.
   deliveries++;
  },
 };
 const pi = {
  on(name: string, hook: Hook) { hooks.set(name, [...(hooks.get(name) ?? []), hook]); },
  registerCommand(name: string, command: unknown) { commands.set(name, command); },
  registerTool() {}, registerShortcut() {}, registerMessageRenderer() {}, registerEntryRenderer() {},
  appendEntry() {},
  sendMessage(message: Message, options: Envelope) {
   sent.push({ message, options });
   const custom = { ...message, role: "custom" };
   session.state.messages.push(custom);
   for (const listener of listeners) listener({ type: "message_end", message: custom });
  },
 };
 createRosclawExtension({
  profile: "developer", version: "offline", systemPrompt: "offline", rosclawHome: home,
  active: { current: { sessionId: "offline-session", missionId: "offline-mission", mode: "SIMULATION" } },
  coordinator: { setNotify() {} },
  center: { call, noteWorkspace() {} }, locale: { effective: "en" },
  taskContext: { workspaceRoot: home, workspaceSource: "explicit" },
  workspaceStore: { current: undefined, recent: [] },
 } as never)(pi as never);
 const runtime = {
  session, setRebindSession() {},
  async dispose() { disposals++; await emit("session_shutdown", {}); },
 };
 return { runtime, notices, sent, session, listeners,
  deliveries: () => deliveries, aborts: () => aborts, disposals: () => disposals };
}

async function printed(f: ReturnType<typeof fixture>, initialMessage?: string) {
 const stderr: string[] = [], original = console.error;
 console.error = (...args: unknown[]) => { stderr.push(args.map(String).join(" ")); };
 try { return { code: await runPiPrint(f.runtime, { initialMessage }), stderr }; }
 finally { console.error = original; }
}
function temp() {
 const home = mkdtempSync(join(tmpdir(), "bridge-input-"));
 return { home, cleanup: () => rmSync(home, { recursive: true, force: true }) };
}

for (const rejected of ["throw", "false", "missing-ok", "null"] as const) {
 test(`controller ${rejected}: no dispatch, bounded notice, force-new retained`, async () => {
  const calls: string[] = [], notices: string[] = [];
  const controller = new InputController({
   call: async method => {
    calls.push(method);
    if (rejected === "throw") throw new Error("offline failure " + "x".repeat(10000));
    if (rejected === "null") return null as never;
    return rejected === "false" ? { ok: false, error: "store refused" } : {};
   }, missionId: () => "m", sessionRef: () => "s", backendNativeId: () => "s", cwd: () => ".",
   notify: text => notices.push(text),
  });
  controller.forceNewNext = true;
  assert.equal(await controller.persist("do not deliver"), null);
  assert.deepEqual(calls, ["pi.input.persist"]);
  assert.equal(controller.forceNewNext, true);
  assert.equal(notices.length, 1); assert.ok(notices[0].length <= 200);
  assert.match(notices[0], /持久化失败/);
 });
}

test("production extension and print: explicit failure is visible/nonzero, not model delivery", async () => {
 const t = temp();
 try {
  const calls: string[] = [];
  const f = fixture(t.home, async method => {
   calls.push(method); return method === "pi.input.persist" ? { ok: false, error: "offline store refusal" } : { ok: true };
  });
  const result = await printed(f, "a user input that must not reach a model");
  assert.equal(result.code, 1); assert.match(result.stderr.join("\n"), /INPUT_PERSIST_FAILED/);
  assert.equal(f.deliveries(), 0); assert.ok(!calls.includes("pi.input.dispatched"));
  assert.equal(f.sent.length, 1); assert.equal(f.sent[0].message.customType, "rosclaw.input_error");
  assert.equal(f.sent[0].message.display, true); assert.equal(f.sent[0].options.triggerTurn, false);
  assert.ok(String(f.sent[0].message.content).length <= 200);
  assert.ok(!String(f.sent[0].message.content).includes("a user input"));
  assert.equal(f.disposals(), 1); assert.equal(f.listeners.size, 0);
 } finally { t.cleanup(); }
});

// Real offline AF_UNIX client failures; Python tests separately exercise the
// actual production PiBridgeServer (not this protocol fixture).
for (const behavior of ["reject", "disconnect", "unavailable"] as const) {
 test(`public print via production bridge client: ${behavior} cannot silently succeed`, { timeout: 12000 }, async () => {
  const t = temp(), sockets = new Set<Socket>();
  const received: string[] = [];
  mkdirSync(join(t.home, "run"));
  const server = createServer(conn => {
   sockets.add(conn); conn.on("close", () => sockets.delete(conn));
   let pending = "";
   conn.on("data", chunk => {
    pending += chunk.toString();
    const end = pending.indexOf("\n"); if (end < 0) return;
    const request = JSON.parse(pending.slice(0, end)); pending = pending.slice(end + 1);
    received.push(request.method);
    if (request.method === "pi.input.persist") {
     if (behavior === "disconnect") { conn.end(); return; }
     conn.end(JSON.stringify({ ok: false, error: "request too large" }) + "\n"); return;
    }
    conn.end('{"ok":true}\n');
   });
  });
  try {
   if (behavior !== "unavailable") await new Promise<void>(resolve => server.listen(join(t.home, "run/pi-bridge.sock"), resolve));
   const f = fixture(t.home, (method, params) => bridgeCall(t.home, method, params));
   const result = await printed(f, "offline transport input");
   assert.equal(result.code, 1); assert.equal(f.deliveries(), 0);
   assert.match(result.stderr.join("\n"), /INPUT_PERSIST_FAILED/);
   assert.ok(result.stderr.join("\n").length <= 200);
   assert.ok(!received.includes("pi.input.dispatched")); assert.equal(f.listeners.size, 0);
   if (behavior !== "unavailable") assert.ok(received.includes("pi.input.persist"));
  } finally {
   for (const conn of sockets) conn.destroy();
   if (server.listening) await new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve()));
   t.cleanup();
  }
 });
}

test("successful persistence precedes the delivery boundary and dispatch", async () => {
 const t = temp(), calls: string[] = [];
 try {
  const f = fixture(t.home, async (method, params) => {
   calls.push(method);
   if (method === "pi.input.persist") { assert.equal(params.text, "hello"); return { ok: true, input: { input_id: "i" } }; }
   return { ok: true };
  });
  const result = await printed(f, "hello");
  assert.equal(result.code, 0); assert.equal(f.deliveries(), 1); assert.deepEqual(result.stderr, []);
  assert.ok(calls.indexOf("pi.input.persist") < calls.indexOf("pi.input.dispatched"));
  assert.equal(f.sent.length, 0); assert.equal(f.listeners.size, 0);
 } finally { t.cleanup(); }
});

for (const owner of ["TASK_ROUTER", "EXPLAIN_HANDLER", "legacy-auto-task", "nl-stop"] as const) {
 test(`legitimate handled ${owner}: no model, no input error, print remains zero`, async () => {
  const t = temp(), calls: string[] = [];
  try {
   const persisted: Envelope = { ok: true, input: { input_id: "i" } };
   if (owner === "legacy-auto-task") persisted.auto_task = { task_id: "t" };
   else persisted.turn_disposition = {
    owner: owner === "nl-stop" ? "TASK_ROUTER" : owner, suppress_model_turn: true,
    ...(owner === "nl-stop" ? { cancel_report: { operations_cancelled: 2, task_cancelled: true } } : {}),
   };
   if (owner === "EXPLAIN_HANDLER") persisted.explain = { task_id: "t", goal: "prior task", state: "RUNNING" };
   const f = fixture(t.home, async method => { calls.push(method); return method === "pi.input.persist" ? persisted : { ok: true }; }, [], owner !== "nl-stop");
   const result = await printed(f, owner === "nl-stop" ? "stop" : "known deterministic input");
   assert.equal(result.code, 0); assert.equal(f.deliveries(), 0); assert.deepEqual(result.stderr, []);
   assert.ok(calls.includes("pi.input.dispatched"));
   assert.equal(f.sent.length, 1); assert.notEqual(f.sent[0].message.customType, "rosclaw.input_error");
   assert.equal(f.sent[0].options.triggerTurn, false);
   if (owner === "nl-stop") { assert.equal(f.aborts(), 1); assert.match(f.notices.join("\n"), /已停止/); }
   assert.equal(f.listeners.size, 0);
  } finally { t.cleanup(); }
 });
}

test("legitimate registered command and blank print do not need an assistant or persistence", async () => {
 const t = temp();
 try {
  for (const message of ["/newtask", undefined]) {
   const calls: string[] = [];
   const f = fixture(t.home, async method => { calls.push(method); throw new Error("must not persist a command"); });
   const result = await printed(f, message);
   assert.equal(result.code, 0); assert.equal(f.deliveries(), 0);
   assert.ok(!calls.includes("pi.input.persist")); assert.deepEqual(result.stderr, []);
   assert.equal(f.listeners.size, 0);
  }
 } finally { t.cleanup(); }
});

test("past failure is not sticky; upstream assistant cancellation remains nonzero/distinct", async () => {
 const t = temp();
 try {
  const past = fixture(t.home, async () => ({ ok: true }), [{ role: "custom", customType: "rosclaw.input_error" }]);
  assert.equal((await printed(past)).code, 0);
  const cancelled = fixture(t.home, async () => ({ ok: true }), [{ role: "assistant", stopReason: "aborted", errorMessage: "Request cancelled", content: [] }]);
  const result = await printed(cancelled);
  assert.equal(result.code, 1); assert.match(result.stderr.join("\n"), /Request cancelled/);
  assert.doesNotMatch(result.stderr.join("\n"), /INPUT_PERSIST_FAILED/);
  assert.equal(cancelled.listeners.size, 0);
 } finally { t.cleanup(); }
});
